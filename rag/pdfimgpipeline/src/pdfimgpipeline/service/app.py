"""FastAPI 服务层。

端点：
    GET  /health   存活/就绪（LB 探活，不鉴权）
    GET  /info     配置摘要（不泄露密钥）
    POST /route    只做路由决策（便宜），返回 decision/reason/features
    POST /extract  跑完整流水线，返回最终 JSON
    GET  /metrics  Prometheus 文本格式
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import List, Optional

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, Response, UploadFile

from .. import __version__
from ..config import Settings
from ..pipeline import PDFImagePipeline
from ..wiring import build_pipeline
from .metrics import METRICS
from .schemas import BatchItem, BatchResponse, ExtractResponse, HealthResponse, RouteResponse

logger = logging.getLogger(__name__)

log = logging.getLogger("pdfimgpipeline")


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


async def _persist_upload(upload: UploadFile, max_bytes: int) -> str:
    """把上传流式落盘到临时文件（核心流水线以文件路径为输入）。"""
    suffix = os.path.splitext(upload.filename or "")[1] or ".img"
    fd, path = tempfile.mkstemp(prefix="pdfimg_", suffix=suffix)
    size = 0
    try:
        with os.fdopen(fd, "wb") as out:
            while True:
                chunk = await upload.read(1 << 20)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"文件超过上限 {max_bytes // (1024 * 1024)}MB",
                    )
                out.write(chunk)
    except Exception:
        _unlink(path)
        raise
    return path


def _unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


async def _persist_many(files: List[UploadFile], max_bytes: int):
    """批量落盘；任一张失败则回滚已写文件并抛出。"""
    paths: List[str] = []
    names: List[str] = []
    try:
        for f in files:
            names.append(f.filename or f"file{len(names)}")
            paths.append(await _persist_upload(f, max_bytes))
    except Exception:
        for p in paths:
            _unlink(p)
        raise
    return paths, names


def _cleanup(paths: List[str]) -> None:
    for p in paths:
        _unlink(p)


def _batch_response(outcomes, names: List[str], metric_prefix: str) -> BatchResponse:
    items = [
        BatchItem(index=i, filename=names[i], ok=o.ok, data=o.data, error=o.error)
        for i, o in enumerate(outcomes)
    ]
    ok = sum(1 for it in items if it.ok)
    failed = len(items) - ok
    METRICS.inc(f"{metric_prefix}_ok", ok)
    METRICS.inc(f"{metric_prefix}_error", failed)
    return BatchResponse(
        count=len(items), ok_count=ok, failed_count=failed, results=items)


def create_app(
    settings: Optional[Settings] = None,
    *,
    pipeline: Optional[PDFImagePipeline] = None,
) -> FastAPI:
    """应用工厂。可注入 settings 与 pipeline，便于测试。"""
    settings = settings or Settings()
    _configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if pipeline is not None:
            app.state.pipeline = pipeline
            app.state.cache = None
        else:
            app.state.pipeline, app.state.cache = await build_pipeline(settings)
        app.state.settings = settings
        logger.info(
            "服务就绪: cache=%s vlm_model=%s vlm_configured=%s auth=%s",
            getattr(app.state.cache, "name", "injected"),
            settings.vlm_model,
            bool(settings.resolved_vlm_api_key),
            "on" if settings.pdfimg_api_key else "off",
        )
        try:
            yield
        finally:
            cache = app.state.cache
            if cache is not None:
                await cache.aclose()
            logger.info("已优雅退出")

    app = FastAPI(title="pdfimgpipeline", version=__version__, lifespan=lifespan)

    def require_key(x_api_key: Optional[str] = Header(default=None)) -> None:
        if settings.pdfimg_api_key and x_api_key != settings.pdfimg_api_key:
            raise HTTPException(status_code=401, detail="无效或缺失 X-API-Key")

    @app.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        return HealthResponse(
            status="ok",
            version=__version__,
            cache_backend=getattr(request.app.state.cache, "name", "injected"),
            vlm_configured=bool(settings.resolved_vlm_api_key),
            vlm_model=settings.vlm_model,
        )

    @app.get("/info", dependencies=[Depends(require_key)])
    async def info() -> dict:
        return {
            "version": __version__,
            "vlm_model": settings.vlm_model,
            "cache_backend": settings.cache_backend,
            "cache_ttl": settings.cache_ttl,
            "ocr_max_workers": settings.ocr_max_workers,
            "vlm_max_concurrency": settings.vlm_max_concurrency,
            "max_upload_mb": settings.max_upload_mb,
            "max_batch_size": settings.max_batch_size,
            "batch_max_concurrency": settings.batch_max_concurrency,
        }

    @app.post("/route", response_model=RouteResponse,
              dependencies=[Depends(require_key)])
    async def route(request: Request, file: UploadFile = File(...)) -> RouteResponse:
        path = await _persist_upload(file, settings.max_upload_bytes)
        try:
            result = await request.app.state.pipeline.router.decide(path)
        finally:
            _unlink(path)
        return RouteResponse(**asdict(result))

    @app.post("/extract", response_model=ExtractResponse,
              dependencies=[Depends(require_key)])
    async def extract(request: Request, file: UploadFile = File(...)) -> ExtractResponse:
        path = await _persist_upload(file, settings.max_upload_bytes)
        started = time.perf_counter()
        try:
            out = await request.app.state.pipeline.process(path)
            METRICS.inc("extract_ok")
            return ExtractResponse(**out)
        except HTTPException:
            raise
        except Exception as e:  # noqa: BLE001
            METRICS.inc("extract_error")
            logger.exception("extract 失败: %s", e)
            raise HTTPException(status_code=500, detail=f"处理失败: {e}") from e
        finally:
            _unlink(path)
            METRICS.observe("extract_latency", time.perf_counter() - started)

    def _check_batch(files: List[UploadFile]) -> None:
        if not files:
            raise HTTPException(status_code=422, detail="files 不能为空")
        if len(files) > settings.max_batch_size:
            raise HTTPException(
                status_code=413,
                detail=f"批量上限 {settings.max_batch_size} 张，收到 {len(files)} 张",
            )

    @app.post("/route:batch", response_model=BatchResponse,
              dependencies=[Depends(require_key)])
    async def route_batch(
        request: Request, files: List[UploadFile] = File(...)
    ) -> BatchResponse:
        _check_batch(files)
        paths, names = await _persist_many(files, settings.max_upload_bytes)
        started = time.perf_counter()
        try:
            outcomes = await request.app.state.pipeline.router.decide_many(
                paths, max_concurrency=settings.batch_max_concurrency)
        finally:
            _cleanup(paths)
            METRICS.observe("route_batch_latency", time.perf_counter() - started)
        return _batch_response(outcomes, names, "route_batch")

    @app.post("/extract:batch", response_model=BatchResponse,
              dependencies=[Depends(require_key)])
    async def extract_batch(
        request: Request, files: List[UploadFile] = File(...)
    ) -> BatchResponse:
        _check_batch(files)
        paths, names = await _persist_many(files, settings.max_upload_bytes)
        started = time.perf_counter()
        try:
            outcomes = await request.app.state.pipeline.process_many(
                paths, max_concurrency=settings.batch_max_concurrency)
        finally:
            _cleanup(paths)
            METRICS.observe("extract_batch_latency", time.perf_counter() - started)
        return _batch_response(outcomes, names, "extract_batch")

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(
            content=METRICS.render(),
            media_type="text/plain; version=0.0.4",
        )

    return app


app = create_app()


def run() -> None:
    """控制台脚本入口：pdfimgpipeline。"""
    import uvicorn

    s = Settings()
    uvicorn.run(
        "pdfimgpipeline.service.app:app",
        host=s.pdfimg_host,
        port=s.pdfimg_port,
        log_level=s.log_level.lower(),
    )
