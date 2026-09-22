"""顶层 Pipeline：路由 -> 解析 -> 缓存，对外产出统一结构化结果。"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Any, Dict, List, Optional, Sequence

from .cache import ResultCache
from .config import Settings
from .models import BatchOutcome, RouteResult
from .router import PdfRouter
from .utils import file_sha256, short

logger = logging.getLogger(__name__)


class PdfPipeline:
    def __init__(
        self,
        router: PdfRouter,
        parsers: Dict[str, Any],
        result_cache: Optional[ResultCache] = None,
        settings: Optional[Settings] = None,
    ):
        self.router = router
        self.parsers = parsers
        self.result_cache = result_cache
        self.settings = settings

    # ---------------------------------------------------------------- 入口
    def process(
        self,
        pdf_path: str,
        out_dir: str,
        *,
        chunk_limit: int = 600,
        image_describer: Optional[object] = None,
        force: Optional[str] = None,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        pdf_hash = file_sha256(pdf_path)
        route = self.router.decide(pdf_path, force=force, use_cache=use_cache)
        engine = route.decision

        ident = self._result_ident(pdf_hash, engine, out_dir, chunk_limit, image_describer)

        if use_cache and self.result_cache is not None:
            cached = self.result_cache.get(ident)
            if cached:
                cached["from_cache"] = True
                logger.info("[result cache hit] %s -> %s", short(pdf_hash), engine)
                return cached

        parser, engine_used = self._resolve_parser(engine)
        logger.info("解析开始: %s engine=%s", os.path.basename(pdf_path), engine_used)
        result = parser.parse(
            pdf_path,
            out_dir,
            chunk_limit=chunk_limit,
            image_describer=image_describer,
        )

        if engine_used != engine:
            route = RouteResult(
                decision=engine_used,
                kind=route.kind,
                reason=f"{route.reason}|fallback:{engine}->{engine_used}",
                precheck=route.precheck,
                forced=route.forced,
                from_cache=route.from_cache,
            )

        result["engine"] = engine_used
        result["route"] = route.to_dict()
        result["from_cache"] = False

        if use_cache and self.result_cache is not None:
            serializable = dict(result)
            self.result_cache.set(ident, serializable)

        return result

    def process_many(
        self,
        pdf_paths: Sequence[str],
        out_dir: str,
        *,
        chunk_limit: int = 600,
        image_describer: Optional[object] = None,
        force: Optional[str] = None,
        use_cache: bool = True,
    ) -> List[BatchOutcome]:
        """逐篇处理，单篇失败隔离（顺序与输入一致）。"""
        outcomes: List[BatchOutcome] = []
        for pdf_path in pdf_paths:
            try:
                outcomes.append(BatchOutcome(
                    ok=True,
                    data=self.process(
                        pdf_path, out_dir, chunk_limit=chunk_limit,
                        image_describer=image_describer, force=force,
                        use_cache=use_cache,
                    ),
                ))
            except Exception as e:  # noqa: BLE001 - 批量隔离单项失败
                logger.warning("处理失败: %s (%s)", pdf_path, e)
                outcomes.append(BatchOutcome(ok=False, error=str(e)))
        return outcomes

    # ---------------------------------------------------------------- 内部
    def _resolve_parser(self, engine: str):
        parser = self.parsers.get(engine)
        if parser is None:
            raise ValueError(f"没有注册的解析引擎: {engine}")

        available = getattr(parser, "available", None)
        if callable(available) and not available():
            fallback_enabled = bool(
                self.settings and self.settings.mineru_fallback_text
            )
            fallback = self.parsers.get("pymupdf")
            if engine != "pymupdf" and fallback is not None:
                if fallback_enabled:
                    logger.warning("%s 不可用，降级到 pymupdf（结果质量可能下降）", engine)
                    return fallback, "pymupdf"
                raise RuntimeError(
                    f"解析引擎 {engine} 不可用；安装 MinerU / 配置 MINERU_HTTP_URL，"
                    f"或使用 --fallback-text 降级到 pymupdf"
                )
            raise RuntimeError(f"解析引擎 {engine} 不可用")
        return parser, engine

    def _result_ident(
        self,
        pdf_hash: str,
        engine: str,
        out_dir: str,
        chunk_limit: int,
        image_describer: Optional[object],
    ) -> str:
        """结果缓存标识：任一影响产出的参数变化都会换 key。"""
        if image_describer is None:
            desc = "nodesc"
        else:
            fingerprint = getattr(image_describer, "fingerprint", None)
            desc = fingerprint() if callable(fingerprint) else "desc"
        material = "|".join([
            pdf_hash,
            engine,
            os.path.abspath(out_dir),
            str(chunk_limit),
            desc,
        ])
        digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]
        return f"{pdf_hash}:{digest}"
