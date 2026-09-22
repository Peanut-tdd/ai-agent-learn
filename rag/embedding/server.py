"""bge-m3 常驻嵌入服务 (FastAPI) —— 可独立部署、对外开放, 不是 demo。

定位: 模型启动时加载一次、常驻内存, 对外提供 HTTP 接口, 索引/查询共用同一份模型。
本版本补齐了「独立服务」的关键差距, 让它不再依赖"客户端自己攒批":

  1. 服务端自动攒批 (adaptive batching): 并发小请求在队列里合并成一次模型前向,
     结果按请求切回。对"单模型并发前向无收益(并发=1)"的模型, 这是小请求高吞吐
     的唯一正确姿势 —— 调用方随便发单条 query 都高效, 无需自己实现攒批客户端。
     攒批器把模型前向放进单线程 executor, 天然串行、不占事件循环。
  2. 可选 API Key 鉴权 (X-API-Key), 默认关, 内网/代理后可开。
  3. /metrics (Prometheus 文本格式): 请求数 / 延迟 / 攒批统计, 接监控就绪。
  4. 部署配套: 见 deploy/ (systemd / Dockerfile), 多副本 + 负载均衡横向扩容。

启动:  HF_HUB_OFFLINE=1 uvicorn server:app --host 0.0.0.0 --port 8001
接口:  GET  /health   存活/就绪 (LB 探活用, 不鉴权)
       GET  /info     模型信息
       POST /embed    嵌入 (dense / sparse / colbert)
       GET  /metrics  Prometheus 文本格式

⚠️ 生产要点 (详见 Readme.md):
  - 不要 --workers>1 / 多进程: 每进程复制一份 ~2.3GB 模型, 显存被重复占用。
    扩容 = 多起几个实例(副本) + 负载均衡, 而不是加 worker。
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import logging
import os
import time
from collections.abc import Callable
from contextlib import asynccontextmanager

# torch 多线程 tokenizer 与 fastapi 的进程模型会打架, 关掉
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ---------------------------------------------------------------------------
# HF 下载网络策略 (必须在任何 transformers / huggingface_hub 调用前生效):
#   国内直连 huggingface.co 经常被拒; 即使已缓存, transformers 启动时仍会联网
#   做 etag 校验, 连不上就报 httpx.ConnectTimeout 而启动失败。
#   HF_ENDPOINT       默认切到 hf-mirror.com 镜像 (需联网下载/校验时走它)
#   HF_HUB_OFFLINE=1  完全离线, 只用 ~/.cache/huggingface 缓存 (本机模型已缓存, 推荐)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from fastapi import Depends, FastAPI, Header, HTTPException, Response
from pydantic import BaseModel, Field

logger = logging.getLogger("bge-server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

# ---------------------------------------------------------------------------
# 配置: 全部可用环境变量覆盖 (见 .env.example)
# ---------------------------------------------------------------------------
MODEL_NAME      = os.getenv("EMBED_MODEL", "BAAI/bge-m3")   # 本地路径或 HF 模型名
DEVICE          = os.getenv("EMBED_DEVICE", "")             # 空=自动: cuda > mps > cpu
USE_FP16        = os.getenv("EMBED_FP16", "1") == "1"
INFER_BATCH     = int(os.getenv("EMBED_INFER_BATCH", "8"))  # 单次前向批大小(内存峰值决定者)
PASSAGE_MAX_LEN = int(os.getenv("EMBED_MAX_LENGTH", "8192"))     # bge-m3 支持 8192 token
QUERY_MAX_LEN   = int(os.getenv("EMBED_QUERY_MAX_LENGTH", "512"))
MAX_BATCH       = int(os.getenv("EMBED_MAX_BATCH", "128"))       # 单请求文本条数上限
MAX_TEXT_CHARS  = int(os.getenv("EMBED_MAX_TEXT_CHARS", "65536")) # 单条文本字符数上限
BATCH_TARGET    = int(os.getenv("EMBED_BATCH_TARGET", str(4 * INFER_BATCH)))
    # 攒批器攒够多少条文本就发车 (默认 4×INFER_BATCH); 0 = 关闭服务端攒批
BATCH_GRACE_MS  = float(os.getenv("EMBED_BATCH_GRACE_MS", "4"))
    # 队列空了后最多等多少毫秒等"同路人": 单个小请求的额外延迟 ≈ 此值
API_KEY         = os.getenv("EMBED_API_KEY", "").strip()
    # 留空 = 不鉴权(仅限内网/本机); 设置后 /embed /info 需带 X-API-Key 头


# ---------------------------------------------------------------------------
# 模型单例: 启动时加载 + 预热, 前向只在攒批器的单线程 executor 里执行
# ---------------------------------------------------------------------------
class ModelStore:
    """模型单例。启动时加载 + 预热(提前分配/编译, 快速暴露 OOM/下载失败)。"""

    def __init__(self) -> None:
        self.model = None
        self.device: str | None = None
        self.dim: int | None = None
        self._loaded_at: float | None = None

    def load(self) -> None:
        from FlagEmbedding import BGEM3FlagModel

        kwargs: dict = dict(
            model_name_or_path=MODEL_NAME,
            use_fp16=USE_FP16,
            query_max_length=QUERY_MAX_LEN,
            passage_max_length=PASSAGE_MAX_LEN,
            batch_size=INFER_BATCH,
        )
        if DEVICE:
            kwargs["devices"] = DEVICE

        logger.info("加载模型 %s (fp16=%s, device=%s)...", MODEL_NAME, USE_FP16, DEVICE or "auto")
        t0 = time.perf_counter()
        self.model = BGEM3FlagModel(**kwargs)      # 首次会从 HF 下载到 ~/.cache/huggingface
        logger.info("模型加载完成, 耗时 %.1fs", time.perf_counter() - t0)

        self.device = self.model.target_devices[0]
        self._loaded_at = time.time()

        # 预热: 触发一次前向, 让 MPS/CUDA kernel 编译、显存分配发生在启动期而非首个请求
        warm = self.encode(["ping"], return_dense=True, return_sparse=False,
                           return_colbert_vecs=False, max_length=None)
        self.dim = int(warm["dense"][0].__len__()) if warm["dense"] else 0
        logger.info("预热完成, device=%s, dim=%d", self.device, self.dim)

    def unload(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
        import torch
        if torch.backends.mps.is_available():       # 只在 mps 真实可用时清理, 否则原生崩溃
            try:
                torch.mps.empty_cache()
            except Exception:  # pragma: no cover
                pass

    def encode(self, texts: list[str], *, return_dense: bool, return_sparse: bool,
               return_colbert_vecs: bool, max_length: int | None) -> dict:
        """同步编码一批文本 (在攒批器单线程 executor 中调用, 前向天然串行)。

        返回 {dense, sparse_ids, sparse_tokens, colbert}, 各自与 texts 顺序对齐;
        未请求的通道为 None。此方法只应在 executor 线程里被攒批器调用。
        """
        assert self.model is not None, "模型未加载"
        out = self.model.encode(
            texts,
            batch_size=INFER_BATCH,                 # 单次前向内部仍按此批大小分块
            max_length=max_length or PASSAGE_MAX_LEN,
            return_dense=return_dense,
            return_sparse=return_sparse,
            return_colbert_vecs=return_colbert_vecs,
        )

        # 稀疏权重原生是 {token_id(str): weight}, 顺便解码成词形式(人看/调试用)
        sparse_ids = sparse_tokens = None
        if return_sparse:
            sparse_ids = out["lexical_weights"]
            sparse_tokens = self.model.convert_id_to_token(out["lexical_weights"])

        return {
            "dense": out["dense_vecs"].tolist() if return_dense else None,
            "sparse_ids": sparse_ids,
            "sparse_tokens": sparse_tokens,
            "colbert": [v.tolist() for v in out["colbert_vecs"]] if return_colbert_vecs else None,
        }


# ---------------------------------------------------------------------------
# 服务端攒批器: 并发小请求 -> 合并为一次模型前向 -> 结果按请求切回
# ---------------------------------------------------------------------------
class Batcher:
    """把同选项(dense/sparse/colbert/max_length)的并发请求合并成一次 encode。

    为什么需要它 (并发=1 的模型的唯一高吞吐姿势):
      单模型同时跑多个前向只会互相拖慢、还可能爆显存 —— 所以模型前向在
      单线程 executor 里串行。若"一个请求 = 一次前向", 那 N 个并发小请求就是
      N 次低效的前向(每批 1 条)。攒批器让它们排队合并成一批再前向,
      相当于在服务端替调用方实现了"攒 batch", 调用方无需感知。

    攒批策略 (简单、延迟可预期):
      - 请求入队后立即并入当前批次; 队列还有货时不做任何人肉等待。
      - 只有当队列空了、在等"同路人"时才按 GRACE 滑窗等待: 窗口内来了新请求
        就并入并重新计时; 窗口内没人来就单独发车 —— 单个小请求的额外延迟 ≈ GRACE。
      - 批次内文本条数达到 TARGET 立即发车, 防止持续流量下批次无限长大;
        单请求自带 ≥TARGET 条文本时(批量索引场景)会立刻发车, 零额外延迟。
    """

    def __init__(self, encode: Callable[..., dict], *, target: int, grace_ms: float) -> None:
        self._encode = encode                 # 同步函数 encode(texts, *, return_dense, ...) -> dict
        self._target = target
        self._grace_s = max(0.0, grace_ms / 1000.0)
        self._exec: concurrent.futures.ThreadPoolExecutor | None = None
        self._buckets: dict[tuple, asyncio.Queue] = {}
        self._tasks: list[asyncio.Task] = []
        self._stats = {"batches": 0, "requests": 0, "texts": 0, "errors": 0}

    # -- 生命周期 ----------------------------------------------------------
    def start(self) -> None:
        if self._exec is None:
            # 单 worker: 模型前向全局串行。线程名方便排查, executor 由 lifespan 管理
            self._exec = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="bge-encode")

    async def stop(self) -> None:
        for t in self._tasks:
            t.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        self._buckets.clear()
        if self._exec is not None:
            self._exec.shutdown(wait=True)    # 等在途 encode 收尾(优雅排空)
            self._exec = None

    # -- 对外: 异步提交一组文本, 返回与 texts 对齐的 {dense, sparse_ids, ...} ----
    async def submit(self, texts: list[str], *, dense: bool, sparse: bool,
                     colbert: bool, max_length: int | None) -> dict:
        if self._exec is None:
            raise RuntimeError("攒批器未启动")
        # 不同选项的请求互相独立攒批(避免"为了一个 colbert 请求给所有人算 colbert")
        key = (dense, sparse, colbert, max_length)
        q = self._buckets.get(key)
        if q is None:
            q = asyncio.Queue()
            self._buckets[key] = q
            self._tasks.append(asyncio.create_task(self._worker(key, q)))

        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        q.put_nowait((texts, fut))
        return await fut                       # encode 失败时这里抛异常

    # -- 攒批 worker: 每个 key 一个循环 ---------------------------------------
    async def _worker(self, key: tuple, q: asyncio.Queue) -> None:
        while True:
            first_texts, first_fut = await q.get()
            group: list[tuple[list[str], asyncio.Future]] = [(first_texts, first_fut)]
            total = len(first_texts)

            if self._target and total < self._target:
                while total < self._target:
                    # 队列还有货 -> 立即取出并入; 队列空 -> 给 grace 滑窗等"同路人"
                    try:
                        texts, fut = await asyncio.wait_for(q.get(), timeout=self._grace_s)
                    except asyncio.TimeoutError:
                        break
                    group.append((texts, fut))
                    total += len(texts)

            merged_texts = [t for t, _ in group for t in t]
            dense, sparse, colbert, max_length = key
            try:
                # run_in_executor 只收位置参数, 而 store.encode 是 keyword-only 签名,
                # 所以用 partial 把选项绑成关键字, 在单线程 executor 里执行这次前向
                call = functools.partial(
                    self._encode, merged_texts,
                    return_dense=dense, return_sparse=sparse,
                    return_colbert_vecs=colbert, max_length=max_length)
                data = await asyncio.get_running_loop().run_in_executor(self._exec, call)
            except Exception as e:               # OOM / 算子错误等
                logger.error("批次编码失败 (%d 请求, %d 条): %s", len(group), total, e)
                self._stats["errors"] += 1
                for _, fut in group:
                    if not fut.done():
                        fut.set_exception(e)
                continue

            self._stats["batches"] += 1
            self._stats["requests"] += len(group)
            self._stats["texts"] += total
            if total > 1:
                logger.debug("攒批发车: %d 请求 -> %d 条文本 一次前向", len(group), total)

            # 按各请求的文本数把结果切回去
            pos = 0
            for texts, fut in group:
                n = len(texts)
                if not fut.done():
                    fut.set_result({
                        name: (None if data.get(name) is None else data[name][pos:pos + n])
                        for name in ("dense", "sparse_ids", "sparse_tokens", "colbert")
                    })
                pos += n

    # -- 监控 ----------------------------------------------------------------
    def stats(self) -> dict:
        return dict(self._stats, pending=sum(q.qsize() for q in self._buckets.values()),
                    target=self._target, grace_ms=self._grace_s * 1000)


# ---------------------------------------------------------------------------
# FastAPI 应用
# ---------------------------------------------------------------------------
store = ModelStore()
batcher = Batcher(store.encode, target=BATCH_TARGET, grace_ms=BATCH_GRACE_MS)
_request_stats = {"ok": 0, "err": 0, "lat_sum_s": 0.0}     # /embed 侧统计(事件循环线程内更新)


@asynccontextmanager
async def lifespan(_: FastAPI):
    store.load()          # 失败抛异常 -> 启动即退出, 不让"半死"服务上线
    batcher.start()
    logger.info("服务就绪: model=%s device=%s dim=%d | 服务端攒批 target=%d, grace=%.0fms"
                " | api_key=%s", MODEL_NAME, store.device, store.dim,
                BATCH_TARGET, BATCH_GRACE_MS, "on" if API_KEY else "off")
    yield
    logger.info("收到关闭信号, 排空在途请求...")
    await batcher.stop()
    store.unload()
    logger.info("已优雅退出")


app = FastAPI(title="bge-m3 embedding service", lifespan=lifespan)


# ---- 鉴权: 设置 EMBED_API_KEY 后, 受保护端点需带 X-API-Key -------------------
def require_key(x_api_key: str | None = Header(default=None)) -> None:
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="无效或缺失 X-API-Key")


class EmbedRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=MAX_BATCH,
                             description=f"文本列表, 1~{MAX_BATCH} 条; 服务端会跨请求合并攒批, 无需自行分块")
    return_dense: bool = True
    return_sparse: bool = False
    return_colbert_vecs: bool = False
    max_length: int | None = Field(default=None, ge=16, le=PASSAGE_MAX_LEN,
                                   description="覆盖默认截断长度(token 数)")


class EmbedResponse(BaseModel):
    dense: list[list[float]] | None = None          # 每条文本一个 1024 维向量
    sparse_ids: list[dict[str, float]] | None = None    # {token_id: weight}
    sparse_tokens: list[dict[str, float]] | None = None # {token文本: weight}
    colbert: list[list[list[float]]] | None = None      # 每条文本: [token数, 1024]
    meta: dict


def _validate(req: EmbedRequest) -> None:
    """入参校验(不占模型资源, 进队列前先做)。"""
    for t in req.texts:
        if not t.strip():
            raise HTTPException(status_code=422, detail="texts 不允许空字符串")
        if len(t) > MAX_TEXT_CHARS:
            raise HTTPException(status_code=422,
                                detail=f"单条文本超过 {MAX_TEXT_CHARS} 字符, 请先切分")
    if not (req.return_dense or req.return_sparse or req.return_colbert_vecs):
        raise HTTPException(status_code=422, detail="至少要返回一种嵌入")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if store.model else "loading",
        "model": MODEL_NAME,
        "device": store.device,
        "dim": store.dim,
        "max_tokens": PASSAGE_MAX_LEN,
        "pending": batcher.stats()["pending"],      # 排队中的请求数(调大 = 服务忙)
        "uptime_s": round(time.time() - store._loaded_at) if store._loaded_at else None,
    }


@app.get("/info", dependencies=[Depends(require_key)])
def info() -> dict:
    return {"model": MODEL_NAME, "dim": store.dim, "max_tokens": PASSAGE_MAX_LEN,
            "device": store.device, "fp16": USE_FP16, "batcher": batcher.stats()}


@app.post("/embed", dependencies=[Depends(require_key)])
async def embed(req: EmbedRequest) -> EmbedResponse:
    """嵌入端点: 入队 -> 攒批 -> 一次前向 -> 切回结果。同步校验, 异步等待。"""
    if store.model is None:
        raise HTTPException(status_code=503, detail="模型尚未加载完成")
    _validate(req)

    t0 = time.perf_counter()
    try:
        # 攒批器内按 (dense/sparse/colbert/max_length) 合并并发请求;
        # 模型前向在单线程 executor 执行, 不会阻塞事件循环。
        data = await batcher.submit(req.texts, dense=req.return_dense,
                                    sparse=req.return_sparse, colbert=req.return_colbert_vecs,
                                    max_length=req.max_length)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("encode 失败: %s", e)
        _request_stats["err"] += 1
        raise HTTPException(status_code=500,
                            detail=f"编码失败(可调低 EMBED_INFER_BATCH / 看服务端日志): {e}") from e

    data["meta"] = {
        "model": MODEL_NAME, "dim": store.dim, "device": store.device,
        "num_texts": len(req.texts), "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
    }
    _request_stats["ok"] += 1
    _request_stats["lat_sum_s"] += time.perf_counter() - t0
    return EmbedResponse(**data)


@app.get("/metrics")
def metrics() -> Response:
    """Prometheus 文本格式。接入: 在 Prometheus 加 scrape_configs 指向本端点。"""
    b = batcher.stats()
    ok, err = _request_stats["ok"], _request_stats["err"]
    lat = _request_stats["lat_sum_s"]
    body = "\n".join([
        "# HELP bge_embed_requests_total 完成的 /embed 请求数(含失败)",
        "# TYPE bge_embed_requests_total counter",
        f'bge_embed_requests_total{{status="ok"}} {ok}',
        f'bge_embed_requests_total{{status="error"}} {err}',
        "# HELP bge_embed_latency_seconds /embed 端到端延迟(平均)",
        "# TYPE bge_embed_latency_seconds gauge",
        f"bge_embed_latency_seconds {{status=\"ok\"}} {round(lat / ok, 4) if ok else 0}",
        "# HELP bge_batcher_batches_total 服务端攒批发车次数",
        "# TYPE bge_batcher_batches_total counter",
        f"bge_batcher_batches_total {b['batches']}",
        "# HELP bge_batcher_requests_total 进入攒批器的请求数",
        "# TYPE bge_batcher_requests_total counter",
        f"bge_batcher_requests_total {b['requests']}",
        "# HELP bge_batcher_texts_total 经攒批器编码的文本条数",
        "# TYPE bge_batcher_texts_total counter",
        f"bge_batcher_texts_total {b['texts']}",
        "# HELP bge_batcher_errors_total 编码失败批次",
        "# TYPE bge_batcher_errors_total counter",
        f"bge_batcher_errors_total {b['errors']}",
        "# HELP bge_batcher_pending 队列中待处理请求数",
        "# TYPE bge_batcher_pending gauge",
        f"bge_batcher_pending {b['pending']}",
        "",
    ])
    return Response(content=body, media_type="text/plain; version=0.0.4")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.getenv("EMBED_HOST", "0.0.0.0"),
                port=int(os.getenv("EMBED_PORT", "8001")), log_level="info")
