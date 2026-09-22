"""pdfagent —— PDF 预检 / 版式路由 / 缓存 / 图片语义化 编排层。

参考 pdfimgpipeline 的分层架构：
    config   集中配置（env / .env）
    models   领域模型（PdfKind / PrecheckResult / RouteResult / BatchOutcome）
    precheck 预检：逐页文本覆盖率 -> text / hybrid / scanned
    cache    可插拔缓存（redis/memory/none）+ 版本号命名空间
    router   路由决策：预检 -> pymupdf / mineru，带缓存
    adapters pymupdf 版式解析、mineru 版式解析
    pipeline 顶层入口：路由 -> 解析 -> 缓存
    wiring   Settings -> pipeline 依赖装配

对外入口::

    from pdfagent import Settings, build_pipeline
"""

from __future__ import annotations

__version__ = "0.1.0"

# 预检 / 路由结果的数据结构版本。字段语义变更时 +1，配合缓存版本号一起作废旧缓存。
SCHEMA_VERSION = "1"

from .config import Settings, load_dotenv
from .cache import (
    CacheBackend,
    MemoryCache,
    NullCache,
    RedisCache,
    ResultCache,
    RouteCache,
    VersionedCache,
    build_cache,
)
from .models import (
    BatchOutcome,
    PageFeature,
    PdfKind,
    PrecheckResult,
    RouteResult,
)
from .pipeline import PdfPipeline
from .precheck import PdfPrechecker
from .router import PdfRouter
from .utils import file_sha256
from .wiring import build_pipeline

__all__ = [
    "__version__",
    "SCHEMA_VERSION",
    "Settings",
    "load_dotenv",
    "CacheBackend",
    "NullCache",
    "MemoryCache",
    "RedisCache",
    "VersionedCache",
    "RouteCache",
    "ResultCache",
    "build_cache",
    "BatchOutcome",
    "PageFeature",
    "PdfKind",
    "PrecheckResult",
    "RouteResult",
    "PdfPipeline",
    "PdfPrechecker",
    "PdfRouter",
    "file_sha256",
    "build_pipeline",
]
