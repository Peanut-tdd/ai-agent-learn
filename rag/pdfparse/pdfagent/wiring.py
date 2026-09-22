"""依赖装配：把 Settings 组装成可用的 PdfPipeline。"""

from __future__ import annotations

import logging
from typing import Tuple

from .adapters import MinerUAdapter, PyMuPDFLayoutAdapter
from .cache import CacheBackend, ResultCache, RouteCache, build_cache
from .config import Settings
from .pipeline import PdfPipeline
from .precheck import PdfPrechecker
from .router import PdfRouter

logger = logging.getLogger(__name__)


def build_pipeline(
    settings: Settings,
    *,
    backend: CacheBackend | None = None,
) -> Tuple[PdfPipeline, CacheBackend]:
    """构建缓存后端 + 预检器 + 路由器 + 解析器 + 顶层 Pipeline。"""
    backend = backend or build_cache(settings)

    prechecker = PdfPrechecker(settings)
    route_cache = RouteCache(backend, settings)
    router = PdfRouter(prechecker, route_cache)

    parsers = {
        "pymupdf": PyMuPDFLayoutAdapter(),
        "mineru": MinerUAdapter(settings),
    }
    result_cache = ResultCache(backend, settings)
    pipeline = PdfPipeline(router, parsers, result_cache, settings)
    return pipeline, backend
