"""依赖装配：把 Settings 组装成可用的 PDFImagePipeline。"""

from __future__ import annotations

import logging
from typing import Tuple

from .adapters import OpenAICompatVLM, RapidOCRAdapter
from .cache import CacheBackend, ResultCache, RouteCache, build_cache
from .config import Settings
from .executor import ExtractorExecutor
from .features import PDFImageRouter
from .pipeline import PDFImagePipeline
from .router import HybridRouter

logger = logging.getLogger(__name__)


async def build_pipeline(settings: Settings) -> Tuple[PDFImagePipeline, CacheBackend]:
    """构建缓存后端 + OCR/VLM 适配器 + 路由器 + 执行器 + 顶层 Pipeline。"""
    backend = await build_cache(settings)

    ocr = RapidOCRAdapter(serialize=settings.ocr_serialize)
    vlm = OpenAICompatVLM(
        model=settings.vlm_model,
        api_key=settings.resolved_vlm_api_key,
        base_url=settings.resolved_vlm_base_url,
        timeout=settings.vlm_sdk_timeout,
        prompt=settings.vlm_prompt,
    )

    rule_router = PDFImageRouter(
        text_ratio_threshold=settings.text_ratio_threshold,
        entropy_threshold=settings.entropy_threshold,
        edge_density_threshold=settings.edge_density_threshold,
        color_var_threshold=settings.color_var_threshold,
    )
    route_cache = RouteCache(backend, settings.cache_ttl)
    hybrid_router = HybridRouter(
        rule_router,
        route_cache,
        ocr,
        ocr_threshold_chars=settings.ocr_threshold_chars,
        ocr_threshold_conf=settings.ocr_threshold_conf,
        min_probe_chars_for_hybrid=settings.min_probe_chars_for_hybrid,
    )
    executor = ExtractorExecutor(
        ocr_fn=ocr,
        vlm_fn=vlm,
        ocr_timeout=settings.ocr_timeout,
        vlm_timeout=settings.vlm_timeout,
        max_retries=settings.max_retries,
        retry_backoff=settings.retry_backoff,
        ocr_max_concurrency=settings.ocr_max_workers,
        vlm_max_concurrency=settings.vlm_max_concurrency,
    )
    result_cache = ResultCache(backend, settings.cache_ttl)
    return PDFImagePipeline(hybrid_router, executor, result_cache), backend
