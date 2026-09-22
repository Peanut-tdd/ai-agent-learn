"""pdfimgpipeline —— 文档图像 OCR / VLM 智能路由与执行。

对外入口：
    from pdfimgpipeline import build_pipeline, PDFImagePipeline, Settings
"""

from __future__ import annotations

__version__ = "0.1.0"

from .config import Settings
from .executor import ExtractorExecutor
from .features import PDFImageRouter
from .models import BatchOutcome, ImageFeatures, RouteDecision, RouteResult
from .pipeline import PDFImagePipeline
from .router import HybridRouter
from .wiring import build_pipeline

__all__ = [
    "__version__",
    "Settings",
    "build_pipeline",
    "PDFImagePipeline",
    "HybridRouter",
    "PDFImageRouter",
    "ExtractorExecutor",
    "BatchOutcome",
    "ImageFeatures",
    "RouteDecision",
    "RouteResult",
]
