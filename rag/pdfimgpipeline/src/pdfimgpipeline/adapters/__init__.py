"""可替换的 OCR / VLM 适配器。"""

from __future__ import annotations

from .ocr_rapid import RapidOCRAdapter
from .vlm_openai import OpenAICompatVLM

__all__ = ["RapidOCRAdapter", "OpenAICompatVLM"]
