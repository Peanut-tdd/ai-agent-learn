"""HTTP 请求/响应模型。"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    version: str
    cache_backend: str
    vlm_configured: bool
    vlm_model: str


class RouteResponse(BaseModel):
    decision: str
    reason: str
    features: Optional[dict] = None
    ocr_probe_chars: int = 0
    ocr_probe_conf: float = 0.0
    from_cache: bool = False


class ExtractResponse(BaseModel):
    image_hash: str
    decision: str
    reason: str
    features: Optional[dict] = None
    ocr_probe: dict
    result: dict
    from_cache: bool = False


class BatchItem(BaseModel):
    index: int
    filename: Optional[str] = None
    ok: bool
    data: Optional[dict] = None
    error: Optional[str] = None


class BatchResponse(BaseModel):
    count: int
    ok_count: int
    failed_count: int
    results: list[BatchItem]
