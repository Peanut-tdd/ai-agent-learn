"""领域模型：路由决定、图像特征、路由结果。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class RouteDecision(Enum):
    OCR = "ocr"
    VLM = "vlm"
    HYBRID = "hybrid"


@dataclass
class ImageFeatures:
    """6 维轻量图像特征（不依赖任何模型）。"""

    text_ratio: float
    image_entropy: float
    edge_density: float
    color_variance: float
    num_connected_components: int
    aspect_ratio: float


@dataclass
class RouteResult:
    """一次路由决策的结果，可解释、可缓存。"""

    decision: str
    reason: str
    features: Optional[dict] = None
    ocr_probe_chars: int = 0
    ocr_probe_conf: float = 0.0
    from_cache: bool = False


@dataclass
class BatchOutcome:
    """批量处理中单项的结果：成功带 data，失败带 error，互不影响。"""

    ok: bool
    data: Optional[dict] = None
    error: Optional[str] = None
