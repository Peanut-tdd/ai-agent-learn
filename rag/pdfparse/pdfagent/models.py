"""领域模型：PDF 类型、页特征、预检结果、路由结果、批量结果。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class PdfKind(Enum):
    """预检判定的 PDF 类型。"""

    TEXT = "text"        # 文本型：有可靠文本层
    HYBRID = "hybrid"    # 混合型：部分页有文本层、部分页是扫描图
    SCANNED = "scanned"  # 扫描型：几乎整篇无文本层


#: PDF 类型 -> 版式解析引擎
KIND_TO_ENGINE = {
    PdfKind.TEXT: "pymupdf",
    PdfKind.HYBRID: "mineru",
    PdfKind.SCANNED: "mineru",
}


class PageKind(Enum):
    TEXT = "text"
    HYBRID = "hybrid"
    SCANNED = "scanned"
    BLANK = "blank"


@dataclass
class PageFeature:
    """单页预检特征。"""

    page: int              # 0-based 页码
    text_chars: int        # 去空白后的文本层字符数
    image_count: int       # 图片块数量
    image_area_ratio: float  # 图片覆盖面积 / 页面面积
    kind: str              # PageKind.value

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PrecheckResult:
    """整篇 PDF 的预检结果：可解释、可序列化、可缓存。"""

    pdf_hash: str
    pages: int
    text_pages: int
    hybrid_pages: int
    scanned_pages: int
    blank_pages: int
    text_ratio: float
    scanned_ratio: float
    kind: str                       # PdfKind.value
    reason: str
    page_features: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PrecheckResult":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class RouteResult:
    """一次路由决策：决定走哪个引擎，附预检证据。"""

    decision: str            # "pymupdf" | "mineru"
    kind: str                # PdfKind.value
    reason: str
    precheck: Dict[str, Any] = field(default_factory=dict)
    forced: bool = False
    from_cache: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RouteResult":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class BatchOutcome:
    """批量处理中单项的结果：成功带 data，失败带 error，互不影响。"""

    ok: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
