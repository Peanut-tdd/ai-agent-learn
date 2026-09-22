"""版式解析适配器：pymupdf（文本型）/ mineru（混合、扫描型）。"""

from __future__ import annotations

from .mineru import MinerUAdapter
from .pymupdf_layout import PyMuPDFLayoutAdapter

__all__ = ["PyMuPDFLayoutAdapter", "MinerUAdapter"]
