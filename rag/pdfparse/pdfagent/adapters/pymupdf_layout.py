"""pymupdf 版式解析适配器：包装项目根目录的 ``layout_parser``。

文本型 PDF 走这里：规则式解析章节 / 段落 / 表格 / 图片，产出与 MinerU 适配器
一致的 result 契约（meta/sections/tables/figures/chunks/elements）。
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional

# layout_parser 位于包外的项目根目录，确保可 import
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


class PyMuPDFLayoutAdapter:
    name = "pymupdf"

    def available(self) -> bool:
        return True

    def parse(
        self,
        pdf_path: str,
        out_dir: str,
        *,
        chunk_limit: int = 600,
        image_describer: Optional[object] = None,
    ) -> Dict[str, Any]:
        import layout_parser  # 延迟 import，避免包外路径问题

        result = layout_parser.parse_document(
            pdf_path, out_dir, chunk_limit=chunk_limit, image_describer=image_describer
        )
        result["engine"] = self.name
        return result
