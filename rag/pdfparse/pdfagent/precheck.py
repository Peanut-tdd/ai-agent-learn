"""PDF 预检：逐页文本覆盖率 -> text / hybrid / scanned。

判定思路（纯 PyMuPDF，不依赖任何模型，毫秒级）：
1. 每页统计去空白后的文本层字符数、图片块数量与图片覆盖面积占比；
2. 单页分类：
   - 字符数 >= text_page_min_chars            -> 文本页
   - 有少量字符（>= sparse_page_min_chars）   -> 混合页
   - 无文本 + 图片覆盖面积占比达阈值          -> 扫描页
   - 其余                                     -> 空白页（不参与比例）
3. 整篇聚合（只在非空白页上算比例）：
   - 扫描页占比 >= scanned_ratio_threshold -> scanned
   - 文本页占比 >= text_ratio_threshold    -> text
   - 其余                                  -> hybrid

结果带 reason，便于排查“为什么这份 PDF 走了 MinerU”。
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import pymupdf

from .config import Settings
from .models import PageFeature, PageKind, PdfKind, PrecheckResult
from .utils import file_sha256

logger = logging.getLogger(__name__)


class PdfPrechecker:
    def __init__(self, settings: Settings):
        self.settings = settings

    # ---------------------------------------------------------------- 特征
    def extract_features(self, pdf_path: str) -> List[PageFeature]:
        """逐页提取轻量特征（CPU 阻塞但很快）。"""
        doc = pymupdf.open(pdf_path)
        try:
            return [self._page_feature(doc[pno], pno) for pno in range(len(doc))]
        finally:
            doc.close()

    def _page_feature(self, page, pno: int) -> PageFeature:
        text = page.get_text("text") or ""
        chars = len("".join(text.split()))

        page_area = abs(page.rect.width * page.rect.height) or 1.0
        image_area = 0.0
        image_count = 0
        try:
            infos = page.get_image_info()
        except Exception:  # noqa: BLE001 - 个别页可能取不到图片信息
            infos = []
        for info in infos:
            bbox = info.get("bbox")
            if not bbox:
                continue
            image_count += 1
            image_area += max(0.0, pymupdf.Rect(bbox).get_area())
        ratio = min(1.0, image_area / page_area)

        kind = self._classify_page(chars, ratio)
        return PageFeature(
            page=pno,
            text_chars=chars,
            image_count=image_count,
            image_area_ratio=round(ratio, 4),
            kind=kind.value,
        )

    def _classify_page(self, chars: int, image_area_ratio: float) -> PageKind:
        s = self.settings
        if chars >= s.text_page_min_chars:
            return PageKind.TEXT
        if chars >= s.sparse_page_min_chars:
            return PageKind.HYBRID
        if image_area_ratio >= s.scanned_image_area_ratio:
            return PageKind.SCANNED
        return PageKind.BLANK

    # ---------------------------------------------------------------- 聚合
    def classify(self, features: List[PageFeature]) -> Tuple[PdfKind, str]:
        text = sum(1 for f in features if f.kind == PageKind.TEXT.value)
        hybrid = sum(1 for f in features if f.kind == PageKind.HYBRID.value)
        scanned = sum(1 for f in features if f.kind == PageKind.SCANNED.value)
        blank = sum(1 for f in features if f.kind == PageKind.BLANK.value)
        total = len(features)
        effective = total - blank

        if effective <= 0:
            reason = f"pages={total} 全为空白页 -> 按文本型处理"
            return PdfKind.TEXT, reason

        text_ratio = text / effective
        scanned_ratio = scanned / effective
        s = self.settings

        if scanned_ratio >= s.scanned_ratio_threshold:
            kind, tail = PdfKind.SCANNED, (
                f"scanned_ratio={scanned_ratio:.0%}>={s.scanned_ratio_threshold:.0%}->scanned")
        elif text_ratio >= s.text_ratio_threshold:
            kind, tail = PdfKind.TEXT, (
                f"text_ratio={text_ratio:.0%}>={s.text_ratio_threshold:.0%}->text")
        else:
            kind, tail = PdfKind.HYBRID, (
                f"text_ratio={text_ratio:.0%}<{s.text_ratio_threshold:.0%}"
                f"且scanned_ratio={scanned_ratio:.0%}<{s.scanned_ratio_threshold:.0%}->hybrid")

        reason = (
            f"pages={total}(blank={blank}) text={text} hybrid={hybrid} scanned={scanned}|"
            + tail
        )
        return kind, reason

    # ---------------------------------------------------------------- 入口
    def precheck(self, pdf_path: str, pdf_hash: str = "") -> PrecheckResult:
        pdf_hash = pdf_hash or file_sha256(pdf_path)
        features = self.extract_features(pdf_path)
        kind, reason = self.classify(features)
        return self.build_result(pdf_hash, features, kind, reason)

    def build_result(
        self,
        pdf_hash: str,
        features: List[PageFeature],
        kind: PdfKind,
        reason: str,
    ) -> PrecheckResult:
        text = sum(1 for f in features if f.kind == PageKind.TEXT.value)
        hybrid = sum(1 for f in features if f.kind == PageKind.HYBRID.value)
        scanned = sum(1 for f in features if f.kind == PageKind.SCANNED.value)
        blank = sum(1 for f in features if f.kind == PageKind.BLANK.value)
        effective = len(features) - blank
        return PrecheckResult(
            pdf_hash=pdf_hash,
            pages=len(features),
            text_pages=text,
            hybrid_pages=hybrid,
            scanned_pages=scanned,
            blank_pages=blank,
            text_ratio=round(text / effective, 4) if effective else 1.0,
            scanned_ratio=round(scanned / effective, 4) if effective else 0.0,
            kind=kind.value,
            reason=reason,
            page_features=[f.to_dict() for f in features],
        )
