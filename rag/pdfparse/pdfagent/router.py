"""路由决策：预检 -> 引擎，带缓存。

    auto:  缓存命中 -> 复用；否则预检并按 PdfKind 分流
    force: text / mineru 跳过预检（仍做预检以便记录证据，避免误判）
"""

from __future__ import annotations

import logging
from typing import Optional

from .cache import RouteCache
from .models import KIND_TO_ENGINE, PdfKind, RouteResult
from .precheck import PdfPrechecker
from .utils import file_sha256, short

logger = logging.getLogger(__name__)

#: 对外 force 取值 -> (PdfKind, 引擎)
FORCE_MAP = {
    "text": (PdfKind.TEXT, "pymupdf"),
    "pymupdf": (PdfKind.TEXT, "pymupdf"),
    "mineru": (PdfKind.SCANNED, "mineru"),
    "scanned": (PdfKind.SCANNED, "mineru"),
    "hybrid": (PdfKind.HYBRID, "mineru"),
}


class PdfRouter:
    def __init__(
        self,
        prechecker: PdfPrechecker,
        cache: RouteCache,
    ):
        self.prechecker = prechecker
        self.cache = cache

    def decide(
        self,
        pdf_path: str,
        *,
        force: Optional[str] = None,
        use_cache: bool = True,
    ) -> RouteResult:
        pdf_hash = file_sha256(pdf_path)

        if force and force.lower() != "auto":
            key = force.lower()
            if key not in FORCE_MAP:
                raise ValueError(f"未知 force 路由: {force}（可选 auto/text/mineru）")
            kind, engine = FORCE_MAP[key]
            precheck = self.prechecker.precheck(pdf_path, pdf_hash)
            result = RouteResult(
                decision=engine,
                kind=kind.value,
                reason=f"forced:{key}|precheck({precheck.reason})",
                precheck=precheck.to_dict(),
                forced=True,
                from_cache=False,
            )
            logger.info("[route forced] %s -> %s", short(pdf_hash), engine)
            return result

        if use_cache:
            cached = self.cache.get(pdf_hash)
            if cached:
                result = RouteResult.from_dict(cached)
                result.from_cache = True
                logger.info("[route cache hit] %s -> %s", short(pdf_hash), result.decision)
                return result

        precheck = self.prechecker.precheck(pdf_path, pdf_hash)
        engine = KIND_TO_ENGINE[PdfKind(precheck.kind)]
        result = RouteResult(
            decision=engine,
            kind=precheck.kind,
            reason=precheck.reason,
            precheck=precheck.to_dict(),
            forced=False,
            from_cache=False,
        )

        if use_cache:
            self.cache.set(pdf_hash, result.to_dict())
        logger.info("[route decided] %s -> %s (%s)",
                    short(pdf_hash), engine, precheck.reason)
        return result
