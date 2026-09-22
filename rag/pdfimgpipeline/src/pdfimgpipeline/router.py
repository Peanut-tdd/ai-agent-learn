"""混合策略路由器：缓存 -> 特征 -> OCR 探测 -> 决策 -> 写缓存。"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict
from typing import Protocol, Sequence, Tuple

from .cache import RouteCache
from .features import PDFImageRouter
from .models import BatchOutcome, RouteDecision, RouteResult
from .utils import image_sha256

logger = logging.getLogger(__name__)


class OCRProbe(Protocol):
    """OCR 探测接口：返回 (字符数, 平均置信度)。"""

    async def probe(self, image_path: str) -> Tuple[int, float]: ...


class HybridRouter:
    def __init__(
        self,
        router: PDFImageRouter,
        cache: RouteCache,
        ocr_probe: OCRProbe,
        *,
        ocr_threshold_chars: int = 50,
        ocr_threshold_conf: float = 0.85,
        min_probe_chars_for_hybrid: int = 10,
    ):
        self.router = router
        self.cache = cache
        self.ocr_probe = ocr_probe
        self.ocr_threshold_chars = ocr_threshold_chars
        self.ocr_threshold_conf = ocr_threshold_conf
        self.min_probe_chars_for_hybrid = min_probe_chars_for_hybrid

    async def decide(self, image_path: str) -> RouteResult:
        # ---- 1. 缓存查询 ----
        # 文件 hash 与 cv2 提特征是阻塞 IO/CPU，放线程池，避免卡住事件循环
        img_hash = await asyncio.to_thread(image_sha256, image_path)
        cached = await self.cache.get(img_hash)
        if cached:
            logger.info("[cache hit] %s -> %s", img_hash[:8], cached.decision)
            return cached

        # ---- 2. 特征提取 + 规则打分 ----
        feats = await asyncio.to_thread(self.router.extract_features, image_path)
        score, reasons = self.router.rule_score(feats)

        # ---- 3. OCR 探测（仅在模糊区间才探测，节省算力）----
        probe_chars, probe_conf = 0, 0.0
        if -1 < score < 3:
            probe_chars, probe_conf = await self.ocr_probe.probe(image_path)
            reasons.append(f"ocr_probe(chars={probe_chars},conf={probe_conf:.2f})")

        # ---- 4. 混合决策 ----
        if score >= 3:
            decision = RouteDecision.VLM
        elif score <= -1:
            decision = RouteDecision.OCR
        else:
            # hybrid 区间：依赖 OCR 探测结果二次判断
            if probe_chars == 0:
                decision = RouteDecision.VLM
                reasons.append("probe_no_text->vlm")
            elif (probe_chars > self.ocr_threshold_chars
                  and probe_conf > self.ocr_threshold_conf):
                decision = RouteDecision.OCR
                reasons.append("probe_high_quality_text->ocr")
            elif probe_chars < self.min_probe_chars_for_hybrid:
                decision = RouteDecision.VLM
                reasons.append("probe_very_few_chars->vlm")
            else:
                decision = RouteDecision.HYBRID
                reasons.append("probe_medium->hybrid")

        result = RouteResult(
            decision=decision.value,
            reason="|".join(reasons),
            features=asdict(feats),
            ocr_probe_chars=probe_chars,
            ocr_probe_conf=probe_conf,
            from_cache=False,
        )

        # ---- 5. 写缓存 ----
        await self.cache.set(img_hash, result)
        logger.info("[decided] %s -> %s", img_hash[:8], decision.value)
        return result

    async def decide_many(
        self, image_paths: Sequence[str], *, max_concurrency: int = 8
    ) -> list:
        """并发批量决策，单张失败隔离为 BatchOutcome(ok=False)。"""
        sem = asyncio.Semaphore(max(1, max_concurrency))

        async def one(path: str) -> BatchOutcome:
            async with sem:
                try:
                    return BatchOutcome(ok=True, data=asdict(await self.decide(path)))
                except Exception as e:  # noqa: BLE001
                    logger.warning("batch route failed: %s (%s)", path, e)
                    return BatchOutcome(ok=False, error=str(e))

        return list(await asyncio.gather(*(one(p) for p in image_paths)))
