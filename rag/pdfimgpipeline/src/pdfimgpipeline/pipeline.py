"""顶层 Pipeline：缓存 -> 决策 -> 执行 -> 缓存，对外产出结构化 JSON。"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional, Sequence

from .cache import ResultCache
from .executor import ExtractorExecutor
from .models import BatchOutcome
from .router import HybridRouter
from .utils import image_sha256

logger = logging.getLogger(__name__)


class PDFImagePipeline:
    def __init__(
        self,
        router: HybridRouter,
        executor: ExtractorExecutor,
        result_cache: Optional[ResultCache] = None,
    ):
        self.router = router
        self.executor = executor
        self.result_cache = result_cache

    async def process(self, image_path: str) -> dict:
        img_hash = await asyncio.to_thread(image_sha256, image_path)

        # 结果缓存优先，避免重复执行
        if self.result_cache is not None:
            cached = await self.result_cache.get(img_hash)
            if cached:
                data = json.loads(cached)
                data["from_cache"] = True
                return data

        decision = await self.router.decide(image_path)
        exec_result = await self.executor.execute(image_path, decision.decision)

        final = {
            "image_hash": img_hash,
            "decision": decision.decision,
            "reason": decision.reason,
            "features": decision.features,
            "ocr_probe": {
                "chars": decision.ocr_probe_chars,
                "conf": decision.ocr_probe_conf,
            },
            "result": exec_result,
            "from_cache": False,
        }

        if self.result_cache is not None:
            await self.result_cache.set(
                img_hash, json.dumps(final, ensure_ascii=False))

        return final

    async def process_many(
        self,
        image_paths: Sequence[str],
        *,
        max_concurrency: int = 8,
    ) -> list:
        """并发批量处理，返回与输入等长的 :class:`BatchOutcome` 列表。

        - 单张失败只影响该项（error 带原因），不会让整批挂掉；
        - 结果顺序与输入一致；
        - 缓存对每张图独立生效，重复图片会直接命中。
        """
        sem = asyncio.Semaphore(max(1, max_concurrency))

        async def one(path: str) -> BatchOutcome:
            async with sem:
                try:
                    return BatchOutcome(ok=True, data=await self.process(path))
                except Exception as e:  # noqa: BLE001 - 批量隔离单项失败
                    logger.warning("batch item failed: %s (%s)", path, e)
                    return BatchOutcome(ok=False, error=str(e))

        return list(await asyncio.gather(*(one(p) for p in image_paths)))
