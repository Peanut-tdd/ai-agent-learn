"""执行器：真实超时 + 重试 + 降级 + 并发上限 + 并发融合。

ocr_fn / vlm_fn 均为 async 可调用：``async def fn(image_path) -> dict``。
同步阻塞实现（如 RapidOCR）请用 asyncio.to_thread 包一层，见 adapters/。

超时策略（两层，缺一不可）：
  1) 编排层：asyncio.wait_for 对整次调用限时，到期 cancel 内层任务并触发降级；
  2) SDK 层：HTTP 客户端自带超时，保证底层连接真正中断。
注意：to_thread 派发的同步调用无法被强杀，超时后线程仍会跑完、结果被丢弃；
对幂等 OCR 可接受，需硬隔离请改用进程池。
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Awaitable, Callable, Optional, Union

logger = logging.getLogger(__name__)

AsyncFn = Callable[[str], Awaitable[dict]]

# 只对瞬时故障重试；参数/解析类错误立即抛出
_RETRYABLE = (asyncio.TimeoutError, TimeoutError, ConnectionError, OSError)


class ExtractorExecutor:
    def __init__(
        self,
        ocr_fn: AsyncFn,
        vlm_fn: AsyncFn,
        *,
        ocr_timeout: float = 15.0,
        vlm_timeout: float = 30.0,
        max_retries: int = 2,
        retry_backoff: float = 0.5,
        ocr_max_concurrency: int = 4,
        vlm_max_concurrency: int = 4,
    ):
        self.ocr_fn = ocr_fn
        self.vlm_fn = vlm_fn
        self.ocr_timeout = ocr_timeout
        self.vlm_timeout = vlm_timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        # 背压：VLM 很贵，限制在途并发
        self._ocr_sem = asyncio.Semaphore(max(1, ocr_max_concurrency))
        self._vlm_sem = asyncio.Semaphore(max(1, vlm_max_concurrency))

    # ---------------------------------------------------------------- 核心
    async def _call_with_timeout(
        self, fn: AsyncFn, image_path: str, timeout: float, label: str
    ) -> dict:
        """真实超时 + 指数退避重试，超时/连接类错误才重试。"""
        last_exc: Optional[BaseException] = None
        for attempt in range(self.max_retries + 1):
            started = time.perf_counter()
            try:
                return await asyncio.wait_for(fn(image_path), timeout=timeout)
            except asyncio.CancelledError:
                raise
            except _RETRYABLE as e:
                last_exc = e
                logger.warning(
                    "%s failed (attempt %d/%d, %.2fs): %s",
                    label, attempt + 1, self.max_retries + 1,
                    time.perf_counter() - started, e,
                )
                if attempt < self.max_retries:
                    delay = self.retry_backoff * (2 ** attempt) * (0.5 + random.random())
                    await asyncio.sleep(delay)
            except Exception:
                logger.error("%s non-retryable error", label, exc_info=True)
                raise
        assert last_exc is not None
        raise last_exc

    async def _guarded(
        self, sem: asyncio.Semaphore, fn: AsyncFn,
        image_path: str, timeout: float, label: str,
    ) -> dict:
        async with sem:
            return await self._call_with_timeout(fn, image_path, timeout, label)

    # ---------------------------------------------------------------- 调度
    async def execute(self, image_path: str, decision: str) -> dict:
        if decision == "ocr":
            return await self._safe_ocr(image_path)
        if decision == "vlm":
            return await self._safe_vlm(image_path)
        return await self._run_hybrid(image_path)

    async def _safe_ocr(self, image_path: str) -> dict:
        try:
            data = await self._guarded(
                self._ocr_sem, self.ocr_fn, image_path, self.ocr_timeout, "OCR")
            return {"engine": "ocr", "data": data}
        except Exception as e:  # noqa: BLE001
            logger.error("OCR failed: %s", e)
            return {"engine": "ocr", "data": None, "error": str(e)}

    async def _safe_vlm(self, image_path: str) -> dict:
        """VLM 失败降级到 OCR。"""
        try:
            data = await self._guarded(
                self._vlm_sem, self.vlm_fn, image_path, self.vlm_timeout, "VLM")
            return {"engine": "vlm", "data": data}
        except Exception as e:  # noqa: BLE001
            logger.warning("VLM failed, fallback to OCR: %s", e)
            fallback = await self._safe_ocr(image_path)
            fallback["fallback_from"] = "vlm"
            fallback["vlm_error"] = str(e)
            return fallback

    async def _run_hybrid(self, image_path: str) -> dict:
        """双跑融合：OCR / VLM 并发执行，谁成功用谁，都成功则合并。"""
        async def run_ocr() -> dict:
            return await self._guarded(
                self._ocr_sem, self.ocr_fn, image_path, self.ocr_timeout, "hybrid-OCR")

        async def run_vlm() -> dict:
            return await self._guarded(
                self._vlm_sem, self.vlm_fn, image_path, self.vlm_timeout, "hybrid-VLM")

        ocr_res, vlm_res = await asyncio.gather(
            run_ocr(), run_vlm(), return_exceptions=True)

        ocr_ok = None if isinstance(ocr_res, BaseException) else ocr_res
        vlm_ok = None if isinstance(vlm_res, BaseException) else vlm_res
        if isinstance(ocr_res, BaseException):
            logger.warning("hybrid-OCR failed: %s", ocr_res)
        if isinstance(vlm_res, BaseException):
            logger.warning("hybrid-VLM failed: %s", vlm_res)

        if ocr_ok and vlm_ok:
            return {"engine": "hybrid", "data": self._merge(ocr_ok, vlm_ok)}
        if ocr_ok:
            return {"engine": "ocr(fallback)", "data": ocr_ok}
        if vlm_ok:
            return {"engine": "vlm(fallback)", "data": vlm_ok}
        return {"engine": "none", "data": None, "error": "both failed"}

    @staticmethod
    def _merge(ocr_res: dict, vlm_res: dict) -> dict:
        """简单融合：OCR 文本 + VLM 描述，生产可换成 LLM 融合。"""
        return {
            "ocr_text": ocr_res.get("text", ""),
            "vlm_description": vlm_res.get("text", ""),
            "merged": f"{ocr_res.get('text', '')}\n[VLM]:{vlm_res.get('text', '')}",
        }
