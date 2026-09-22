"""RapidOCR 适配器：懒加载单例 + 可选串行化 + to_thread 不阻塞事件循环。

RapidOCR 底层是 onnxruntime，官方未承诺线程安全；默认 ``serialize=True``
用锁串行化引擎调用，避免并发下崩溃/结果错乱。实测可并发时置 False。
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


class RapidOCRAdapter:
    def __init__(self, *, serialize: bool = True):
        self._engine = None
        self._load_lock = threading.Lock()
        self._call_lock = threading.Lock() if serialize else None

    def _ensure_engine(self):
        if self._engine is None:
            with self._load_lock:
                if self._engine is None:
                    from rapidocr_onnxruntime import RapidOCR
                    self._engine = RapidOCR()
        return self._engine

    def _raw(self, path: str):
        engine = self._ensure_engine()
        if self._call_lock is None:
            return engine(path)
        with self._call_lock:
            return engine(path)

    async def __call__(self, path: str) -> dict:
        def _run() -> dict:
            res, _ = self._raw(path)
            text = "\n".join(line[1] for line in res) if res else ""
            return {"text": text}

        return await asyncio.to_thread(_run)

    async def probe(self, path: str) -> Tuple[int, float]:
        """返回 (字符数, 平均置信度)；失败返回 (0, 0.0)。"""

        def _run() -> Tuple[int, float]:
            res, _ = self._raw(path)
            if not res:
                return 0, 0.0
            total = sum(len(line[1]) for line in res)
            conf = float(np.mean([line[2] for line in res]))
            return total, conf

        try:
            return await asyncio.to_thread(_run)
        except Exception as e:  # noqa: BLE001
            logger.warning("OCR probe failed: %s", e)
            return 0, 0.0
