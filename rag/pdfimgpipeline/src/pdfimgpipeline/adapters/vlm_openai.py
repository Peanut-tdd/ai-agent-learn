"""OpenAI 兼容视觉接口适配器（qwen-vl-max / gpt-4o / glm-4v …）。"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
from typing import Optional

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class OpenAICompatVLM:
    def __init__(
        self,
        *,
        model: str,
        api_key: str = "",
        base_url: Optional[str] = None,
        timeout: float = 28.0,
        prompt: str,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.prompt = prompt
        self._client: Optional[AsyncOpenAI] = None
        self._lock = asyncio.Lock()

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            if not self.api_key:
                raise RuntimeError(
                    "VLM 未配置：请设置 VLM_API_KEY / OPENAI_API_KEY")
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
                max_retries=0,  # 重试统一交给 ExtractorExecutor
            )
        return self._client

    async def __call__(self, path: str) -> dict:
        client = self._get_client()

        def _read() -> str:
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode()

        b64 = await asyncio.to_thread(_read)
        resp = await client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": self.prompt},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }],
            temperature=0.2,
        )
        return {"text": resp.choices[0].message.content or ""}
