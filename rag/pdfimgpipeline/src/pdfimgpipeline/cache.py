"""可插拔缓存后端 + 决策/结果缓存封装。

后端实现：RedisCache / MemoryCache / NullCache。
`build_cache` 在 `cache_backend="auto"` 时先试 Redis，连不上自动降级内存缓存，
保证缓存故障不会拖垮服务。
"""

from __future__ import annotations

import json
import logging
import time
from collections import OrderedDict
from dataclasses import asdict
from typing import Optional, Protocol, Tuple

from redis import asyncio as aioredis

from .config import Settings
from .models import RouteResult

logger = logging.getLogger(__name__)


class CacheBackend(Protocol):
    """异步 KV 缓存后端。读写失败应自行吞掉异常并降级，不向上抛。"""

    name: str

    async def get(self, key: str) -> Optional[str]: ...

    async def set(self, key: str, value: str, ttl: int) -> None: ...

    async def ping(self) -> bool: ...

    async def aclose(self) -> None: ...


class NullCache:
    """禁用缓存。"""

    name = "none"

    async def get(self, key: str) -> Optional[str]:
        return None

    async def set(self, key: str, value: str, ttl: int) -> None:
        return None

    async def ping(self) -> bool:
        return True

    async def aclose(self) -> None:
        return None


class MemoryCache:
    """进程内 TTL + LRU 缓存，Redis 不可用时的兜底。"""

    name = "memory"

    def __init__(self, max_entries: int = 10000):
        self._data: "OrderedDict[str, Tuple[str, float]]" = OrderedDict()
        self._max = max_entries

    async def get(self, key: str) -> Optional[str]:
        item = self._data.get(key)
        if item is None:
            return None
        value, expire_at = item
        if expire_at and expire_at < time.time():
            self._data.pop(key, None)
            return None
        self._data.move_to_end(key)
        return value

    async def set(self, key: str, value: str, ttl: int) -> None:
        expire_at = time.time() + ttl if ttl else 0.0
        self._data[key] = (value, expire_at)
        self._data.move_to_end(key)
        while len(self._data) > self._max:
            self._data.popitem(last=False)

    async def ping(self) -> bool:
        return True

    async def aclose(self) -> None:
        self._data.clear()


class RedisCache:
    """redis.asyncio 后端，全异步不阻塞事件循环。"""

    name = "redis"

    def __init__(self, settings: Settings):
        self._client = aioredis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            password=settings.redis_password or None,
            decode_responses=True,
            socket_timeout=settings.redis_socket_timeout,
            socket_connect_timeout=settings.redis_socket_timeout,
        )

    async def get(self, key: str) -> Optional[str]:
        try:
            return await self._client.get(key)
        except aioredis.RedisError as e:
            logger.warning("Redis GET failed: %s", e)
            return None

    async def set(self, key: str, value: str, ttl: int) -> None:
        try:
            if ttl:
                await self._client.setex(key, ttl, value)
            else:
                await self._client.set(key, value)
        except aioredis.RedisError as e:
            logger.warning("Redis SET failed: %s", e)

    async def ping(self) -> bool:
        await self._client.ping()
        return True

    async def aclose(self) -> None:
        await self._client.aclose()


async def build_cache(settings: Settings) -> CacheBackend:
    """按配置构建缓存后端，auto 模式下 Redis 失败自动降级内存。"""
    mode = settings.cache_backend.lower()
    if mode == "none":
        return NullCache()
    if mode == "memory":
        return MemoryCache()
    if mode in ("redis", "auto"):
        redis_cache = RedisCache(settings)
        try:
            await redis_cache.ping()
            logger.info("缓存后端: redis (%s:%s/%s)",
                        settings.redis_host, settings.redis_port, settings.redis_db)
            return redis_cache
        except Exception as e:  # noqa: BLE001 - 连接失败类型多样
            await redis_cache.aclose()
            if mode == "redis":
                raise
            logger.warning("Redis 不可用(%s)，降级为内存缓存", e)
    return MemoryCache()


class RouteCache:
    """决策结果缓存：key = pdfimg:route:<image_hash>。"""

    prefix = "pdfimg:route:"

    def __init__(self, backend: CacheBackend, ttl: int):
        self.backend = backend
        self.ttl = ttl

    async def get(self, img_hash: str) -> Optional[RouteResult]:
        raw = await self.backend.get(self.prefix + img_hash)
        if not raw:
            return None
        data = json.loads(raw)
        data["from_cache"] = True
        return RouteResult(**data)

    async def set(self, img_hash: str, result: RouteResult) -> None:
        data = asdict(result)
        data["from_cache"] = False
        await self.backend.set(self.prefix + img_hash, json.dumps(data), self.ttl)


class ResultCache:
    """最终结果缓存：key = pdfimg:result:<image_hash>。"""

    prefix = "pdfimg:result:"

    def __init__(self, backend: CacheBackend, ttl: int):
        self.backend = backend
        self.ttl = ttl

    async def get(self, img_hash: str) -> Optional[str]:
        return await self.backend.get(self.prefix + img_hash)

    async def set(self, img_hash: str, payload: str) -> None:
        await self.backend.set(self.prefix + img_hash, payload, self.ttl)
