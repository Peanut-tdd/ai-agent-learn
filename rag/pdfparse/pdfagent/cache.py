"""可插拔缓存后端 + 版本号命名空间。

后端实现：RedisCache / MemoryCache / NullCache。
`build_cache` 在 `cache_backend="auto"` 时先试 Redis，连不上或未安装 redis 客户端
自动降级内存缓存，保证缓存故障不会拖垮主流程。

所有键都带版本号：``<namespace>:<version>:<kind>:<hash>``。
项目更新（改 __version__ / SCHEMA_VERSION）或显式设置 ``PDF_CACHE_VERSION``
都会让旧缓存自然失效，无需手工清理 Redis。
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import OrderedDict
from typing import Any, Optional, Protocol, Tuple

from .config import Settings

logger = logging.getLogger(__name__)


class CacheBackend(Protocol):
    """同步 KV 缓存后端。读写失败应自行吞掉异常并降级，不向上抛。"""

    name: str

    def get(self, key: str) -> Optional[str]: ...

    def set(self, key: str, value: str, ttl: int) -> None: ...

    def delete(self, key: str) -> None: ...

    def ping(self) -> bool: ...

    def close(self) -> None: ...


class NullCache:
    """禁用缓存。"""

    name = "none"

    def get(self, key: str) -> Optional[str]:
        return None

    def set(self, key: str, value: str, ttl: int) -> None:
        return None

    def delete(self, key: str) -> None:
        return None

    def ping(self) -> bool:
        return True

    def close(self) -> None:
        return None


class MemoryCache:
    """进程内 TTL + LRU 缓存，Redis 不可用时的兜底（线程安全）。"""

    name = "memory"

    def __init__(self, max_entries: int = 10000):
        self._data: "OrderedDict[str, Tuple[str, float]]" = OrderedDict()
        self._max = max_entries
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            value, expire_at = item
            if expire_at and expire_at < time.time():
                self._data.pop(key, None)
                return None
            self._data.move_to_end(key)
            return value

    def set(self, key: str, value: str, ttl: int) -> None:
        expire_at = time.time() + ttl if ttl else 0.0
        with self._lock:
            self._data[key] = (value, expire_at)
            self._data.move_to_end(key)
            while len(self._data) > self._max:
                self._data.popitem(last=False)

    def delete(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)

    def ping(self) -> bool:
        return True

    def close(self) -> None:
        with self._lock:
            self._data.clear()


class RedisCache:
    """redis-py 同步后端；redis 包缺失或连接失败由 build_cache 处理。"""

    name = "redis"

    def __init__(self, settings: Settings):
        import redis  # 延迟 import：未装 redis 时仍可跑内存缓存

        self._client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            password=settings.redis_password or None,
            decode_responses=True,
            socket_timeout=settings.redis_socket_timeout,
            socket_connect_timeout=settings.redis_socket_timeout,
        )

    def get(self, key: str) -> Optional[str]:
        try:
            return self._client.get(key)
        except Exception as e:  # noqa: BLE001 - 缓存失败一律降级
            logger.warning("Redis GET failed: %s", e)
            return None

    def set(self, key: str, value: str, ttl: int) -> None:
        try:
            if ttl:
                self._client.setex(key, ttl, value)
            else:
                self._client.set(key, value)
        except Exception as e:  # noqa: BLE001
            logger.warning("Redis SET failed: %s", e)

    def delete(self, key: str) -> None:
        try:
            self._client.delete(key)
        except Exception as e:  # noqa: BLE001
            logger.warning("Redis DEL failed: %s", e)

    def ping(self) -> bool:
        self._client.ping()
        return True

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:  # noqa: BLE001
            pass


def build_cache(settings: Settings) -> CacheBackend:
    """按配置构建缓存后端，auto 模式下 Redis 失败自动降级内存。"""
    mode = (settings.cache_backend or "auto").lower()
    print(f"mode:{mode}")
    if mode == "none":
        return NullCache()
    if mode == "memory":
        return MemoryCache()
    if mode in ("redis", "auto"):
        try:
            redis_cache = RedisCache(settings)
            redis_cache.ping()
            logger.info(
                "缓存后端: redis (%s:%s/%s)",
                settings.redis_host, settings.redis_port, settings.redis_db,
            )
            return redis_cache
        except Exception as e:  # noqa: BLE001 - 连接失败/未安装 redis 类型多样
            if mode == "redis":
                raise
            logger.warning("Redis 不可用(%s)，降级为内存缓存", e)
    return MemoryCache()


class VersionedCache:
    """带命名空间 + 版本号 + TTL 的 JSON 缓存封装。"""

    kind = "generic"

    def __init__(self, backend: CacheBackend, settings: Settings):
        self.backend = backend
        self.namespace = settings.cache_namespace
        self.version = settings.resolved_cache_version
        self.ttl = settings.cache_ttl

    @property
    def prefix(self) -> str:
        return f"{self.namespace}:{self.version}:{self.kind}"

    def key(self, ident: str) -> str:
        return f"{self.prefix}:{ident}"

    def get(self, ident: str) -> Optional[Any]:
        raw = self.backend.get(self.key(ident))
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("缓存内容损坏，忽略: %s", self.key(ident))
            return None

    def set(self, ident: str, payload: Any) -> None:
        self.backend.set(
            self.key(ident),
            json.dumps(payload, ensure_ascii=False),
            self.ttl,
        )

    def delete(self, ident: str) -> None:
        self.backend.delete(self.key(ident))


class RouteCache(VersionedCache):
    """路由决策缓存：key = <ns>:<ver>:route:<pdf_hash>。"""

    kind = "route"


class ResultCache(VersionedCache):
    """解析结果缓存：key = <ns>:<ver>:result:<pdf_hash>:<fingerprint>。"""

    kind = "result"
