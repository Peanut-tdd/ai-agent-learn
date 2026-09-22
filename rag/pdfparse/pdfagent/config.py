"""集中配置：全部可用环境变量 / .env 覆盖。

不依赖 pydantic，纯标准库实现，保证在只装了 pymupdf 的环境也能 import。
字段名小写即对应大写环境变量（大小写不敏感），例如 ``cache_ttl`` -> ``CACHE_TTL``。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

_TRUE = {"1", "true", "yes", "on", "y"}
_FALSE = {"0", "false", "no", "off", "n", ""}


def load_dotenv(path: str = ".env") -> None:
    """极简 .env 加载：已存在的环境变量优先，不覆盖。"""
    if not path or not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                os.environ.setdefault(key, value)


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None else value


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    raw = raw.strip().lower()
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    """运行期配置快照。"""

    # ---------------- 缓存 ----------------
    cache_backend: str = "auto"          # auto(先试 redis) | redis | memory | none
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str = ""
    redis_socket_timeout: float = 2.0
    cache_ttl: int = 7 * 24 * 3600
    cache_namespace: str = "rag02"       # 缓存命名空间（多项目共用同一 Redis 时隔离）
    cache_version: str = ""              # 空 = 自动使用 __version__+schema，项目更新自动失效

    # ---------------- PDF 预检 ----------------
    text_page_min_chars: int = 100       # 去空白字符数 >= 此值 -> 文本页
    sparse_page_min_chars: int = 10      # 有少量文本但不足文本页 -> 混合页
    scanned_image_area_ratio: float = 0.5  # 无文本页中图片覆盖面积占比 >= 此值 -> 扫描页
    scanned_ratio_threshold: float = 0.5   # 扫描页占比 >= 此值 -> 整体 scanned
    text_ratio_threshold: float = 0.9      # 文本页占比 >= 此值 -> 整体 text，否则 hybrid

    # ---------------- MinerU ----------------
    mineru_backend: str = "auto"         # auto | cli | http
    mineru_command: str = "mineru"       # 或 magic-pdf
    mineru_mode: str = "auto"            # auto | txt | ocr
    mineru_extra_args: str = ""          # 追加参数，shell 风格
    mineru_http_url: str = ""            # backend=http 时的接口地址
    mineru_timeout: float = 1800.0
    mineru_fallback_text: bool = False   # MinerU 不可用时是否降级回 pymupdf
    mineru_describe_images: bool = False  # MinerU 路由是否再调 pdfimgpipeline 描述图片

    # ---------------- 图片解析（pdfimgpipeline HTTP 服务） ----------------
    pdfimg_base_url: str = "http://127.0.0.1:8000"
    pdfimg_api_key: str = ""             # 对应服务的 PDFIMG_API_KEY
    pdfimg_action: str = "extract"       # extract(完整) | route(仅决策)
    pdfimg_timeout: float = 60.0
    pdfimg_batch_size: int = 16          # 单次 /extract:batch 最多张数
    pdfimg_max_concurrency: int = 4      # 批量分组并发上限
    pdfimg_max_retries: int = 1          # 瞬时失败重试次数
    curl_bin: str = "curl"               # curl 可执行文件路径

    # ---------------- 运行 ----------------
    log_level: str = "INFO"

    @property
    def resolved_cache_version(self) -> str:
        """缓存版本号：显式配置优先，否则由包版本 + 结构版本推导。"""
        if self.cache_version:
            return self.cache_version
        from . import SCHEMA_VERSION, __version__

        return f"{__version__}+s{SCHEMA_VERSION}"

    @property
    def mineru_resolved_backend(self) -> str:
        backend = self.mineru_backend.lower()
        if backend in ("cli", "http"):
            return backend
        return "http" if self.mineru_http_url else "cli"

    @property
    def pdfimg_endpoint(self) -> str:
        base = self.pdfimg_base_url.rstrip("/")
        action = self.pdfimg_action.lower()
        return f"{base}/route" if action == "route" else f"{base}/extract"

    @classmethod
    def from_env(cls, *, env_file: Optional[str] = ".env") -> "Settings":
        if env_file:
            load_dotenv(env_file)
        return cls(
            cache_backend=_env("PDF_CACHE_BACKEND", _env("CACHE_BACKEND", "auto")),
            redis_host=_env("REDIS_HOST", "localhost"),
            redis_port=_env_int("REDIS_PORT", 6379),
            redis_db=_env_int("REDIS_DB", 0),
            redis_password=_env("REDIS_PASSWORD", ""),
            redis_socket_timeout=_env_float("REDIS_SOCKET_TIMEOUT", 2.0),
            cache_ttl=_env_int("CACHE_TTL", 7 * 24 * 3600),
            cache_namespace=_env("PDF_CACHE_NS", "rag02"),
            cache_version=_env("PDF_CACHE_VERSION", ""),
            text_page_min_chars=_env_int("PDF_TEXT_PAGE_MIN_CHARS", 100),
            sparse_page_min_chars=_env_int("PDF_SPARSE_PAGE_MIN_CHARS", 10),
            scanned_image_area_ratio=_env_float("PDF_SCANNED_IMAGE_AREA_RATIO", 0.5),
            scanned_ratio_threshold=_env_float("PDF_SCANNED_RATIO_THRESHOLD", 0.5),
            text_ratio_threshold=_env_float("PDF_TEXT_RATIO_THRESHOLD", 0.9),
            mineru_backend=_env("MINERU_BACKEND", "auto"),
            mineru_command=_env("MINERU_COMMAND", "mineru"),
            mineru_mode=_env("MINERU_MODE", "auto"),
            mineru_extra_args=_env("MINERU_EXTRA_ARGS", ""),
            mineru_http_url=_env("MINERU_HTTP_URL", ""),
            mineru_timeout=_env_float("MINERU_TIMEOUT", 1800.0),
            mineru_fallback_text=_env_bool("MINERU_FALLBACK_TEXT", False),
            mineru_describe_images=_env_bool("MINERU_DESCRIBE_IMAGES", False),
            pdfimg_base_url=_env("PDFIMG_BASE_URL", "http://127.0.0.1:8000"),
            pdfimg_api_key=_env("PDFIMG_API_KEY", ""),
            pdfimg_action=_env("PDFIMG_ACTION", "extract"),
            pdfimg_timeout=_env_float("PDFIMG_TIMEOUT", 60.0),
            pdfimg_batch_size=_env_int("PDFIMG_BATCH_SIZE", 16),
            pdfimg_max_concurrency=_env_int("PDFIMG_MAX_CONCURRENCY", 4),
            pdfimg_max_retries=_env_int("PDFIMG_MAX_RETRIES", 1),
            curl_bin=_env("CURL_BIN", "curl"),
            log_level=_env("LOG_LEVEL", "INFO"),
        )

    def replace(self, **changes) -> "Settings":
        import dataclasses

        return dataclasses.replace(self, **changes)
