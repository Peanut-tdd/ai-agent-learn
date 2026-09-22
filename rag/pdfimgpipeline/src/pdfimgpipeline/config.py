"""集中配置：全部可用环境变量 / .env 覆盖。

字段名小写即对应大写环境变量（大小写不敏感），例如 `vlm_model` -> `VLM_MODEL`。
"""

from __future__ import annotations

from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_VLM_PROMPT = (
    "你是文档图像阅读器。请描述这张图：图类型（表格/折线图/结构式/扫描文字…）、"
    "坐标轴与单位、分组与曲线含义、图中可读的关键数值与结论。只输出正文，中文，"
    "不超过 150 字，不推断图中没有的信息。"
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---------------- VLM ----------------
    vlm_model: str = "qwen-vl-max"
    vlm_api_key: str = ""
    vlm_base_url: str = ""
    openai_api_key: str = ""      # 回退别名（与仓库其它模块一致）
    openai_base_url: str = ""
    vlm_timeout: float = 30.0     # 编排层兜底超时
    vlm_sdk_timeout: float = 28.0  # SDK 层超时（略小，保证真断连）
    vlm_max_concurrency: int = 4
    vlm_prompt: str = DEFAULT_VLM_PROMPT

    # ---------------- OCR ----------------
    ocr_timeout: float = 15.0
    ocr_max_workers: int = 4
    ocr_serialize: bool = True    # RapidOCR 引擎调用是否加锁串行化（默认安全）

    # ---------------- 路由阈值 ----------------
    text_ratio_threshold: float = 0.15
    entropy_threshold: float = 6.5
    edge_density_threshold: float = 0.12
    color_var_threshold: float = 800.0
    ocr_threshold_chars: int = 50
    ocr_threshold_conf: float = 0.85
    min_probe_chars_for_hybrid: int = 10

    # ---------------- 执行 ----------------
    max_retries: int = 2
    retry_backoff: float = 0.5

    # ---------------- 缓存 ----------------
    cache_backend: str = "auto"   # auto | redis | memory | none
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str = ""
    redis_socket_timeout: float = 2.0
    cache_ttl: int = 7 * 24 * 3600

    # ---------------- 服务 ----------------
    pdfimg_api_key: str = ""      # 留空 = 不鉴权（内网/本机）
    pdfimg_host: str = "0.0.0.0"
    pdfimg_port: int = 8000
    max_upload_mb: int = 16
    max_batch_size: int = 32      # 单次批量请求最多图片数
    batch_max_concurrency: int = 8  # 批量内部并发上限
    log_level: str = "INFO"

    @property
    def resolved_vlm_api_key(self) -> str:
        return self.vlm_api_key or self.openai_api_key

    @property
    def resolved_vlm_base_url(self) -> Optional[str]:
        return self.vlm_base_url or self.openai_base_url or None

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024
