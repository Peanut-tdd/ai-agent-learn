"""小工具：图片内容哈希。"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Union


def image_sha256(path: Union[str, Path]) -> str:
    """图片内容 SHA-256，作为缓存键：同图换路径也能命中。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
