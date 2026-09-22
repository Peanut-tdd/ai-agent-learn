"""小工具：文件内容哈希。"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Union


def file_sha256(path: Union[str, Path], chunk_size: int = 1 << 20) -> str:
    """文件内容 SHA-256，作为缓存键：同文件换路径也能命中。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def short(text: str, n: int = 8) -> str:
    return (text or "")[:n]
