#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""describe_images.py — 图片语义化：curl 调用 pdfimgpipeline 处理接口。

本项目的图片 OCR / VLM 描述不再本地实现，而是交给独立的
``pdfimgpipeline`` HTTP 服务（见 ../pdfimgpipeline）：

    POST /extract        单张图片 -> OCR / VLM / hybrid 结果
    POST /extract:batch  多张图片 -> 并发处理，单项失败隔离

集成方式：
* ``layout_parser`` 解析到每张图时回调 describer，把 OCR 文字 + VLM 描述拼进
  图 chunk 的正文，于是 layout.json / chunks.md / report.html 自动可见；
* **整篇图一次性批量调用** ``/extract:batch``（而不是逐张 curl），
  由服务端做并发、缓存与降级；批量失败自动回退逐张。

配置（环境变量 / .env，见 pdfagent/config.py）::

    PDFIMG_BASE_URL=http://127.0.0.1:8000   # pdfimgpipeline 服务地址
    PDFIMG_API_KEY=                         # 服务端设了鉴权时填写
    PDFIMG_ACTION=extract                   # extract(默认) | route(仅决策)
    PDFIMG_TIMEOUT=60
    PDFIMG_BATCH_SIZE=16
    PDFIMG_MAX_CONCURRENCY=4

自检::

    python describe_images.py               # 打印配置并探测服务 /health
"""
from __future__ import annotations

import os
import tempfile

from pdfagent.config import Settings
from pdfagent.image_client import PdfImgPipelineClient

#: 兼容旧接口的默认 prompt 说明（实际 prompt 由 pdfimgpipeline 服务端配置）
VLM_PROMPT = (
    "你是医药文档图表阅读器。请描述这张图：图类型、坐标轴与单位、分组与曲线含义、"
    "图中可读的关键数值/结论。只输出描述正文，中文，不超过 150 字。"
)


class PdfImgPipelineDescriber:
    """把 pdfimgpipeline HTTP 客户端适配成 layout_parser 的 describer。

    * ``describe_batch(items)``：批量语义化，items = [{"path","caption"}, ...]；
    * ``__call__(image_bytes, caption)``：兼容旧逐图回调（落临时文件后单张调用）；
    * ``fingerprint()``：参与结果缓存 key，服务地址变化即失效。
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings.from_env()
        self.client = PdfImgPipelineClient(self.settings)

    # ---------------------------------------------------------------- 批量
    def describe_batch(self, items) -> list[dict]:
        return self.client.describe_batch(items)

    # ---------------------------------------------------------------- 单张
    def __call__(self, image_bytes: bytes, caption: str | None = None) -> dict:
        suffix = ".png"
        fd, path = tempfile.mkstemp(prefix="describe_", suffix=suffix)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(image_bytes or b"")
            desc = self.client.describe_one(path, caption)
            return {"ocr_text": desc.get("ocr_text", ""),
                    "vlm_description": desc.get("vlm_description", "")}
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    # ---------------------------------------------------------------- 其他
    def fingerprint(self) -> str:
        return self.client.fingerprint()


# --------------------------------------------------------------- 兼容入口
def _client(settings: Settings | None = None) -> PdfImgPipelineClient:
    return PdfImgPipelineClient(settings or Settings.from_env())


def ocr_image(image_bytes: bytes, caption: str | None = None) -> str:
    """单张图片的 OCR 文字（走 pdfimgpipeline）。保留旧签名便于外部调用。"""
    return PdfImgPipelineDescriber()(image_bytes, caption).get("ocr_text", "")


def vlm_caption(image_bytes: bytes, caption: str | None = None) -> str:
    """单张图片的 VLM 描述（走 pdfimgpipeline）。保留旧签名便于外部调用。"""
    return PdfImgPipelineDescriber()(image_bytes, caption).get("vlm_description", "")


def build_describer(settings: Settings | None = None) -> PdfImgPipelineDescriber:
    """构造传给 ``parse_document(image_describer=...)`` 的回调对象。"""
    describer = PdfImgPipelineDescriber(settings)
    s = describer.settings
    print(f"[describe] 图片语义化 -> {s.pdfimg_base_url} "
          f"(action={s.pdfimg_action}, batch={s.pdfimg_batch_size}, "
          f"concurrency={s.pdfimg_max_concurrency})")
    return describer


if __name__ == "__main__":
    settings = Settings.from_env()
    print("describe_images 配置:")
    print(f"  PDFIMG_BASE_URL      = {settings.pdfimg_base_url}")
    print(f"  PDFIMG_ACTION        = {settings.pdfimg_action}")
    print(f"  PDFIMG_BATCH_SIZE    = {settings.pdfimg_batch_size}")
    print(f"  curl                 = {settings.curl_bin}")
    try:
        health = _client(settings).health()
        print(f"  服务 /health         = {health}")
    except Exception as e:  # noqa: BLE001
        print(f"  服务不可用（先启动 pdfimgpipeline）: {e}")
