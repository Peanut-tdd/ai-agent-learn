"""pdfimgpipeline HTTP 客户端：用 curl 调用图片 OCR/VLM 处理接口。

协议对应 pdfimgpipeline 服务：
    POST /extract        单张  -> {image_hash, decision, reason, features, result:{engine,data}, ...}
    POST /extract:batch  多张  -> {count, ok_count, failed_count, results:[{index,filename,ok,data,error}]}
    GET  /health

设计取舍（单张 vs 批量）：
* ``describe_batch`` 优先走 **批量端点**：一次 HTTP 往返处理整篇 PDF 的所有图，
  由服务端做并发与缓存，比逐张 curl 少很多进程/连接开销；
* 批量按 ``pdfimg_batch_size`` 分组、``pdfimg_max_concurrency`` 并发，
  单张失败只影响该项，顺序与输入严格一致；
* 单张接口保留给 ``describe_one`` / 兼容旧的逐图回调。
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence

from .config import Settings
from .models import BatchOutcome
from .utils import short

logger = logging.getLogger(__name__)


def _first_result(response: Dict[str, Any]) -> Dict[str, Any]:
    results = response.get("results") or []
    return results[0] if results else {}


def to_description(response: Dict[str, Any]) -> Dict[str, str]:
    """把 pdfimgpipeline 的 /extract 返回归一化成 describer 需要的字段。"""
    result = response.get("result") or {}
    engine = str(result.get("engine") or "")
    data = result.get("data") or {}
    if not isinstance(data, dict):
        data = {}

    ocr_text = str(data.get("ocr_text") or "")
    vlm_desc = str(data.get("vlm_description") or "")
    fallback_text = str(data.get("text") or "")

    if not ocr_text and engine.startswith("ocr"):
        ocr_text = fallback_text
    if not vlm_desc and engine.startswith("vlm"):
        vlm_desc = fallback_text
    if not ocr_text and not vlm_desc and data.get("merged"):
        vlm_desc = str(data["merged"])

    return {"ocr_text": ocr_text.strip(), "vlm_description": vlm_desc.strip()}


class PdfImgPipelineClient:
    """基于 curl 的同步客户端。"""

    def __init__(self, settings: Settings):
        self.settings = settings

    # ---------------------------------------------------------------- curl
    def _curl(self, url: str, form_args: List[str], timeout: float) -> Any:
        cmd = [
            self.settings.curl_bin,
            "--silent",
            "--show-error",
            "--max-time",
            str(int(timeout)),
            "-X",
            "POST",
            url,
        ]
        if self.settings.pdfimg_api_key:
            cmd += ["-H", f"X-API-Key: {self.settings.pdfimg_api_key}"]
        for form in form_args:
            cmd += ["-F", form]

        last_error = ""
        attempts = max(0, self.settings.pdfimg_max_retries) + 1
        for attempt in range(attempts):
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=timeout + 5
                )
            except FileNotFoundError as e:
                raise RuntimeError(f"找不到 curl 可执行文件: {self.settings.curl_bin}") from e
            except subprocess.TimeoutExpired as e:
                last_error = f"curl 超时: {e}"
            else:
                if proc.returncode == 0 and proc.stdout.strip():
                    try:
                        return json.loads(proc.stdout)
                    except json.JSONDecodeError as e:
                        last_error = f"响应不是 JSON: {e}; body={proc.stdout[:200]}"
                else:
                    last_error = (
                        f"curl 退出码 {proc.returncode}: "
                        f"{(proc.stderr or proc.stdout or '').strip()[:300]}"
                    )
            if attempt < attempts - 1:
                time.sleep(0.5 * (2 ** attempt))
        raise RuntimeError(last_error or "curl 调用失败")

    # ---------------------------------------------------------------- 单张
    def extract(self, image_path: str) -> Dict[str, Any]:
        response = self._curl(
            self.settings.pdfimg_endpoint,
            [f"file=@{image_path}"],
            self.settings.pdfimg_timeout,
        )
        return response if isinstance(response, dict) else {}

    def describe_one(self, image_path: str, caption: Optional[str] = None) -> Dict[str, str]:
        return to_description(self.extract(image_path))

    # ---------------------------------------------------------------- 批量
    def _extract_group(self, paths: Sequence[str]) -> List[BatchOutcome]:
        url = f"{self.settings.pdfimg_base_url.rstrip('/')}/extract:batch"
        try:
            response = self._curl(
                url,
                [f"files=@{p}" for p in paths],
                self.settings.pdfimg_timeout,
            )
        except Exception as e:  # noqa: BLE001 - 整组传输失败，逐项标记
            logger.warning("批量图片处理失败(%d 张): %s", len(paths), e)
            return [BatchOutcome(ok=False, error=str(e)) for _ in paths]

        # 服务端返回 results 带 index，按 index 回填，缺失/乱序都能对齐
        by_index: Dict[int, Dict[str, Any]] = {}
        for item in response.get("results") or []:
            try:
                by_index[int(item.get("index"))] = item
            except (TypeError, ValueError):
                continue

        outcomes: List[BatchOutcome] = []
        for i, path in enumerate(paths):
            item = by_index.get(i)
            if item is None:
                outcomes.append(BatchOutcome(ok=False, error="批量响应缺少该索引"))
            elif item.get("ok"):
                outcomes.append(BatchOutcome(ok=True, data=item.get("data") or {}))
            else:
                outcomes.append(BatchOutcome(ok=False, error=item.get("error") or "处理失败"))
        return outcomes

    def extract_many(self, image_paths: Sequence[str]) -> List[BatchOutcome]:
        """并发批量处理，返回与输入等长的结果列表。"""
        paths = list(image_paths)
        if not paths:
            return []

        size = max(1, self.settings.pdfimg_batch_size)
        groups = [paths[i:i + size] for i in range(0, len(paths), size)]
        workers = max(1, min(self.settings.pdfimg_max_concurrency, len(groups)))

        with ThreadPoolExecutor(max_workers=workers) as pool:
            group_results = list(pool.map(self._extract_group, groups))

        ordered: List[BatchOutcome] = []
        for result in group_results:
            ordered.extend(result)
        return ordered

    # ---------------------------------------------------------------- 描述
    def describe_batch(
        self, items: Sequence[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """批量语义化。

        items: [{"path": str, "caption": str|None}, ...]
        返回: [{"ocr_text": str, "vlm_description": str, "error": str|None}, ...]
        """
        paths = [it["path"] for it in items]
        outcomes = self.extract_many(paths)

        descriptions: List[Dict[str, str]] = []
        for item, outcome in zip(items, outcomes):
            if not outcome.ok:
                descriptions.append(
                    {"ocr_text": "", "vlm_description": "", "error": outcome.error}
                )
                continue
            if self.settings.pdfimg_action.lower() == "route":
                # 只做了路由决策，没有正文
                descriptions.append({"ocr_text": "", "vlm_description": "", "error": None})
                continue
            desc = to_description(outcome.data or {})
            desc["error"] = None
            descriptions.append(desc)
            _ = item  # caption 目前不参与请求，保留接口便于后续 prompt 定制
        logger.info(
            "图片语义化: %d 张，失败 %d 张",
            len(items), sum(1 for d in descriptions if d.get("error")),
        )
        return descriptions

    # ---------------------------------------------------------------- 健康
    def health(self) -> Dict[str, Any]:
        url = f"{self.settings.pdfimg_base_url.rstrip('/')}/health"
        cmd = [self.settings.curl_bin, "--silent", "--show-error",
               "--max-time", str(int(self.settings.pdfimg_timeout)), url]
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=self.settings.pdfimg_timeout + 5)
        if proc.returncode != 0:
            raise RuntimeError((proc.stderr or "health 失败").strip()[:200])
        return json.loads(proc.stdout)

    def fingerprint(self) -> str:
        """缓存指纹：服务地址+动作+密钥是否启用（密钥本身不进指纹）。"""
        return (
            f"pdfimg:{self.settings.pdfimg_base_url}"
            f"|{self.settings.pdfimg_action}"
            f"|auth={int(bool(self.settings.pdfimg_api_key))}"
        )

    # 兼容旧的短接口
    def short(self, value: str) -> str:  # pragma: no cover - 调试用
        return short(value)
