"""MinerU 版式解析适配器：混合型 / 扫描型 PDF 走这里。

支持两种后端：
* ``cli``  —— 调用 ``mineru``（或 ``magic-pdf``）命令行，参数 ``-p/-o/-m``；
* ``http`` —— 调用远端 MinerU 服务，返回 JSON（markdown / content_list）。

输出统一归一化为与 pymupdf 适配器一致的 result 契约：
    meta / sections / tables / figures / chunks / elements

不再依赖 MinerU 具体版本：优先读 ``*_content_list.json``，没有则退化为对
``*.md`` 做规则切分。
"""

from __future__ import annotations

import glob
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..config import Settings

logger = logging.getLogger(__name__)

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


class MinerUAdapter:
    name = "mineru"

    def __init__(self, settings: Settings):
        self.settings = settings

    # ---------------------------------------------------------------- 可用性
    def available(self) -> bool:
        if self.settings.mineru_resolved_backend == "http":
            return bool(self.settings.mineru_http_url)
        return shutil.which(self.settings.mineru_command) is not None

    # ---------------------------------------------------------------- 入口
    def parse(
        self,
        pdf_path: str,
        out_dir: str,
        *,
        chunk_limit: int = 600,
        image_describer: Optional[object] = None,
    ) -> Dict[str, Any]:
        if not self.available():
            raise RuntimeError(
                f"MinerU 不可用：backend={self.settings.mineru_resolved_backend} "
                f"command={self.settings.mineru_command!r}；请安装 MinerU、配置 "
                f"MINERU_HTTP_URL，或使用 --fallback-text 降级到 pymupdf"
            )

        backend = self.settings.mineru_resolved_backend
        if backend == "http":
            raw = self._run_http(pdf_path, out_dir)
        else:
            raw = self._run_cli(pdf_path, out_dir)
        return self._normalize(pdf_path, out_dir, raw, chunk_limit, image_describer)

    # ---------------------------------------------------------------- CLI
    def _run_cli(self, pdf_path: str, out_dir: str) -> Dict[str, Any]:
        os.makedirs(out_dir, exist_ok=True)
        cmd = [
            self.settings.mineru_command,
            "-p", pdf_path,
            "-o", out_dir,
            "-m", self.settings.mineru_mode,
        ]
        if self.settings.mineru_extra_args:
            cmd += shlex.split(self.settings.mineru_extra_args)

        logger.info("MinerU CLI: %s", " ".join(cmd))
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.settings.mineru_timeout,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"MinerU CLI 退出码 {proc.returncode}: "
                f"{(proc.stderr or proc.stdout or '').strip()[-500:]}"
            )

        stem = os.path.splitext(os.path.basename(pdf_path))[0]
        md_path, content_path = self._locate_outputs(out_dir, stem)
        content_list = self._read_json(content_path) if content_path else []
        markdown = self._read_text(md_path) if md_path else ""
        return {
            "backend": "cli",
            "output_dir": os.path.dirname(content_path or md_path or out_dir),
            "markdown_path": md_path,
            "content_list_path": content_path,
            "markdown": markdown,
            "content_list": content_list or [],
            "stdout": proc.stdout[-2000:],
        }

    def _locate_outputs(self, out_dir: str, stem: str) -> Tuple[Optional[str], Optional[str]]:
        def pick(pattern: str) -> Optional[str]:
            matches = glob.glob(os.path.join(out_dir, "**", pattern), recursive=True)
            if not matches:
                return None
            preferred = [m for m in matches if stem and stem in os.path.basename(m)]
            pool = preferred or matches
            return max(pool, key=os.path.getmtime)

        content = pick("*_content_list.json")
        md = pick("*.md")
        return md, content

    @staticmethod
    def _read_json(path: str) -> Any:
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:  # noqa: BLE001
            logger.warning("读取 MinerU JSON 失败 %s: %s", path, e)
            return []

    @staticmethod
    def _read_text(path: str) -> str:
        try:
            with open(path, encoding="utf-8") as f:
                return f.read()
        except Exception as e:  # noqa: BLE001
            logger.warning("读取 MinerU Markdown 失败 %s: %s", path, e)
            return ""

    # ---------------------------------------------------------------- HTTP
    def _run_http(self, pdf_path: str, out_dir: str) -> Dict[str, Any]:
        url = self.settings.mineru_http_url
        cmd = [
            self.settings.curl_bin, "--silent", "--show-error",
            "--max-time", str(int(self.settings.mineru_timeout)),
            "-X", "POST", url, "-F", f"file=@{pdf_path}",
        ]
        logger.info("MinerU HTTP: %s", url)
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=self.settings.mineru_timeout + 5
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"MinerU HTTP 失败: {(proc.stderr or proc.stdout or '').strip()[:300]}"
            )
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"MinerU HTTP 响应不是 JSON: {e}") from e

        content_list = payload.get("content_list") or []
        markdown = payload.get("markdown") or ""
        return {
            "backend": "http",
            "output_dir": payload.get("output_dir") or out_dir,
            "markdown_path": None,
            "content_list_path": None,
            "markdown": markdown,
            "content_list": content_list,
            "stdout": "",
        }

    # ---------------------------------------------------------------- 归一化
    def _normalize(
        self,
        pdf_path: str,
        out_dir: str,
        raw: Dict[str, Any],
        chunk_limit: int,
        image_describer: Optional[object],
    ) -> Dict[str, Any]:
        meta = self._build_meta(pdf_path, raw)
        content_list = raw.get("content_list") or []
        if content_list:
            sections, tables, figures, chunks = self._from_content_list(
                content_list, raw, meta, out_dir, chunk_limit, image_describer
            )
        else:
            sections, tables, figures, chunks = self._from_markdown(
                raw.get("markdown") or "", meta, chunk_limit
            )
        return {
            "engine": self.name,
            "meta": meta,
            "sections": sections,
            "tables": tables,
            "figures": figures,
            "chunks": chunks,
            "elements": [],
            "raw": {
                "backend": raw.get("backend"),
                "markdown_path": raw.get("markdown_path"),
                "content_list_path": raw.get("content_list_path"),
            },
        }

    def _build_meta(self, pdf_path: str, raw: Dict[str, Any]) -> Dict[str, Any]:
        content_list = raw.get("content_list") or []
        pages = 0
        for item in content_list:
            try:
                pages = max(pages, int(item.get("page_idx", 0)) + 1)
            except (TypeError, ValueError):
                continue
        meta: Dict[str, Any] = {
            "file": os.path.basename(pdf_path),
            "pages": pages,
            "doc_id": "",
            "doc_type": "药品说明书",
            "engine": self.name,
        }
        m = re.search(r"[A-Z]{4}\d{7}(-\d+)?", os.path.basename(pdf_path))
        if m:
            meta["doc_id"] = m.group(0)
        return meta

    # -------------------------------------------------- content_list -> 结构
    def _from_content_list(
        self,
        content_list: Sequence[Dict[str, Any]],
        raw: Dict[str, Any],
        meta: Dict[str, Any],
        out_dir: str,
        chunk_limit: int,
        image_describer: Optional[object],
    ) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
        base = raw.get("output_dir") or out_dir
        table_dir = os.path.join(out_dir, "tables")
        os.makedirs(table_dir, exist_ok=True)
        os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)

        # 先批量把图片送 pdfimgpipeline 语义化（可选），再按顺序建 chunk
        image_descs = self._describe_images(content_list, base, out_dir, image_describer)

        sections: List[Dict] = []
        section_items: Dict[str, Dict] = {}

        def touch_section(title: str, page: int) -> None:
            if title not in section_items:
                section_items[title] = {
                    "title": title, "page_start": page, "page_end": page, "n_items": 0
                }
                sections.append(section_items[title])
            section_items[title]["page_end"] = max(section_items[title]["page_end"], page)

        touch_section("文档头", 1)

        tables: List[Dict] = []
        figures: List[Dict] = []
        chunks: List[Dict] = []
        current_section = "文档头"
        buf: List[str] = []
        buf_start = 1
        last_page = 1

        def add_chunk(kind: str, content: str, page_start: int, page_end: int,
                      section: str, caption: Optional[str] = None) -> None:
            chunks.append({
                "id": f"c{len(chunks) + 1:03d}",
                "type": kind,
                "section": section,
                "sub_section": None,
                "caption": caption,
                "content": content,
                "page_start": page_start,
                "page_end": page_end,
                "char_len": len(content),
                "meta": {
                    "doc_id": meta.get("doc_id", ""),
                    "doc_type": meta.get("doc_type", ""),
                    "drug_name": meta.get("通用名称", ""),
                    "drug_alias": meta.get("商品名称", ""),
                },
            })
            section_items.setdefault(section, {
                "title": section, "page_start": page_start,
                "page_end": page_end, "n_items": 0,
            })["n_items"] += 1

        def flush_text() -> None:
            nonlocal buf
            if buf:
                add_chunk("text", "\n".join(buf), buf_start, last_page, current_section)
                buf = []

        for idx, item in enumerate(content_list):
            item_type = item.get("type")
            try:
                page = int(item.get("page_idx", 0)) + 1
            except (TypeError, ValueError):
                page = 1
            last_page = page

            if item_type == "text":
                text = (item.get("text") or "").strip()
                if not text:
                    continue
                if item.get("text_level"):
                    flush_text()
                    current_section = text
                    touch_section(current_section, page)
                    section_items[current_section]["n_items"] += 1
                    continue
                if buf and sum(len(x) for x in buf) + len(text) > chunk_limit:
                    flush_text()
                if not buf:
                    buf_start = page
                buf.append(text)

            elif item_type == "table":
                flush_text()
                caption = " ".join(item.get("table_caption") or []) or None
                body = item.get("table_body") or ""
                content = f"{caption}\n{body}" if caption else body
                tid = f"table-{len(tables) + 1:02d}"
                rel = f"tables/{tid}.md"
                with open(os.path.join(out_dir, rel), "w", encoding="utf-8") as f:
                    f.write(content + "\n")
                add_chunk("table", content, page, page, current_section, caption)
                tables.append({
                    "id": tid, "caption": caption, "section": current_section,
                    "page_start": page, "page_end": page,
                    "rows": body.count("<tr>"), "cols": 0,
                    "merged_fragments": 1, "file": rel,
                })

            elif item_type == "image":
                flush_text()
                caption = " ".join(item.get("image_caption") or []) or None
                rel = self._relocate_image(item.get("img_path"), base, out_dir)
                desc = image_descs.get(idx) or {}
                desc_clean = {
                    k: str(desc.get(k) or "").strip()
                    for k in ("ocr_text", "vlm_description")
                }
                desc_clean = {k: v for k, v in desc_clean.items() if v}
                content = "\n".join(
                    x for x in (caption, desc_clean.get("ocr_text"),
                                desc_clean.get("vlm_description"), rel) if x
                )
                add_chunk("figure", content, page, page, current_section, caption)
                fid = f"figure-{len(figures) + 1:02d}"
                figures.append({
                    "id": fid, "caption": caption,
                    "description": desc_clean or None,
                    "description_error": desc.get("error"),
                    "section": current_section, "page": page, "file": rel,
                })

        flush_text()
        sections = [s for s in sections if s["n_items"] or s["title"] != "文档头"] or sections
        return sections, tables, figures, chunks

    def _relocate_image(self, img_path: Optional[str], base: str, out_dir: str) -> Optional[str]:
        if not img_path:
            return None
        source = img_path if os.path.isabs(img_path) else os.path.join(base, img_path)
        if not os.path.isfile(source):
            return img_path
        try:
            return os.path.relpath(source, out_dir)
        except ValueError:
            return source

    def _describe_images(
        self,
        content_list: Sequence[Dict[str, Any]],
        base: str,
        out_dir: str,
        image_describer: Optional[object],
    ) -> Dict[int, Dict[str, str]]:
        """MinerU 已做 OCR/VLM；仅在显式开启时再走 pdfimgpipeline。"""
        if image_describer is None or not self.settings.mineru_describe_images:
            return {}
        items: List[Dict[str, Any]] = []
        indices: List[int] = []
        for idx, item in enumerate(content_list):
            if item.get("type") != "image":
                continue
            img_path = item.get("img_path")
            if not img_path:
                continue
            source = img_path if os.path.isabs(img_path) else os.path.join(base, img_path)
            if not os.path.isfile(source):
                continue
            items.append({"path": source, "caption": " ".join(item.get("image_caption") or [])})
            indices.append(idx)
        if not items:
            return {}
        describe_batch = getattr(image_describer, "describe_batch", None)
        if not callable(describe_batch):
            return {}
        try:
            results = describe_batch(items)
        except Exception as e:  # noqa: BLE001
            logger.warning("MinerU 图片语义化失败: %s", e)
            return {}
        return {idx: res for idx, res in zip(indices, results)}

    # -------------------------------------------------- markdown -> 结构
    def _from_markdown(
        self, markdown: str, meta: Dict[str, Any], chunk_limit: int
    ) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
        sections: List[Dict] = []
        chunks: List[Dict] = []
        current_section = "文档头"
        buf: List[str] = []

        def add(title: str, content: str) -> None:
            chunks.append({
                "id": f"c{len(chunks) + 1:03d}",
                "type": "text",
                "section": title,
                "sub_section": None,
                "caption": None,
                "content": content,
                "page_start": 1,
                "page_end": 1,
                "char_len": len(content),
                "meta": {
                    "doc_id": meta.get("doc_id", ""),
                    "doc_type": meta.get("doc_type", ""),
                    "drug_name": "", "drug_alias": "",
                },
            })

        def flush() -> None:
            nonlocal buf
            text = "\n".join(buf).strip()
            if text:
                add(current_section, text)
            buf = []

        for line in (markdown or "").splitlines():
            m = _HEADING_RE.match(line.strip())
            if m:
                flush()
                current_section = m.group(2).strip()
                continue
            buf.append(line)
            if sum(len(x) for x in buf) > chunk_limit:
                flush()
        flush()

        page = 1
        sections.append({
            "title": current_section, "page_start": page,
            "page_end": page, "n_items": len(chunks),
        })
        return sections, [], [], chunks
