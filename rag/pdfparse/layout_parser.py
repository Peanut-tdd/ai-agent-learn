"""版面解析核心：把药品说明书 PDF 解析为结构化结果（章节 / 段落 / 表格 / 图片 / chunk）。

解析策略（针对 CDE 药品说明书的规整版式，用规则而非版面模型）：
1. 文本块按行合并（中英文混排 span 碎裂，按行拼接还原）；
2. 【...】独占一行的块 → 一级章节标题（全文统一 12pt，字体无层级特征，只能靠正则）；
3. 加粗且以「表N/图N」开头 → 表图题，挂到相邻表格/图片上；
4. 加粗独立行 → 小节标题（如「低血糖症」），作为 chunk 的子章节元数据；
   判定要求：整段只含一个非空行、无句末标点、不是脚注样式；
   这样「…(汇总组 II)」这类较长的粗体标题不会被当成正文，
   而跨页续写的粗体句尾（以「。」结尾）也不会被误当成标题；
5. find_tables() 检测表格区域，重叠文本块丢弃，避免表格文字被打散进段落；
6. 连续同列数表格合并（跨页断表 + 同页碎片表），去掉重复表头行；
7. 按章节组织 chunk：短章节整章一个 chunk，长章节按小节/段落累积到上限，
   表格、图片始终是原子 chunk。
"""
from __future__ import annotations

import contextlib
import io
import os
import re

import pymupdf

SECTION_RE = re.compile(r"^【[^】]{1,20}】$")
CAPTION_RE = re.compile(r"^(表|图)\s*\d+\s*[:：]?\s*")
SUBHEAD_MAX_LEN = 30        # 普通小节标题字符上限（如「低血糖症」）
SUBHEAD_TITLE_MAX_LEN = 50  # 无句末标点、非脚注样式的标题可放宽到此上限
SENTENCE_END_RE = re.compile(r"[。；！？]$")
FOOTNOTE_MARK_RE = re.compile(r"^(?:[\*†‡§¶]|[a-z]\s)")
CHUNK_LIMIT = 600  # 文本 chunk 字符上限；表格/图片为原子 chunk 不受限


# ---------------------------------------------------------------- 基础工具

def _is_bold(line) -> bool:
    return any((s["flags"] & 16) or ("Bold" in s["font"]) for s in line["spans"])


def _looks_like_subheading(text: str) -> bool:
    """粗体独立行是否为小节标题（排除粗体正文与脚注）。

    说明书里正文跨页时，续写的半句常被单独排成一行且带粗体（如
    「(FPG)得到显著改善(见表8)。」），而真正的小节标题不会以句末
    标点结尾；较长的标题（如「…(汇总组 II)」）也不算句子。脚注以
    */†/‡/§ 或字母编号开头，且常含「=」等符号，一并排除。
    """
    if SENTENCE_END_RE.search(text):
        return False
    if len(text) <= SUBHEAD_MAX_LEN:
        return True
    return (len(text) <= SUBHEAD_TITLE_MAX_LEN
            and not FOOTNOTE_MARK_RE.match(text)
            and "=" not in text)


def _join_lines(texts) -> str:
    """块内多行拼成一段：英文断行补空格，中文直接拼接。"""
    out = ""
    for t in texts:
        t = t.strip()
        if not t:
            continue
        if out and out[-1].isascii() and out[-1].isalnum() and t[0].isascii() and t[0].isalnum():
            out += " " + t
        else:
            out += t
    return out


def _overlap_ratio(inner_bbox, outer_bbox) -> float:
    r = pymupdf.Rect(inner_bbox) & pymupdf.Rect(outer_bbox)
    if r.is_empty:
        return 0.0
    area = pymupdf.Rect(inner_bbox).get_area()
    return r.get_area() / area if area else 0.0


def _norm_row(row) -> str:
    return "|".join((c or "").replace("\n", "").strip() for c in row)


def table_to_markdown(rows) -> str:
    def cell(c):
        return (c or "").replace("\n", "<br>").replace("|", "\\|").strip()

    width = max(len(r) for r in rows)  # 合并表头碎片后各行列数可能不一致，按最宽行补齐
    header = list(rows[0]) + [None] * (width - len(rows[0]))
    lines = [
        "| " + " | ".join(cell(c) for c in header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for r in rows[1:]:
        r = list(r) + [None] * (width - len(r))
        lines.append("| " + " | ".join(cell(c) for c in r[:width]) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------- 元素提取

def _extract_elements(page, pno: int) -> list[dict]:
    """提取一页上的元素：heading / subheading / caption / paragraph / table / figure。"""
    tables = []
    # find_tables 缺 pymupdf_layout 时每页都会打印提示，静默掉。
    # lines_strict：只按明确框线检测，避免把对齐段落误判为表格。
    with contextlib.redirect_stdout(io.StringIO()):
        finder = page.find_tables(vertical_strategy="lines_strict",
                                  horizontal_strategy="lines_strict")
    for t in finder.tables:
        tables.append({
            "kind": "table", "page": pno, "page_end": pno, "bbox": list(t.bbox),
            "rows": t.extract(), "col_count": t.col_count,
        })

    elements = []
    for b in page.get_text("dict")["blocks"]:
        if b["type"] == 1:  # 图片块
            elements.append({
                "kind": "figure", "page": pno, "bbox": list(b["bbox"]),
                "ext": b.get("ext", "png"), "image": b.get("image"),
            })
            continue
        # 与表格区域重叠过半的文本块视为表内文字，丢弃（表格已整体结构化）
        if any(_overlap_ratio(b["bbox"], t["bbox"]) > 0.5 for t in tables):
            continue
        # PyMuPDF 有时把【】标题行并进相邻文本块，按行切出标题行再分类其余行
        seg_lines: list = []

        def flush_seg():
            if not seg_lines:
                return
            text = _join_lines("".join(s["text"] for s in l["spans"]) for l in seg_lines)
            if not text:
                seg_lines.clear()
                return
            bold = any(_is_bold(l) for l in seg_lines)
            if bold and CAPTION_RE.match(text):
                kind = "caption"
            elif bold and len(seg_lines) == 1 and _looks_like_subheading(text):
                kind = "subheading"
            else:
                kind = "paragraph"
            bbox = seg_lines[0]["bbox"]
            for l in seg_lines[1:]:
                bbox = list(pymupdf.Rect(bbox) | pymupdf.Rect(l["bbox"]))
            elements.append({"kind": kind, "page": pno, "bbox": bbox, "text": text})
            seg_lines.clear()

        for l in b["lines"]:
            line_text = "".join(s["text"] for s in l["spans"]).strip()
            if SECTION_RE.match(line_text):
                flush_seg()
                elements.append({"kind": "heading", "page": pno,
                                 "bbox": list(l["bbox"]), "text": line_text})
            elif line_text:
                # 空行（块内用于分段的空白 span）不参与合并与分类，
                # 否则它会让「空行 + 粗体标题」被当成多行段落（如「…汇总组 II」）。
                seg_lines.append(l)
        flush_seg()

    elements.extend(tables)
    elements.sort(key=lambda e: (e["page"], e["bbox"][1], e["bbox"][0]))
    return elements


def _merge_tables(elements: list[dict]) -> list[dict]:
    """合并连续表格：跨页断表（续页重复表头）与同页碎片表。相邻表格间无任何元素才合并。"""
    out = []
    for el in elements:
        prev = out[-1] if out else None
        if el["kind"] == "table" and prev and prev["kind"] == "table":
            same_cols = el["col_count"] == prev["col_count"]
            adjacent = el["page"] - prev["page_end"] <= 1
            # 跨页断表（续页重复表头）与同页碎片表：列数相同且相邻
            # 表头碎片（紧邻表题、仅几行、列数与表体不一致）单独放行，如表16表头跨页
            # （caption 挂载在合并之后进行，这里直接看前一个元素是否为 caption）
            header_frag = (len(out) >= 2 and out[-2]["kind"] == "caption"
                           and len(prev["rows"]) <= 3
                           and el["page"] - prev["page_end"] == 1)
            if adjacent and (same_cols or header_frag):
                rows = el["rows"]
                # 去掉与上一表格表头重复的前导行（多行表头取前 3 行做匹配）
                header_keys = {_norm_row(r) for r in prev["rows"][:3]}
                while rows and _norm_row(rows[0]) in header_keys:
                    rows = rows[1:]
                if rows:
                    prev["rows"].extend(rows)
                    prev["col_count"] = max(prev["col_count"], el["col_count"])
                    prev["merged"] = prev.get("merged", 1) + 1
                prev["page_end"] = el["page"]
                continue
        out.append(el)
    return out


def _attach_captions(elements: list[dict]) -> list[dict]:
    """表图题挂到表格/图片上，挂不出去的题退化为段落（正文不丢）。

    说明书里常见两种版式：
    * 表题：紧贴在表上方 → 挂给下方第一张表（维持原逻辑）；
    * 图题：整图常被拆成 A/B 两块图块、题在最上方（或题在上一页页末），
      中间还隔着 “A) DAPA-HF研究” 这类面板小标；若只认“紧邻下一个元素”，
      这类图的题会被小标打断而丢（如 图4~图7）。故对「图N」题改按版面窗口
      挂载：从题行起、到下一个题/表为止，凡同页或下一页且位于题下方的图块
      都打上该题（同一图题可覆盖 A/B 两个面板）；窗口内没挂到任何图块的题
      才退化为普通段落。
    """
    items: list[dict] = []
    pend: str | None = None        # 等待挂载的题文本
    pend_page: int | None = None   # 题行所在页
    pend_y: float | None = None    # 题行顶部 y（判同页图块是否在题下方）
    pend_fig = False               # 题是否为「图N」（启用版面窗口）
    hold: dict | None = None       # 未确认挂载的题元素：窗口关闭仍没挂上才退化为段落

    def flush_hold():
        """窗口关闭：pending 的题若始终没挂到图块/表格 → 保留为普通段落。"""
        nonlocal hold, pend, pend_page, pend_y, pend_fig
        if hold is not None:
            items.append({"kind": "paragraph", "page": hold["page"],
                          "bbox": hold["bbox"], "text": hold["text"]})
        hold = None
        pend = pend_page = pend_y = None
        pend_fig = False

    def covers(el: dict) -> bool:
        """图块是否落在题行的版面窗口内：同页且在题下方，或紧邻的下一页。"""
        if el["page"] == pend_page:
            y0 = (el.get("bbox") or [0, 0, 0, 0])[1]
            return pend_y is None or y0 >= pend_y - 1.0
        return el["page"] == pend_page + 1

    for el in elements:
        kind = el["kind"]
        text = el.get("text") or ""
        if kind == "caption":
            flush_hold()          # 新题到来 = 上一个窗口结束
            is_fig = bool(re.match(r"^图\s*\d+", text))
            if is_fig and items and items[-1]["kind"] == "figure" \
                    and "caption" not in items[-1]:
                items[-1]["caption"] = text   # 题紧跟在图块下方（单块图题在图下）
                continue
            pend, pend_fig = text, is_fig
            pend_page = el["page"]
            pend_y = (el.get("bbox") or [0, 0, 0, 0])[1]
            hold = el
        elif kind == "figure":
            if pend:
                if not pend_fig:
                    el["caption"] = pend       # 表题 → 挂给下一张表/图（原行为）
                    pend = None
                    hold = None
                elif covers(el):
                    el["caption"] = pend       # 图题 → 题下方图块（含 A/B 面板、跨页）
                    hold = None                # 已挂到 ≥1 块；窗口保持，等后续面板
            items.append(el)
        elif kind == "table":
            if pend and not pend_fig:
                el["caption"] = pend
                pend, hold = None, None
            flush_hold()          # 表格打断图题窗口
            items.append(el)
        else:
            # paragraph/subheading/heading：正文。
            # 表题必须紧邻表格，正文一到即退化；图题窗口可穿过面板小标与脚注
            #（图4~图7 的 A)/B) 标注、n/N# 注释都在两面板之间）。
            if pend and not pend_fig:
                flush_hold()
            items.append(el)
    flush_hold()
    return items


# ---------------------------------------------------------------- 文档组装

def _split_sections(items: list[dict]) -> list[dict]:
    sections = [{"title": "文档头", "page_start": 0, "items": []}]
    for el in items:
        if el["kind"] == "heading":
            sections.append({"title": el["text"], "page_start": el["page"], "items": []})
        else:
            sections[-1]["items"].append(el)
    for sec in sections:
        sec["page_end"] = max((it["page"] for it in sec["items"]), default=sec["page_start"])
    return [s for s in sections if s["items"] or s["title"] != "文档头"]


def _parse_meta(pdf_path: str, sections: list[dict], n_pages: int) -> dict:
    meta = {
        "file": os.path.basename(pdf_path),
        "pages": n_pages,
        "doc_id": "",
        "doc_type": "药品说明书",
    }
    m = re.search(r"[A-Z]{4}\d{7}(-\d+)?", os.path.basename(pdf_path))
    if m:
        meta["doc_id"] = m.group(0)

    head_text = "\n".join(it.get("text", "") for it in sections[0]["items"]) if sections else ""
    m = re.search(r"核准日期[:：]\s*(\S+)", head_text)
    if m:
        meta["核准日期"] = m.group(1)
    m = re.search(r"修改日期[:：]\s*(.+)", head_text)
    if m:
        meta["修改日期"] = m.group(1).strip()

    for sec in sections:
        if sec["title"] == "【药品名称】":
            for it in sec["items"]:
                t = it.get("text", "")
                for key in ("通用名称", "商品名称", "英文名称"):
                    if t.startswith(key):
                        meta[key] = t.split("：", 1)[-1].split(":", 1)[-1].strip()
    return meta


def _build_chunks(sections: list[dict], meta: dict, limit: int) -> list[dict]:
    chunks = []

    def add(kind, section, sub, content, page_start, page_end, caption=None):
        chunks.append({
            "id": f"c{len(chunks) + 1:03d}",
            "type": kind,
            "section": section,
            "sub_section": sub,
            "caption": caption,
            "content": content,
            "page_start": page_start + 1,
            "page_end": page_end + 1,
            "char_len": len(content),
            "meta": {
                "doc_id": meta.get("doc_id", ""),
                "doc_type": meta.get("doc_type", ""),
                "drug_name": meta.get("通用名称", ""),
                "drug_alias": meta.get("商品名称", ""),
            },
        })

    for sec in sections:
        buf, buf_start, sub = [], None, None

        def flush():
            nonlocal buf, buf_start
            if buf:
                add("text", sec["title"], sub,
                    "\n".join(t for t, _ in buf), buf_start, buf[-1][1])
                buf, buf_start = [], None

        for it in sec["items"]:
            kind = it["kind"]
            if kind == "subheading":
                flush()
                sub = it["text"]
                buf, buf_start = [(it["text"], it["page"])], it["page"]
            elif kind == "paragraph":
                if buf and sum(len(t) for t, _ in buf) + len(it["text"]) > limit:
                    flush()
                if not buf:
                    buf_start = it["page"]
                buf.append((it["text"], it["page"]))
            else:  # table / figure：原子 chunk
                flush()
                if kind == "figure":
                    # 图正文 = 图题 + OCR 图内文字 + VLM 图描述 + 图片路径（空行跳过）。
                    # 前两者由 image_describer 注入（main.py --describe），
                    # 图片路径保留在末行便于回跳原图。
                    desc = it.get("description") or {}
                    content = "\n".join(
                        x for x in (it.get("caption"), desc.get("ocr_text"),
                                    desc.get("vlm_description"), it.get("file"))
                        if x)
                else:
                    content = it.get("md") or it.get("file") or ""
                    if it.get("caption"):
                        content = f"{it['caption']}\n{content}"
                add(kind, sec["title"], sub, content,
                    it["page"], it.get("page_end", it["page"]), it.get("caption"))

        flush()
    return chunks


# ---------------------------------------------------------------- 图语义化

def _clean_desc(desc) -> dict:
    """只保留非空的 OCR / VLM 字段，去掉 error 等控制信息。"""
    cleaned = {
        k: (v or "").strip()
        for k, v in (desc or {}).items()
        if k in ("ocr_text", "vlm_description")
    }
    return {k: v for k, v in cleaned.items() if v}


def _describe_figures(pending: list, image_describer) -> None:
    """给已落盘的图补 OCR/VLM 语义。

    pending: [(item, fid, abs_path), ...]
    优先走描述器的 ``describe_batch``（一次 HTTP 处理整篇图，见 pdfagent.image_client），
    没有该接口或批量失败时回退逐张回调 ``f(image_bytes, caption)``。
    """
    if not pending or image_describer is None:
        return

    batch = getattr(image_describer, "describe_batch", None)
    if callable(batch):
        try:
            requests = [{"path": path, "caption": it.get("caption")}
                        for it, _fid, path in pending]
            results = list(batch(requests))
        except Exception as e:  # noqa: BLE001
            print(f"[warn] 批量图语义化失败，回退逐张: {e}")
            results = []
        if len(results) == len(pending):
            for (it, _fid, _path), desc in zip(pending, results):
                error = (desc or {}).get("error")
                if error:
                    it["description_error"] = str(error)
                it["description"] = _clean_desc(desc)
            return

    for it, fid, _path in pending:
        try:
            it["description"] = _clean_desc(
                image_describer(it.get("image"), it.get("caption")))
        except Exception as e:  # noqa: BLE001
            print(f"[warn] 图语义化失败 {fid}: {e}")
            it["description"] = {}
            it["description_error"] = str(e)


# ---------------------------------------------------------------- 入口

def parse_document(pdf_path: str, out_dir: str, chunk_limit: int = CHUNK_LIMIT,
                   image_describer=None) -> dict:
    """解析 PDF 为结构化结果并落盘（out/layout.json 等）。

    image_describer: 可选回调，图语义化入口（见 describe_images.py）。
        签名 f(image_bytes: bytes, caption: str | None) -> dict；
        返回 {"ocr_text": str, "vlm_description": str}，空串表示无结果。
        None 或缺通道时图片只提取+挂图题，不调外部模型（保持纯解析零依赖）。
    """
    doc = pymupdf.open(pdf_path)
    elements = []
    for pno in range(len(doc)):
        #对每页做版面解析,产出该页元素(heading/subheading/caption/paragraph/table/figure,带page/bbox/text,表格带 rows,图片带 image 字节),按页码拼成整篇一条流 
        elements.extend(_extract_elements(doc[pno], pno))
    n_pages = len(doc)
    doc.close()

    merged_elements = _merge_tables(elements)
    items = _attach_captions(merged_elements)
    sections = _split_sections(items)
    meta = _parse_meta(pdf_path, sections, n_pages)
    #print(f'\nmeta 的内容是什么：{meta}')

    #print(f'\nsections 的内容是什么:{sections}')



    # 表格 / 图片编号、落盘、转 Markdown
    fig_dir = os.path.join(out_dir, "figures")
    tab_dir = os.path.join(out_dir, "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tab_dir, exist_ok=True)

    # 表格：逐张落盘 + 转 Markdown
    tables = []
    for sec in sections:
        for it in sec["items"]:
            if it["kind"] != "table":
                continue
            tid = f"table-{len(tables) + 1:02d}"
            it["md"] = table_to_markdown(it["rows"])
            it["file"] = f"tables/{tid}.md"
            with open(os.path.join(out_dir, it["file"]), "w", encoding="utf-8") as f:
                if it.get("caption"):
                    f.write(it["caption"] + "\n\n")
                f.write(it["md"] + "\n")
            tables.append({"id": tid, "caption": it.get("caption"),
                           "section": sec["title"],
                           "page_start": it["page"] + 1, "page_end": it["page_end"] + 1,
                           "rows": len(it["rows"]), "cols": it["col_count"],
                           "merged_fragments": it.get("merged", 1), "file": it["file"]})

    # 图片：先全部落盘，再统一语义化（批量一次 HTTP，见 pdfagent.image_client）
    figure_entries = [(sec["title"], it) for sec in sections for it in sec["items"]
                      if it["kind"] == "figure"]
    pending = []
    for i, (_title, it) in enumerate(figure_entries):
        fid = f"figure-{i + 1:02d}"
        it["file"] = None
        if it.get("image"):
            it["file"] = f"figures/{fid}.{it['ext']}"
            path = os.path.join(out_dir, it["file"])
            with open(path, "wb") as f:
                f.write(it["image"])
            pending.append((it, fid, path))

    _describe_figures(pending, image_describer)

    figures = []
    for i, (title, it) in enumerate(figure_entries):
        fid = f"figure-{i + 1:02d}"
        it.pop("image", None)  # bytes 不进 JSON
        figures.append({"id": fid, "caption": it.get("caption"),
                        "description": it.get("description") or None,
                        "description_error": it.get("description_error"),
                        "section": title, "page": it["page"] + 1,
                        "file": it["file"]})

    chunks = _build_chunks(sections, meta, chunk_limit)

    return {
        "meta": meta,
        "sections": [{"title": s["title"], "page_start": s["page_start"] + 1,
                      "page_end": s["page_end"] + 1, "n_items": len(s["items"])}
                     for s in sections],
        "tables": tables,
        "figures": figures,
        "chunks": chunks,
        # 供可视化标注用（kind/page/bbox + 摘要）
        "elements": [{"kind": e["kind"], "page": e["page"], "bbox": e["bbox"],
                      "label": (e.get("text") or e.get("caption") or "")[:30]}
                     for e in merged_elements],
    }
