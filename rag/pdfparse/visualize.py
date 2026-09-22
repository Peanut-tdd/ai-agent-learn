"""可观测输出：标注页面 PNG、chunks.md、report.html。"""
from __future__ import annotations

import html
import os

import pymupdf

KIND_STYLE = {
    "heading":    ((0.0, 0.2, 0.9), "heading"),
    "subheading": ((0.85, 0.4, 0.0), "subhead"),
    "caption":    ((0.9, 0.6, 0.0), "caption"),
    "paragraph":  ((0.6, 0.6, 0.6), "para"),
    "table":      ((0.9, 0.0, 0.0), "table"),
    "figure":     ((0.0, 0.65, 0.0), "figure"),
}


def render_annotated_pages(pdf_path: str, elements: list[dict], out_dir: str) -> list[str]:
    """在每页上按元素类型画彩色框 + 标签，导出 PNG 到 out/pages/。"""
    os.makedirs(os.path.join(out_dir, "pages"), exist_ok=True)
    by_page: dict[int, list[dict]] = {}
    for el in elements:
        by_page.setdefault(el["page"], []).append(el)

    doc = pymupdf.open(pdf_path)
    files = []
    for pno in range(len(doc)):
        page = doc[pno]
        for el in by_page.get(pno, []):
            color, label = KIND_STYLE[el["kind"]]
            r = pymupdf.Rect(el["bbox"])
            page.draw_rect(r, color=color, width=1.2)
            page.insert_text((r.x0 + 2, max(9, r.y0 - 3)), label, fontsize=7, color=color)
        pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))
        rel = f"pages/page-{pno + 1:03d}.png"
        pix.save(os.path.join(out_dir, rel))
        files.append(rel)
    doc.close()
    return files


def write_chunks_md(result: dict, out_dir: str) -> str:
    chunks = result["chunks"]
    by_type = {}
    for c in chunks:
        by_type[c["type"]] = by_type.get(c["type"], 0) + 1
    lines = [
        f"# Chunk 列表 — {result['meta']['file']}",
        "",
        f"共 {len(chunks)} 个 chunk（"
        + "，".join(f"{k} {v}" for k, v in sorted(by_type.items())) + "）",
        "",
    ]
    for c in chunks:
        sub = f" > {c['sub_section']}" if c["sub_section"] else ""
        pages = f"p{c['page_start']}" if c["page_start"] == c["page_end"] \
            else f"p{c['page_start']}-{c['page_end']}"
        lines += [
            f"## [{c['id']}] {c['type']} | {c['section']}{sub} | {pages} | {c['char_len']}字",
            "",
            c["content"],
            "",
            "---",
            "",
        ]
    path = os.path.join(out_dir, "chunks.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def write_report(result: dict, out_dir: str, page_files: list[str]) -> str:
    meta = result["meta"]
    chunks = result["chunks"]
    e = html.escape

    meta_rows = "".join(
        f"<tr><th>{e(str(k))}</th><td>{e(str(v))}</td></tr>" for k, v in meta.items())

    sec_rows = "".join(
        f"<tr><td>{e(s['title'])}</td><td>p{s['page_start']}-{s['page_end']}</td>"
        f"<td>{s['n_items']}</td></tr>"
        for s in result["sections"])

    tab_blocks = "".join(
        f"<details><summary>{e(t['id'])}｜{e(t.get('caption') or '（无表题）')}｜"
        f"{e(t['section'])}｜p{t['page_start']}-{t['page_end']}｜"
        f"{t['rows']}行×{t['cols']}列"
        + (f"｜合并{t['merged_fragments']}片" if t["merged_fragments"] > 1 else "")
        + f"</summary><pre>{e(open(os.path.join(out_dir, t['file']), encoding='utf-8').read())}</pre></details>"
        for t in result["tables"])

    fig_blocks = "".join(
        f"<figure><img src='{e(f['file'])}' style='max-width:480px'>"
        f"<figcaption>{e(f['id'])}｜{e(f.get('caption') or '（无图题）')}｜p{f['page']}</figcaption></figure>"
        for f in result["figures"] if f.get("file"))

    chunk_blocks = "".join(
        f"<details><summary>[{c['id']}] {c['type']}｜{e(c['section'])}"
        + (f" &gt; {e(c['sub_section'])}" if c["sub_section"] else "")
        + f"｜p{c['page_start']}-{c['page_end']}｜{c['char_len']}字</summary>"
        + f"<pre>{e(c['content'])}</pre></details>"
        for c in chunks)

    page_imgs = "".join(
        f"<a href='{e(p)}'><img src='{e(p)}' style='width:300px;margin:4px'></a>"
        for p in page_files)

    legend = "　".join(
        f"<span style='color:rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})'>■</span>{label}"
        for c, label in KIND_STYLE.values())

    doc = f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<title>版面解析报告 — {e(meta['file'])}</title>
<style>
body {{ font-family: -apple-system, "PingFang SC", sans-serif; margin: 24px; }}
pre {{ background: #f6f6f6; padding: 12px; overflow-x: auto; font-size: 13px; }}
details {{ margin: 6px 0; border: 1px solid #ddd; border-radius: 6px; padding: 6px 10px; }}
summary {{ cursor: pointer; }}
th {{ text-align: left; padding-right: 16px; }}
h2 {{ border-bottom: 2px solid #333; padding-bottom: 4px; margin-top: 36px; }}
</style></head><body>
<h1>版面解析报告</h1>
<h2>1. 文档概览</h2>
<table>{meta_rows}</table>
<p>章节 {len(result['sections'])} 个｜表格 {len(result['tables'])} 个｜
图片 {len(result['figures'])} 张｜chunk {len(chunks)} 个</p>
<h2>2. 章节结构</h2>
<table><tr><th>章节</th><th>页码</th><th>元素数</th></tr>{sec_rows}</table>
<h2>3. 表格（{len(result['tables'])}）</h2>
{tab_blocks}
<h2>4. 图片（{len(result['figures'])}）</h2>
{fig_blocks}
<h2>5. Chunk 列表（{len(chunks)}）</h2>
{chunk_blocks}
<h2>6. 标注页面</h2>
<p>{legend}</p>
{page_imgs}
</body></html>"""

    path = os.path.join(out_dir, "report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)
    return path
