#!/usr/bin/env python3
"""药品说明书版面解析入口（带 PDF 预检 + 路由分流 + 缓存 + 图片语义化）。

路由逻辑（见 pdfagent）：
    text   -> pymupdf 版式解析（规则式，快、便宜）
    hybrid -> mineru 版式解析
    scanned-> mineru 版式解析
    图片   -> curl 调用 pdfimgpipeline 的 /extract:batch 批量语义化

用法:
    python main.py <pdf路径> [-o 输出目录] [--describe] [--route auto|text|mineru]

示例:
    python main.py "../rag/_samples/达格列净片 （JXHS2200079-80）说明书.pdf" --describe  --no-cache
    python main.py "<扫描版.pdf>" --describe          # 自动路由到 mineru + 图片语义化
    python main.py "<pdf>" --route text                # 强制走 pymupdf
"""
import argparse
import json
import logging
import os
import sys

from pdfagent import Settings, build_pipeline
from visualize import render_annotated_pages, write_chunks_md, write_report


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description="药品说明书版面解析（预检 + 路由）")
    ap.add_argument("pdf", help="PDF 文件路径")
    ap.add_argument("-o", "--out", default=None, help="输出目录（默认 ./out）")
    ap.add_argument("--chunk-limit", type=int, default=600, help="文本 chunk 字符上限")
    ap.add_argument("--describe", action="store_true",
                    help="图片语义化：curl 调 pdfimgpipeline 处理图中文字/描述")
    ap.add_argument("--route", choices=["auto", "text", "mineru"], default="auto",
                    help="强制路由：auto 走预检；text 强制 pymupdf；mineru 强制 MinerU")
    ap.add_argument("--no-cache", action="store_true", help="本次跳过读写缓存")
    ap.add_argument("--fallback-text", action="store_true",
                    help="MinerU 不可用时降级到 pymupdf（质量可能下降）")
    ap.add_argument("--show-precheck", action="store_true",
                    help="打印逐页预检特征（文本字符数 / 图片覆盖占比）")
    return ap.parse_args(argv)


def _override_settings(args) -> Settings:
    settings = Settings.from_env()
    if args.no_cache:
        settings = settings.replace(cache_backend="none")
    if args.fallback_text:
        settings = settings.replace(mineru_fallback_text=True)
    return settings


def _build_describer(args, settings):
    if not args.describe:
        return None
    # 延迟 import，避免未开启图片语义化时占用配置
    from describe_images import build_describer

    return build_describer(settings)


def _print_precheck(route: dict, show_pages: bool) -> None:
    pre = route.get("precheck") or {}
    print(f"预检   : kind={pre.get('kind')} "
          f"text={pre.get('text_pages')} hybrid={pre.get('hybrid_pages')} "
          f"scanned={pre.get('scanned_pages')} blank={pre.get('blank_pages')}")
    print(f"         {pre.get('reason')}")
    if show_pages:
        for page in pre.get("page_features", []):
            print(f"         p{page['page'] + 1:>3} [{page['kind']:<7}] "
                  f"chars={page['text_chars']:<6} imgs={page['image_count']:<3} "
                  f"cover={page['image_area_ratio']:.0%}")
    if route.get("from_cache"):
        print("         （路由来自缓存）")


def _write_outputs(result: dict, pdf: str, out_dir: str):
    with open(os.path.join(out_dir, "layout.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    chunks_md = write_chunks_md(result, out_dir)

    page_files = []
    if result.get("elements"):
        page_files = render_annotated_pages(pdf, result["elements"], out_dir)
    else:
        print("[info] 该路由无 bbox 元素，跳过逐页标注图")
    report = write_report(result, out_dir, page_files)
    return chunks_md, report


def main(argv=None):
    args = _parse_args(argv)
    print(args)
    out_dir = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

    os.makedirs(out_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )

    settings = _override_settings(args)
    pipeline, cache = build_pipeline(settings)
    describer = _build_describer(args, settings)

    force = None if args.route == "auto" else args.route
    print(f"缓存   : {getattr(cache, 'name', 'injected')} "
          f"(version={settings.resolved_cache_version}, ttl={settings.cache_ttl}s)")

    try:
        result = pipeline.process(
            args.pdf, out_dir,
            chunk_limit=args.chunk_limit,
            image_describer=describer,
            force=force,
            use_cache=not args.no_cache,
        )
    except Exception as e:  # noqa: BLE001 - CLI 顶层兜底
        print(f"[error] 解析失败: {e}", file=sys.stderr)
        return 2
    finally:
        cache.close()

    route = result.get("route") or {}
    print(f"引擎   : {result.get('engine')}"
          + ("（强制）" if route.get("forced") else "")
          + ("（结果来自缓存）" if result.get("from_cache") else ""))
    _print_precheck(route, args.show_precheck)

    chunks_md, report = _write_outputs(result, args.pdf, out_dir)

    chunks = result["chunks"]
    by_type = {}
    for c in chunks:
        by_type[c["type"]] = by_type.get(c["type"], 0) + 1
    text_lens = [c["char_len"] for c in chunks if c["type"] == "text"]

    print(f"解析完成: {result['meta']['file']}")
    print(f"  章节   : {len(result['sections'])}")
    print(f"  表格   : {len(result['tables'])}")
    print(f"  图片   : {len(result['figures'])}")
    if describer is not None:
        desc_n = sum(1 for f in result["figures"] if f.get("description"))
        err_n = sum(1 for f in result["figures"] if f.get("description_error"))
        extra = f"，{err_n} 张失败（服务未启动？）" if err_n else ""
        print(f"  (图描述: {desc_n}/{len(result['figures'])} 张成功{extra})")
    print(f"  chunk  : {len(chunks)}（"
          + "，".join(f"{k} {v}" for k, v in sorted(by_type.items())) + "）")
    if text_lens:
        print(f"  文本chunk长度: 平均 {sum(text_lens) // len(text_lens)} 字，"
              f"最短 {min(text_lens)}，最长 {max(text_lens)}")
    print(f"输出目录: {out_dir}")
    print(f"  - layout.json  结构化解析结果")
    print(f"  - chunks.md    全部 chunk（{chunks_md}）")
    print(f"  - report.html  汇总报告（{report}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
