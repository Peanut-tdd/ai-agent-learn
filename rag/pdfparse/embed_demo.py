#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""embed_demo.py — chunk 向量化 + 检索 demo（02-rag 输出 → embedding → 向量库 → top-k 召回）

02-rag 的版面解析器只负责「切 chunk」，输出在 out/layout.json 的 chunks 数组里。
RAG 的下一步——把每个 chunk 的正文嵌入成向量、落盘、再按语义召回——由本 demo 补齐：

    out/layout.json 的 chunks（已切好的最小检索单元）
        │ 每个 chunk 组装成一段检索文本（正文；可选拼上 章节/小节 前缀）
        ▼
    embedding 后端（两选一）
        │  A. local    本进程加载 BAAI/bge-m3（FlagEmbedding，和 ../embedding 同款模型）
        │  B. service  调常驻服务 http://127.0.0.1:8001/embed（../embedding/server.py）
        ▼
    1024 维向量 → L2 归一化 → 落盘 ./vec_store/layout.chunks.npy + .meta.json
        ▼
    query："达格列净推荐起始剂量？" → 同一后端编码 → 余弦相似度 top-k

用法（在 02-rag 目录内）:
    python3 embed_demo.py                     # demo: 建库 + 4 个示例问题检索
    python3 embed_demo.py index               # 只建库
    python3 embed_demo.py query "推荐起始剂量是多少？" "肾功能不全怎么用？" [-k 5]
    python3 embed_demo.py --backend service index   # 连 ../embedding 的常驻 bge-m3 服务
    python3 embed_demo.py -l out2/layout.json index    # 换一份解析结果建库（-s 可另指定向量库目录）

说明:
    * 默认后端 auto：8001 端口有服务就用 service，否则在本进程里加载本地模型。
    * 本地模型 BAAI/bge-m3 首次运行会下载 ~2.3GB（HF_HUB_OFFLINE=1 可用缓存离线跑）。
    * 图 chunk 若无图题（正文只有图片路径、解析器未做 VLM 描述）没有可嵌入语义，建库时跳过。
    * 纯 numpy 实现向量库，零额外依赖；生产可换 faiss / chromadb / OpenSearch（见文末）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# 常量 / 默认值
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_LAYOUT = SCRIPT_DIR / "out" / "layout.json"
DEFAULT_STORE_DIR = SCRIPT_DIR / "vec_store"
DEFAULT_SERVICE = "http://127.0.0.1:8001"   # ../embedding/server.py 的常驻服务
DEFAULT_MODEL = "BAAI/bge-m3"               # 与 ../embedding 项目同一款模型
MAX_BATCH = 64                              # 每次编码最多塞多少条文本
SERVICE_READY_TIMEOUT = 3.0                 # auto 探测服务时 /health 等待秒数
NORM_EPS = 1e-12


# ---------------------------------------------------------------------------
# embedding 后端：service（HTTP 常驻服务） / local（本进程加载模型）
# 统一接口 embed(texts: list[str]) -> np.ndarray，返回 [N, dim] float32
# ---------------------------------------------------------------------------
class ServiceEmbedder:
    """调 ../embedding/server.py 的 POST /embed。仅用标准库 urllib，无第三方依赖。"""

    def __init__(self, base_url: str = DEFAULT_SERVICE, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.dim = None

    @property
    def name(self) -> str:
        return f"service({self.base_url})"

    def is_ready(self, timeout: float = SERVICE_READY_TIMEOUT) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base_url}/health", timeout=timeout) as r:
                return r.status == 200 and r.read(64).startswith(b"{" )
        except Exception:
            return False

    def embed(self, texts: list[str]) -> np.ndarray:
        rows: list[list[float]] = []
        for i in range(0, len(texts), MAX_BATCH):
            body = json.dumps(
                {"texts": texts[i:i + MAX_BATCH],
                 "return_dense": True, "return_sparse": False,
                 "return_colbert_vecs": False},
            ).encode("utf-8")
            req = urllib.request.Request(
                f"{self.base_url}/embed", data=body,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.loads(r.read())
            rows.extend(data["dense"])
        arr = np.asarray(rows, dtype=np.float32)
        self.dim = arr.shape[1]
        return arr


class LocalEmbedder:
    """本进程加载 FlagEmbedding 的 bge 系列模型（与 ../embedding/server.py 同源）。"""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device or self._pick_device()
        self._model = None
        self.dim = None

    @staticmethod
    def _pick_device() -> str:
        try:
            import torch
            if torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    @property
    def name(self) -> str:
        return f"local({self.model_name}, device={self.device})"

    def _load(self):
        if self._model is None:
            from FlagEmbedding import BGEM3FlagModel
            print(f"[embed] 加载模型 {self.model_name}（首次自动下载 ~2.3GB，已缓存则秒起）…")
            self._model = BGEM3FlagModel(
                self.model_name, use_fp16=(self.device != "cpu"),
                devices=self.device)

    def embed(self, texts: list[str]) -> np.ndarray:
        self._load()
        rows: list[np.ndarray] = []
        for i in range(0, len(texts), MAX_BATCH):
            out = self._model.encode(
                texts[i:i + MAX_BATCH],
                return_dense=True, return_sparse=False,
                return_colbert_vecs=False,
                max_length=1024, batch_size=MAX_BATCH)
            rows.append(out["dense_vecs"])
        arr = np.concatenate(rows).astype(np.float32)
        self.dim = arr.shape[1]
        return arr


def make_embedder(backend: str, service_url: str, model_name: str,
                  device: str | None = None) -> "LocalEmbedder | ServiceEmbedder":
    """backend: local / service / auto（8001 有服务就用 service，否则本地）。"""
    if backend == "service":
        return ServiceEmbedder(service_url)
    if backend == "local":
        return LocalEmbedder(model_name, device)
    if backend == "auto":
        svc = ServiceEmbedder(service_url)
        if svc.is_ready():
            print(f"[embed] 探测到常驻服务 {service_url}，走 service 后端")
            return svc
        print(f"[embed] {service_url} 无服务，走本地模型 {model_name}")
        return LocalEmbedder(model_name, device)
    raise SystemExit(f"未知后端: {backend}（可选 local/service/auto）")


# ---------------------------------------------------------------------------
# 从 layout.json 组装「每条 chunk → 一段检索文本」，跳过无图题的图 chunk
# ---------------------------------------------------------------------------
def load_index_docs(layout_path: Path, with_context: bool) -> tuple[list[dict], list[str]]:
    """返回 (docs, texts)。

    docs: 与向量矩阵行对齐的元数据；texts: 逐行要编码的检索文本。
    返回的文本即“答案”，所以图 chunk 只有图片路径/无图题时（未做 VLM 描述）
    直接跳过——文件名没有任何可检索语义。
    """
    with open(layout_path, encoding="utf-8") as f:
        layout = json.load(f)

    docs: list[dict] = []
    texts: list[str] = []
    skipped: list[str] = []
    for c in layout["chunks"]:
        if c["type"] == "figure":
            # 图 chunk 的正文由 layout_parser 拼好：图题 / OCR 图内文字 / VLM 描述，
            # 末行是图片路径（回跳原图用），路径不进 embedding 文本。
            lines = (c["content"] or "").splitlines()
            if lines and lines[-1].startswith("figures/"):
                lines = lines[:-1]
            text = "\n".join(lines)
            if not text.strip():
                # 只剩图片路径 = 没有可嵌入语义（未跑 --describe 且无图题的图）
                skipped.append(c["id"])
                continue
        else:
            text = c["content"]
            # 纯文本 chunk 的 content 不含标题，默认把 章节/小节 前缀拼上便于脱离上下文检索；
            # 表格 chunk 的 content 开头已带表题，不再重复拼。
            if with_context and c["type"] == "text":
                head = c["section"]
                if c.get("sub_section"):
                    head += " / " + c["sub_section"]
                text = f"{head}\n{text}"
            if not text.strip():
                skipped.append(c["id"])
                continue
        docs.append({
            "id": c["id"],
            "type": c["type"],
            "section": c.get("section"),
            "sub_section": c.get("sub_section"),
            "page_start": c.get("page_start"),
            "content": c["content"],      # 展示/喂给 LLM 用原文（图含路径行）
        })
        texts.append(text)
    return docs, texts, skipped, layout.get("meta", {})


# ---------------------------------------------------------------------------
# 建库 / 检索
# ---------------------------------------------------------------------------
def build_index(layout_path: Path, store_dir: Path, backend: str,
                service_url: str, model_name: str, device: str | None,
                with_context: bool) -> Path:
    docs, texts, skipped, doc_meta = load_index_docs(layout_path, with_context)
    if not texts:
        raise SystemExit("没有可向量化的 chunk，检查 layout.json 的 chunks")

    emb = make_embedder(backend, service_url, model_name, device)
    t0 = time.time()
    print(f"[embed] 编码 {len(texts)} 条文本（batch ≤ {MAX_BATCH}）…")
    vecs = emb.embed(texts)
    elapsed = time.time() - t0

    # L2 归一化：之后内积 = 余弦相似度
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + NORM_EPS
    vecs = vecs / norms

    store_dir.mkdir(parents=True, exist_ok=True)
    name = layout_path.stem  # e.g. "layout"
    npy_path = store_dir / f"{name}.chunks.npy"
    meta_path = store_dir / f"{name}.chunks.meta.json"
    np.save(npy_path, vecs.astype(np.float32))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "layout_file": str(layout_path),
            "doc": doc_meta,                       # 药品名/受理号/日期等
            "backend": emb.name,
            "model": emb.model_name if isinstance(emb, LocalEmbedder) else None,
            "dim": int(vecs.shape[1]),
            "with_context": with_context,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_chunks": len(docs),
            "docs": docs,                          # 与矩阵行一一对应
        }, f, ensure_ascii=False, indent=2)

    print(f"[index] 完成: {len(docs)} 个向量 × {vecs.shape[1]} 维, "
          f"耗时 {elapsed:.1f}s（{elapsed / max(len(texts), 1) * 1000:.0f} ms/条）")
    if skipped:
        print(f"[index] 跳过 {len(skipped)} 个无图题图 chunk: {', '.join(skipped[:8])}"
              + (" …" if len(skipped) > 8 else "") + "（无 VLM 描述，无检索语义）")
    print(f"[index] 落盘: {npy_path}\n        {meta_path}")
    return npy_path


def search(layout_path: Path, store_dir: Path, queries: list[str], k: int,
           backend: str, service_url: str, model_name: str,
           device: str | None) -> None:
    name = layout_path.stem
    npy_path = store_dir / f"{name}.chunks.npy"
    meta_path = store_dir / f"{name}.chunks.meta.json"
    if not npy_path.exists():
        raise SystemExit(f"还没建库: {npy_path} 不存在。先跑 `python3 embed_demo.py index`")

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    vecs = np.load(npy_path)                       # 建库时已归一化，行对齐 meta["docs"]
    docs = meta["docs"]

    emb = make_embedder(backend, service_url, model_name, device)
    if emb.dim is not None and emb.dim != meta["dim"]:
        print(f"[warn] 库维度 {meta['dim']} ≠ 当前后端 {emb.dim}，向量空间不匹配，结果无意义")

    q_vecs = emb.embed(queries)
    q_norm = q_vecs / (np.linalg.norm(q_vecs, axis=1, keepdims=True) + NORM_EPS)
    scores = q_norm @ vecs.T                       # [n_queries, n_chunks] 余弦相似度

    for qi, q in enumerate(queries):
        print("\n" + "=" * 78)
        print(f"问: {q}")
        top = np.argsort(-scores[qi])[:k]
        for rank, idx in enumerate(top, 1):
            d = docs[idx]
            score = float(scores[qi, idx])
            loc = (f"{d['section']}" + (f" > {d['sub_section']}" if d.get("sub_section") else "")
                   + f" | p{d['page_start']}")
            print(f"\n  #{rank}  [{d['type']}] score={score:.4f}  {d['id']}  {loc}")
            body = d["content"].replace("\n", " ")
            print(f"      {body[:120]}{'…' if len(body) > 120 else ''}")


# ---------------------------------------------------------------------------
def demo(layout_path: Path, store_dir: Path, backend: str, service_url: str,
         model_name: str, device: str | None, k: int, with_context: bool) -> None:
    build_index(layout_path, store_dir, backend, service_url, model_name, device, with_context)
    sample = [
        "达格列净的分子式是什么？",
        "过量服用达格列净应该怎么办？",
        "eGFR为25至低于45 mL/min/1.73m2的推荐剂量是多少？",
        "这个药应该怎么贮藏？",
    ]
    print("\n" + "#" * 78)
    print("# demo 检索（答案分别在 成份 / 【药物过量】/ 表1 eGFR推荐剂量 / 【贮藏】）")
    print("#" * 78)
    search(layout_path, store_dir, sample, k, backend, service_url, model_name, device)
    print("\n说明: 纯稠密向量(cosine)召回是基线；若 top-k 里答案位置不理想（如短问句对长正文/表格），"
          "生产 RAG 一般再加两层：bge-m3 稀疏通道混合检索（../embedding 服务已支持）"
          "和 bge-reranker 交叉编码重排；向量库也可换 faiss/chromadb。")


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="02-rag chunk 向量化 + 检索 demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n"
               "  python3 embed_demo.py                     # 建库 + 示例检索\n"
               "  python3 embed_demo.py index --backend service\n"
               "  python3 embed_demo.py query \"推荐起始剂量是多少？\" -k 5\n"
               "  python3 embed_demo.py demo --device cpu   # 本地 bge-m3 强制 CPU",
    )
    ap.add_argument("cmd", nargs="?", choices=["index", "query", "demo"],
                    default="demo", help="index=建库 / query=检索 / demo=全流程(默认)")
    ap.add_argument("questions", nargs="*", help="query 子命令的问句（一个或多个）")
    ap.add_argument("-l", "--layout", default=str(DEFAULT_LAYOUT), help="layout.json 路径")
    ap.add_argument("-s", "--store", default=str(DEFAULT_STORE_DIR), help="向量库输出目录")
    ap.add_argument("-b", "--backend", default="auto",
                    help="local(本进程模型)/service(HTTP 常驻服务)/auto(默认, 自动探测)")
    ap.add_argument("--service", default=DEFAULT_SERVICE, help="service 后端地址")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="local 后端模型名")
    ap.add_argument("--device", default=None, help="local 后端设备 cpu/mps（默认自动）")
    ap.add_argument("-k", type=int, default=5, help="query/demo 每问返回条数")
    ap.add_argument("--no-context", action="store_true",
                    help="文本 chunk 不拼 章节/小节 前缀，只用正文编码")
    args = ap.parse_args()

    layout = Path(args.layout)
    store_dir = Path(args.store)
    with_context = not args.no_context

    if args.cmd == "index":
        build_index(layout, store_dir, args.backend, args.service, args.model,
                    args.device, with_context)
    elif args.cmd == "query":
        if not args.questions:
            raise SystemExit("query 需要至少一个问句，如: python3 embed_demo.py query \"推荐起始剂量是多少？\"")
        search(layout, store_dir, args.questions, args.k, args.backend,
               args.service, args.model, args.device)
    else:
        demo(layout, store_dir, args.backend, args.service, args.model,
             args.device, args.k, with_context)


if __name__ == "__main__":
    sys.exit(main())
