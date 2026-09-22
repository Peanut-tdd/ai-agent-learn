"""bge-m3 嵌入服务客户端

用法:
    python client.py demo                              # 跑内置演示(相似度+词权重对比)
    python client.py "text1" "text2" ...               # 编码命令行传入的文本
    python client.py "..." --sparse                    # 顺便返回稀疏词权重
    python client.py --url http://10.0.0.8:8001 demo    # 连远程服务

当库用在别的服务里(它是普通 HTTP 客户端, 可被任意服务 import):
    from client import EmbeddingClient
    c = EmbeddingClient("http://embed-svc:8001",
                        api_key=os.environ["EMBED_API_KEY"])

    # 批量索引: 一次塞一批文本(自动分块), 服务端一次前向编码
    docs = [d.text for d in fetch_docs()]
    vecs  = c.embed(docs)["dense"]

    # 在线检索: 单个 query 也可以随手发 —— 服务端会把并发小请求自动攒批,
    # 无需调用方自己拼 batch (见 server.py 的 Batcher)
    q = c.embed(query_text)["dense"][0]

    c.wait_until_ready()        # 启动编排/探活用: 阻塞直到模型就绪
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import Iterable

import requests

DEFAULT_URL = "http://127.0.0.1:8001"


class EmbeddingClient:
    """/embed 的健壮封装: API key / 失败重试 / 等待就绪 / 按 max_batch 分块。"""

    def __init__(self, base_url: str = DEFAULT_URL, timeout: float = 120.0,
                 max_batch: int = 64, api_key: str | None = None,
                 retries: int = 2) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_batch = max_batch
        self._session = requests.Session()
        if api_key:
            self._session.headers.update({"X-API-Key": api_key})   # 服务端开了鉴权时必带
        self._retries = max(0, retries)

    # -- 传输层: 网络错误 / 5xx 自动重试(指数退避); 4xx 原样返回交给上层判断 ---
    def _request(self, method: str, path: str, *, json_body: dict | None = None,
                 timeout: float) -> requests.Response:
        last_err: requests.RequestException | None = None
        for attempt in range(self._retries + 1):
            try:
                r = self._session.request(method, f"{self.base_url}{path}",
                                          json=json_body, timeout=timeout)
            except requests.RequestException as e:      # 连接失败/超时
                last_err = e
                if attempt < self._retries:
                    time.sleep(0.25 * (2 ** attempt))
                    continue
                raise
            if 500 <= r.status_code < 600 and attempt < self._retries:
                time.sleep(0.25 * (2 ** attempt))       # 服务端抽风/过载, 退避重试
                continue
            return r
        raise RuntimeError("unreachable")  # pragma: no cover

    # -- 基础接口 ----------------------------------------------------------
    def health(self) -> dict:
        r = self._request("GET", "/health", timeout=5)
        r.raise_for_status()
        return r.json()

    def info(self) -> dict:
        r = self._request("GET", "/info", timeout=5)
        r.raise_for_status()
        return r.json()

    def wait_until_ready(self, timeout: float = 300.0, interval: float = 2.0) -> dict:
        """轮询 /health 直到模型就绪(启动编排 / 容器探活用)。超时抛异常。"""
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            try:
                h = self.health()
            except requests.RequestException:
                pass                                  # 服务还没起来, 继续等
            else:
                if h.get("status") == "ok":
                    return h
            time.sleep(interval)
        raise TimeoutError(f"{timeout:.0f}s 内服务未就绪: {self.base_url}")

    # -- 基础接口 ----------------------------------------------------------
    def health(self) -> dict:
        r = self._session.get(f"{self.base_url}/health", timeout=5)
        r.raise_for_status()
        return r.json()

    def info(self) -> dict:
        r = self._session.get(f"{self.base_url}/info", timeout=5)
        r.raise_for_status()
        return r.json()

    # -- 嵌入 --------------------------------------------------------------
    def embed(self, texts: str | Iterable[str], *, return_dense: bool = True,
              return_sparse: bool = False, return_colbert_vecs: bool = False,
              max_length: int | None = None) -> dict:
        """返回 {dense, sparse_ids, sparse_tokens, colbert, meta}。

        dense        list[list[float]]        每条文本一个向量(dim 维)
        sparse_ids   list[dict[int,float]]    token_id -> 权重 (喂 OpenSearch/ES 用)
        sparse_tokens list[dict[str,float]]   token文本 -> 权重 (给人看/调试用)
        """
        if isinstance(texts, str):
            texts = [texts]
        texts = [t for t in texts if t.strip()]  # 丢掉空串
        if not texts:
            raise ValueError("没有可编码的文本")

        merged: dict = {"dense": [], "sparse_ids": [], "sparse_tokens": [], "colbert": []}
        metas: list[dict] = []

        for i in range(0, len(texts), self.max_batch):
            chunk = texts[i:i + self.max_batch]
            payload = {
                "texts": chunk,
                "return_dense": return_dense,
                "return_sparse": return_sparse,
                "return_colbert_vecs": return_colbert_vecs,
            }
            if max_length is not None:
                payload["max_length"] = max_length
            r = self._request("POST", "/embed", json_body=payload, timeout=self.timeout)
            if r.status_code != 200:
                hint = "(若服务端开了鉴权, 检查 api_key 是否设置)" if r.status_code == 401 else ""
                raise RuntimeError(f"服务返回 {r.status_code}: {r.text[:300]} {hint}")
            data = r.json()
            for k in merged:
                if data.get(k) is not None:
                    merged[k].extend(data[k])
            metas.append(data["meta"])

        for k in list(merged):
            if not merged[k]:
                merged[k] = None
        merged["meta"] = metas[0] if len(metas) == 1 else metas
        return merged


# ---------------------------------------------------------------------------
# 工具: 相似度计算(演示用)
# ---------------------------------------------------------------------------
def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb + 1e-12)


def demo(client: EmbeddingClient) -> None:
    """演示 dense + sparse 双路输出: bge-m3 混合检索的两个通道一次都拿到。"""
    query = "什么是BGE M3？"
    docs = [
        "BGE M3是一个支持密集、稀疏和多向量检索的嵌入模型",
        "今天天气很好, 适合出去跑步",
    ]

    print(f"模型信息: {client.info()}")
    print(f"\nquery : {query}")
    for i, d in enumerate(docs):
        print(f"doc{i+1}: {d}")

    out = client.embed([query] + docs, return_sparse=True)
    q_dense, *d_dense = out["dense"]

    print("\n── dense 余弦相似度 ──")
    for i, d in enumerate(d_dense):
        print(f"  doc{i+1}: {cosine(q_dense, d):.4f}")

    print("\n── sparse 词权重(前 10 个 token) ──")
    q_tokens = out["sparse_tokens"][0]
    top = sorted(q_tokens.items(), key=lambda kv: kv[1], reverse=True)[:10]
    for tok, w in top:
        print(f"  {tok!r:<12} {w:.4f}")
    print("(token 为 0.0000 的被过滤; 完整 dict 可直接喂倒排索引做混合检索)")


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="bge-m3 嵌入服务客户端")
    ap.add_argument("texts", nargs="*", help="要编码的文本; 不传则进入 demo 模式")
    ap.add_argument("--url", default=DEFAULT_URL, help=f"服务地址(默认 {DEFAULT_URL})")
    ap.add_argument("--api-key", default=os.environ.get("EMBED_API_KEY", ""),
                    help="服务端若开了鉴权, 传 X-API-Key(默认读环境变量 EMBED_API_KEY)")
    ap.add_argument("--sparse", action="store_true", help="同时返回稀疏词权重")
    ap.add_argument("--colbert", action="store_true", help="同时返回 colbert 多向量(开销大)")
    ap.add_argument("--batch", type=int, default=64, help="每次请求的最大文本数")
    args = ap.parse_args()

    client = EmbeddingClient(base_url=args.url, max_batch=args.batch,
                             api_key=args.api_key or None)

    if not args.texts or args.texts == ["demo"]:
        demo(client)
        return
    

    out = client.embed(args.texts, return_sparse=args.sparse,
                       return_colbert_vecs=args.colbert)
    meta = out["meta"]
    print(f"[meta] {meta.get('num_texts')} 条, {meta.get('elapsed_ms')}ms, "
          f"dim={meta.get('dim')}, device={meta.get('device')}")
    if out["dense"]:
        print(f"[dense] 每条 {len(out['dense'][0])} 维, 前5维示例:")
        for i, vec in enumerate(out["dense"]):
            print(f"  #{i}: {[round(v, 4) for v in vec[:5]]}")
    if args.sparse:
        for i, toks in enumerate(out["sparse_tokens"]):
            top = sorted(toks.items(), key=lambda kv: kv[1], reverse=True)[:8]
            print(f"[sparse#{i}] " + ", ".join(f"{k}:{v:.3f}" for k, v in top))


if __name__ == "__main__":
    sys.exit(main())
