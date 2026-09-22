"""攒批器 (server.Batcher) 离线单测 —— 不加载模型, 用桩 encode 验证调度逻辑。

这是「服务端自动攒批」这一核心机制的行为契约, 模型无关:
  - 单个小请求: 只等 grace 毫秒级就发车, 不拖延迟;
  - 并发同选项小请求: 合并成一次 encode, 结果按请求切回且顺序正确;
  - 不同选项(dense vs dense+sparse): 各自成批, 互不污染;
  - 单请求自带大量文本(≥target): 立即发车, 零额外延迟;
  - encode 抛异常: 原样传给每个等待中的请求;
  - target=0: 关闭攒批, 来一批算一批(回到串行语义, 实验用)。

用法:  .venv/bin/python batcher_test.py    (不联网、不加载模型, 秒级跑完)
"""
from __future__ import annotations

import asyncio
import sys

from server import Batcher


class StubEncoder:
    """桩: 记录每次 encode 的入参, 返回与 store.encode 同构的输出。

    第 i 条文本的 dense 向量 = [i, i], 位置相关 —— 用于校验结果切回的对齐。
    """

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def encode(self, texts, return_dense=True, return_sparse=False,
               return_colbert_vecs=False, max_length=None) -> dict:
        self.calls.append((list(texts), return_dense, return_sparse,
                           return_colbert_vecs, max_length))
        n = len(texts)
        return {
            "dense": [[float(i), float(i)] for i in range(n)] if return_dense else None,
            "sparse_ids": None,
            "sparse_tokens": None,
            "colbert": None,
        }


class _FailEncoder:
    def encode(self, texts, return_dense=True, return_sparse=False,
               return_colbert_vecs=False, max_length=None) -> dict:
        raise RuntimeError("boom: 模拟 OOM")


RESULTS: list[tuple[str, bool, str]] = []   # (名字, 通过?, 说明)


def check(name: str, ok: bool, why: str = "") -> None:
    RESULTS.append((name, ok, why))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {why}" if why and not ok else ""))


async def run() -> None:
    # -- 1. 单个小请求: 只等 grace, 一次 encode, 延迟毫秒级 --------------------
    stub = StubEncoder()
    b = Batcher(stub.encode, target=32, grace_ms=4)
    b.start()
    t0 = asyncio.get_running_loop().time()
    out = await b.submit(["solo"], dense=True, sparse=False, colbert=False, max_length=None)
    elapsed_ms = (asyncio.get_running_loop().time() - t0) * 1000
    check("单请求: 1 次 encode", len(stub.calls) == 1)
    check("单请求: 结果正确", out["dense"] == [[0.0, 0.0]])
    check("单请求: 额外延迟≈grace", elapsed_ms < 200, f"实耗 {elapsed_ms:.0f}ms")
    await b.stop()

    # -- 2. 两个并发小请求(同选项): 合并成一次 encode, 结果按请求切回 ----------
    stub = StubEncoder()
    b = Batcher(stub.encode, target=32, grace_ms=1000)
    b.start()

    async def ask(text: str):
        return await b.submit([text], dense=True, sparse=False, colbert=False, max_length=None)

    t1 = asyncio.create_task(ask("A"))
    await asyncio.sleep(0)                 # 让 A 先入队、worker 进入等"同路人"状态
    t2 = asyncio.create_task(ask("B"))
    oa, ob = await asyncio.gather(t1, t2)
    check("并发合并: 只发生 1 次 encode", len(stub.calls) == 1,
          f"实为 {len(stub.calls)} 次: {stub.calls}")
    check("并发合并: 一次编了 2 条", stub.calls and stub.calls[0][0] == ["A", "B"])
    check("并发合并: A 拿到自己的切片", oa["dense"] == [[0.0, 0.0]])
    check("并发合并: B 拿到自己的切片", ob["dense"] == [[1.0, 1.0]])
    await b.stop()

    # -- 3. 不同选项: 各自成批, 不互相合并/污染 ---------------------------------
    stub = StubEncoder()
    b = Batcher(stub.encode, target=32, grace_ms=1000)
    b.start()

    async def ask2(text: str, sparse: bool):
        return await b.submit([text], dense=True, sparse=sparse, colbert=False, max_length=None)

    r = await asyncio.gather(ask2("C", False), ask2("D", True))
    sparse_calls = [c for c in stub.calls if c[2] is True]
    dense_calls = [c for c in stub.calls if c[2] is False]
    check("分桶: 触发 2 次 encode", len(stub.calls) == 2, f"实为 {len(stub.calls)}")
    check("分桶: sparse 请求独立成批", len(sparse_calls) == 1 and sparse_calls[0][0] == ["D"])
    check("分桶: dense 请求独立成批", len(dense_calls) == 1 and dense_calls[0][0] == ["C"])
    check("分桶: 各自结果正确", r[0]["dense"] == [[0.0, 0.0]] and r[1]["dense"] == [[0.0, 0.0]])
    await b.stop()

    # -- 4. 单请求自带大量文本(≥target): 立即发车, 零额外延迟 -------------------
    stub = StubEncoder()
    b = Batcher(stub.encode, target=4, grace_ms=10_000)   # target 很小, grace 巨大
    b.start()
    t0 = asyncio.get_running_loop().time()
    out = await b.submit([f"d{i}" for i in range(100)], dense=True,
                         sparse=False, colbert=False, max_length=None)
    elapsed_ms = (asyncio.get_running_loop().time() - t0) * 1000
    check("大批量: 1 次 encode 编 100 条", len(stub.calls) == 1 and len(stub.calls[0][0]) == 100)
    check("大批量: 不等 grace, 立刻发车", elapsed_ms < 200, f"实耗 {elapsed_ms:.0f}ms")
    check("大批量: 100 条向量全返回", len(out["dense"]) == 100)
    await b.stop()

    # -- 5. encode 异常: 传给每个等待的请求 -------------------------------------
    fail = _FailEncoder()
    b = Batcher(fail.encode, target=32, grace_ms=100)
    b.start()

    async def ask3(text: str):
        return await b.submit([text], dense=True, sparse=False, colbert=False, max_length=None)

    t1, t2 = asyncio.create_task(ask3("E")), asyncio.create_task(ask3("F"))
    await asyncio.sleep(0)
    errs = await asyncio.gather(t1, t2, return_exceptions=True)
    check("异常传播: 两个请求都收到异常",
          all(isinstance(e, RuntimeError) and "boom" in str(e) for e in errs))
    await b.stop()

    # -- 6. target=0: 关闭攒批, 来一批算一批 -------------------------------------
    stub = StubEncoder()
    b = Batcher(stub.encode, target=0, grace_ms=1000)
    b.start()

    async def ask4(text: str):
        return await b.submit([text], dense=True, sparse=False, colbert=False, max_length=None)

    await asyncio.gather(ask4("G"), ask4("H"))
    check("关闭攒批: 各自 encode", len(stub.calls) == 2)
    await b.stop()


def main() -> int:
    asyncio.run(run())
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\n{passed}/{len(RESULTS)} 项通过")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
