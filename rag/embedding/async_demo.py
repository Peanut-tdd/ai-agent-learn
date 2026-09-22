"""async vs sync 端点对照实验 —— 理解「事件循环不能被阻塞」

核心原理:
  事件循环是【单线程】调度器。async 任务是协作式调度: 只有在 `await` 处
  才把控制权还给事件循环。如果 async 函数里有一段【没有 await 的长活】
  (time.sleep / torch 前向 / 纯 CPU 计算), 事件循环就被这段代码占死,
  其他所有 async 任务(/health /ping)全部排队等待 —— 表现为"服务假死"。

对照的 4 种端点:
  /bad-async    async 里直接塞阻塞调用(错误示范)
  /bad-sleep    async 里用 time.sleep(同样是阻塞!sleep 不会让路)
  /good-await   async 的正确用法: await 真正的异步 IO(asyncio.sleep)
  /good-sync    阻塞工作交给 FastAPI 线程池(同步 def 端点的默认行为)

运行时自动起一个本地服务做并发测量, 打印事件循环是否被占用的证据。

用法: .venv/bin/python async_demo.py
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import threading
import time

import requests

PORT = 8300
HERE = os.path.dirname(os.path.abspath(__file__))


def heavy(seconds: float) -> float:
    """模拟阻塞型工作: time.sleep 等价于 torch 前向(不返回控制权)。

    真实 torch 前向在 C++ 里会释放 GIL, 能让【其他线程】跑;
    但【同一个事件循环上的 async 任务】只有在 await 点才能被调度,
    所以对事件循环来说它和 time.sleep 一样是"死占"。
    """
    time.sleep(seconds)
    return seconds


# ========================== FastAPI 应用 ==========================
from fastapi import FastAPI

app = FastAPI(title="async-vs-sync demo")


@app.get("/ping")
async def ping() -> str:
    return "pong"


@app.get("/bad-async")
async def bad_async() -> str:
    heavy(3)                      # ❌ 没有 await 的阻塞调用
    return "ok"


@app.get("/bad-sleep")
async def bad_sleep() -> str:
    time.sleep(3)                 # ❌ time.sleep 是阻塞, 不是让路
    return "ok"


@app.get("/good-await")
async def good_await() -> str:
    await asyncio.sleep(3)        # ✅ await 真正的异步 -> 让出控制权
    return "ok"


@app.get("/good-sync")
def good_sync() -> str:           # ✅ 同步 def: FastAPI 丢进线程池执行
    heavy(3)                      #    阻塞的只是线程池里的一个线程
    return "ok"


# ========================== 客户端测量 ==========================
def _wait_ready(url: str, proc: subprocess.Popen) -> None:
    t0 = time.time()
    while time.time() - t0 < 60:
        if proc.poll() is not None:
            raise RuntimeError(f"服务进程退出 rc={proc.returncode}")
        try:
            if requests.get(f"{url}/ping", timeout=3).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(0.5)
    raise TimeoutError("服务未就绪")


def ping_latency(url: str) -> float:
    t0 = time.perf_counter()
    requests.get(f"{url}/ping", timeout=10)
    return (time.perf_counter() - t0) * 1000


def scenario(url: str, name: str, path: str, block_s: float) -> float:
    """让 path 请求在后台跑, 立刻测 /ping 的延迟。

    ping 延迟 ≈ 阻塞时间 -> 事件循环被占死;
    ping 延迟 ≈ 0ms        -> 事件循环仍然空闲(阻塞被正确隔离/让路)。
    """
    # 后台线程先发起"长活"请求
    t_done = {}

    def _fire():
        t0 = time.perf_counter()
        requests.get(f"{url}{path}", timeout=60)
        t_done["elapsed"] = time.perf_counter() - t0

    t = threading.Thread(target=_fire, daemon=True)
    t.start()
    time.sleep(0.3)          # 确保长活请求已进入阻塞段
    ping_ms = ping_latency(url)   # 此刻去敲 /ping —— 看它等多久
    t.join(timeout=block_s + 10)
    total = t_done.get("elapsed", float("nan"))
    verdict = "❌ 事件循环被占死" if ping_ms > block_s * 500 else "✅ 事件循环畅通"
    print(f"  {name:<22} ping 延迟 {ping_ms:7.1f} ms  长活总耗时 {total:4.1f}s  {verdict}")
    return ping_ms


def main() -> None:
    with open(os.path.join(HERE, "async_demo_server.log"), "w") as lf:
        proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "async_demo:app",
             "--host", "127.0.0.1", "--port", str(PORT), "--log-level", "warning"],
            cwd=HERE, stdout=lf, stderr=lf)
    url = f"http://127.0.0.1:{PORT}"
    try:
        _wait_ready(url, proc)
        print("服务就绪. 每个场景: 后台跑一个 3 秒的『长活』, 同时敲 /ping 测延迟\n")
        print("--- 场景 A: async 端点 + 阻塞调用(错误示范) ---")
        scenario(url, "/bad-async (heavy)", "/bad-async", 3)
        scenario(url, "/bad-sleep  (time.sleep)", "/bad-sleep", 3)
        print("\n--- 场景 B: 正确的两种姿势 ---")
        scenario(url, "/good-await (await)", "/good-await", 3)
        scenario(url, "/good-sync  (线程池)", "/good-sync", 3)

        print("\n结论: A 组 ping 被拖到 ~3 秒 = 事件循环被阻塞;"
              "\n      B 组 ping 毫秒级返回 = 阻塞被 await 让路 / 线程池隔离。")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        try:
            os.remove(os.path.join(HERE, "async_demo_server.log"))
        except OSError:
            pass


if __name__ == "__main__":
    sys.exit(main())
