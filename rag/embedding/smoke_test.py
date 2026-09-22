"""冒烟测试: 自动起服务 -> 等模型加载 -> 真实调一次 embed -> 断言结果 -> 收尾。

用法:  .venv/bin/python smoke_test.py [--port 8101]
启动较慢(首次需下载/加载 ~2.3GB 模型), 耐心等; 成功后打印 PASS。
子进程的 uvicorn 日志写到 server_smoke.log, 失败时可查看原因。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import requests

PORT = 8101  # 避开日常开发用的 8001, 互不干扰
HERE = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(HERE, "server_smoke.log")


def _wait_healthy(url: str, proc: subprocess.Popen, timeout: float = 300.0) -> None:
    """轮询 /health 直到模型 ready。

    注意: 启动期间(模型加载中) uvicorn 会返回 502, 属于正常现象, 继续轮询即可。
    """
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:  # 服务进程提前退出 -> 立刻失败而非傻等
            raise RuntimeError(f"服务进程提前退出 rc={proc.returncode}, 看 {LOG_FILE} 尾部")
        try:
            r = requests.get(f"{url}/health", timeout=5)
            if r.status_code == 200 and r.json().get("status") == "ok":
                print("  [server] ready:", r.text[:160], flush=True)
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise TimeoutError(f"{timeout}s 内服务未就绪")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=PORT)
    args = ap.parse_args()

    with open(LOG_FILE, "w") as lf:
        proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "server:app",
             "--host", "127.0.0.1", "--port", str(args.port)],
            cwd=HERE, env={**os.environ, "EMBED_PORT": str(args.port)},
            stdout=lf, stderr=lf)  # 隔离子进程日志, 避免污染父进程输出管道
        url = f"http://127.0.0.1:{args.port}"
        try:
            print(f"[1/3] 启动服务并等待模型加载 (pid={proc.pid}) ...", flush=True)
            _wait_healthy(url, proc)

            print("[2/3] 调用 /embed (dense + sparse) ...", flush=True)
            r = requests.post(f"{url}/embed", timeout=300, json={
                "texts": ["什么是BGE M3？",
                          "BGE M3是一个支持密集、稀疏和多向量检索的嵌入模型"],
                "return_sparse": True,
            })
            r.raise_for_status()
            out = r.json()

            dense, sparse = out["dense"], out["sparse_tokens"]
            assert len(dense) == 2, f"dense 条数不对: {len(dense)}"
            assert len(dense[0]) == 1024, f"维度不对: {len(dense[0])}"
            assert len(sparse) == 2 and sparse[0], f"sparse 为空: {sparse}"
            assert all(abs(v) <= 1.0 for v in dense[0]), "dense 应已归一化"
            print(f"  dense: 2 条 × {len(dense[0])} 维 ✓", flush=True)
            print(f"  sparse 示例: {sorted(sparse[0].items(), key=lambda kv: -kv[1])[:5]}",
                  flush=True)

            print("[3/3] 收尾, 关闭服务 ...", flush=True)
            print("\nPASS: 服务启动 / embed 调用 / 输出结构 全部正常", flush=True)
            return 0
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    sys.exit(main())
