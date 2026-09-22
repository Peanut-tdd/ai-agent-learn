"""Agent 主流程：检索相关工具 → 交给 LLM 决策 → 执行 MCP 工具 → 生成回答。"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import threading
from types import SimpleNamespace
from typing import Any

from agent.prompt import system_prompt
from llm.llm import llmapi
from mcp_server.client import SERVER_CONFIGS, MCPClientManager
from mcp_server.tool_registry import ToolRegistry
from mcp_server.tool_retriever import ToolRetriever

_lock = threading.Lock()

MAX_TURNS = 10
MAX_LLM_RETRIES = 2
LLM_RETRY_HINT = "提示：上一次模型响应超时或出错。请根据当前对话历史继续作答，必要时重新调用工具。"
HISTORY_WINDOW_TURNS = 10      # 跨轮持久化滑动窗口（user/assistant 对数）
FALLBACK_ANSWER = "抱歉，我尝试了多种方式仍未解决你的问题，请换个问法或稍后再试。"

# 每轮按用户语句召回的工具数量与相似度阈值
TOOL_TOP_K = 8
TOOL_MIN_SCORE = 0.05


class agent:
    """带 MCP 工具检索能力的对话 Agent（进程内单例）。"""

    _agent_client = None

    def __init__(self, server_configs: dict[str, dict] | None = None) -> None:
        self.llmclient = llmapi.instance()
        self.server_configs = server_configs or SERVER_CONFIGS
        self.history_msg: list[dict] = []
        # MCP 连接 / 工具索引 / 检索器，首次 chat 时懒加载
        self._loop: asyncio.AbstractEventLoop | None = None
        self._manager: MCPClientManager | None = None
        self._registry: ToolRegistry | None = None
        self._retriever: ToolRetriever | None = None

    @classmethod
    def instance(cls) -> "agent":
        if cls._agent_client is None:
            with _lock:
                if cls._agent_client is None:
                    cls._agent_client = cls()
        return cls._agent_client

    # ---------- MCP 工具准备 ----------
    async def _ensure_tools(self) -> None:
        """连接 MCP 服务端，并建立 ToolRegistry / ToolRetriever（只做一次）。

        MCP 会话与事件循环绑定；如果换了事件循环（例如再次调用 asyncio.run），
        就自动重连，避免复用已失效的会话。
        """
        loop = asyncio.get_running_loop()
        if self._registry is not None and self._loop is loop:
            return

        if self._manager is not None:
            try:
                await self._manager.close()
            except Exception:
                pass
            self._manager = None

        manager = MCPClientManager(self.server_configs)
        await manager.connect()
        tools = await manager.list_tools()

        self._manager = manager
        self._loop = loop
        self._registry = ToolRegistry(tools, manager)
        self._retriever = ToolRetriever(self._registry)
        print(
            f"[MCP] 已加载 {len(self._registry)} 个工具"
            f"，来自 {len(self.server_configs)} 个服务端"
        )

    async def aclose(self) -> None:
        """释放 MCP 连接（进程退出前调用）。"""
        if self._manager is not None:
            await self._manager.close()
        self._manager = self._registry = self._retriever = None
        self._loop = None

    # ---------- 工具检索 ----------
    def select_tools(self, query: str) -> list[dict[str, Any]]:
        """按用户语句召回相关工具，返回 OpenAI function-calling 的 tools 参数。"""
        if self._retriever is None or self._registry is None:
            raise RuntimeError("工具尚未加载，请先 await agent.chat(...) 或 _ensure_tools()")

        picked = self._retriever.search(query, k=TOOL_TOP_K, min_score=TOOL_MIN_SCORE)
        print(
            f"[检索] {query!r} → 命中 {len(picked)}/{len(self._registry)} 个工具"
        )
        for key, score in picked:
            print(f"    {score:.3f}  {key}")
        return self._registry.to_llm_tools([key for key, _ in picked])

    # ---------- 历史与兜底 ----------
    def _remember(self, role: str, content: str):
        """持久化一条消息，并按 turn 边界整对裁剪到窗口内。"""
        self.history_msg.append({"role": role, "content": content})
        while len(self.history_msg) > HISTORY_WINDOW_TURNS * 2:
            self.history_msg.pop(0)  # 弹出最早的 user
            self.history_msg.pop(0)  # 及其对应的 assistant

    def _fallback(self, query: str, content: str = FALLBACK_ANSWER):
        """兜底：持久化本轮问答并返回最终回答对象（不抛异常）。"""
        self._remember("user", query)
        self._remember("assistant", content)
        return SimpleNamespace(content=content, tool_calls=None)

    # ---------- 工具执行 ----------
    @staticmethod
    def _tool_result_text(result: Any) -> str:
        """把 MCP 工具结果转成给 LLM 看的纯文本。"""
        data = getattr(result, "data", None)
        if data is not None:
            if isinstance(data, str):
                return data
            if dataclasses.is_dataclass(data) and not isinstance(data, type):
                data = dataclasses.asdict(data)
            try:
                return json.dumps(data, ensure_ascii=False, default=str)
            except TypeError:
                return str(data)

        content = getattr(result, "content", None) or []
        texts = [getattr(block, "text", None) for block in content]
        texts = [text for text in texts if text]
        return "\n".join(texts) if texts else str(result)

    async def _execute_tool(self, tool_call) -> str:
        """解析 LLM 给出的工具名与参数，调用 MCP 工具，失败时回传错误文本。"""
        name = tool_call.function.name
        raw_args = tool_call.function.arguments
        try:
            arguments = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError as exc:
            return f"工具 {name} 参数不是合法 JSON：{exc}；原始参数={raw_args!r}"
        if not isinstance(arguments, dict):
            return f"工具 {name} 的参数必须是 JSON 对象，实际是 {type(arguments).__name__}"

        try:
            result = await self._registry.dispatch(name, arguments)
        except Exception as exc:
            return f"工具 {name} 执行失败：{exc}"
        return self._tool_result_text(result)

    # ---------- 主流程 ----------
    async def chat(self, query: str):
        """处理一轮用户输入，返回最终回答对象（content / tool_calls）。"""
        await self._ensure_tools()
        tools = self.select_tools(query)

        messages: list[dict] = [{"role": "system", "content": system_prompt}]
        messages.extend(self.history_msg)
        messages.append({"role": "user", "content": query})

        llm_retries = 0
        for i in range(MAX_TURNS):
            print(f"\n{'=' * 10}循环{i + 1}次{'=' * 10}")
            try:
                tool_msg = self.llmclient.invoke(messages, tools or None)
            except Exception as e:
                print(f"模型调用出错：{e}")
                if llm_retries < MAX_LLM_RETRIES:
                    # 注入提示让模型继续，而不是整轮崩溃
                    llm_retries += 1
                    messages.append({"role": "user", "content": LLM_RETRY_HINT})
                    continue
                return self._fallback(query)

            if not tool_msg.tool_calls:
                # 已有最终回答就直接交付，不兜底
                self._remember("user", query)
                self._remember("assistant", tool_msg.content or "")
                return tool_msg

            # 把 assistant 的 tool_calls 按 API 要求格式加入本轮历史
            assistant_tool_calls = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in tool_msg.tool_calls
            ]
            messages.append(
                {
                    "role": "assistant",
                    "content": tool_msg.content,
                    "tool_calls": assistant_tool_calls,
                }
            )

            # 逐个执行工具，把结果以 role=tool 回传模型
            for tc in tool_msg.tool_calls:
                result = await self._execute_tool(tc)
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": result}
                )

        # 硬上限兜底：返回最终回答而不是抛异常
        return self._fallback(query)
