"""工具注册表：把 MCPClientManager 拿到的工具列表变成可检索、可调用的索引。

对应 demo.py 里的 ToolRegistry，主要职责：
1. 用工具名（fastmcp 的 `{server}_{tool}`）建立索引；
2. 把选中的工具转成 OpenAI function-calling 的 tools 参数；
3. 把 LLM 返回的工具调用分发回对应的 MCP 服务端。

每个工具是 MCPClientManager.list_tools() 返回的 dict：
    {"name": ..., "server": ..., "tool_name": ..., "description": ..., "input_schema": ...}
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    from mcp_server.client import MCPClientManager


class ToolRegistry:
    """索引所有 MCP 工具，并提供「转 LLM 定义」与「分发调用」两个能力。

    用法::

        tools = await manager.list_tools()
        registry = ToolRegistry(tools, manager)
        llm_tools = registry.to_llm_tools(["filesystem_read_file"])
        result = await registry.dispatch("filesystem_read_file", {"path": "a.txt"})
    """

    def __init__(self, tools: Iterable[dict[str, Any]], client: "MCPClientManager"):
        self.client = client
        # key 直接用完整工具名（多服务端时已带 `{server}_` 前缀）
        self.index: dict[str, dict[str, Any]] = {tool["name"]: tool for tool in tools}

    def __len__(self) -> int:
        return len(self.index)

    def __contains__(self, name: object) -> bool:
        return name in self.index

    @property
    def keys(self) -> list[str]:
        """全部工具名，顺序与 list_tools() 一致。"""
        return list(self.index)

    def get(self, name: str) -> dict[str, Any] | None:
        return self.index.get(name)

    def server_of(self, name: str) -> str | None:
        """工具所属服务端名（交给 MCPClientManager 的前缀解析）。"""
        return self.client.server_of(name)

    def to_llm_tools(self, keys: Iterable[str] | None = None) -> list[dict[str, Any]]:
        """把工具转成 OpenAI function-calling 的 tools 参数。

        Args:
            keys: 要暴露给 LLM 的工具名；不传则暴露全部。
        """
        selected = self.keys if keys is None else list(keys)
        out: list[dict[str, Any]] = []
        for key in selected:
            tool = self.index.get(key)
            if tool is None:
                continue
            schema = tool.get("input_schema") or {"type": "object", "properties": {}}
            out.append(
                {
                    "type": "function",
                    "function": {
                        "name": key,
                        "description": tool.get("description") or "",
                        "parameters": schema,
                    },
                }
            )
        return out

    async def dispatch(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        """按完整工具名调用底层 MCP 工具。"""
        if name not in self.index:
            raise KeyError(f"未注册的工具：{name}")
        return await self.client.call_tool(name, arguments or {})
