"""
使用 fastmcp.Client 连接多种 MCP 服务端配置。

fastmcp.Client 原生支持多服务端：传入 {"mcpServers": {...}} 时会创建
MCPConfigTransport。多个服务端时把所有服务端挂载到一个组合 FastMCP 上，
工具名自动加 `{server_name}_` 前缀；只有 1 个服务端时走直连、不加前缀。

本模块只负责连接管理、工具列举与调用，并从（带前缀的）工具名反推服务端名。
"""

import asyncio
import json
from pathlib import Path
from typing import Any

from fastmcp import Client


class MCPClientManager:
    """多 MCP 服务端客户端封装。

    用法::

        manager = MCPClientManager({
            "filesystem": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "."]},
            "remote": {"transport": "http", "url": "http://127.0.0.1:8000/mcp"},
        })
        async with manager:
            tools = await manager.list_tools()
            result = await manager.call_tool("remote_add", {"a": 1, "b": 2})

    说明：多个服务端时工具名为 `{server_name}_{tool_name}`，可用
    server_of(tool.name) 反推服务端名；单服务端时按 fastmcp
    原生行为直连、不加前缀。
    """

    def __init__(self, server_configs: dict[str, dict], roots: list[str] | None = None):
        if not server_configs:
            raise ValueError("server_configs 不能为空")
        self.server_configs = server_configs
        # 回答服务端的 roots/list 请求（filesystem 等服务器会用它确定可访问目录）。
        # 必须是绝对 file:// URI，默认把当前目录作为根。
        self.roots = roots or [Path.cwd().as_uri()]
        self._client: Client = Client(
            {"mcpServers": server_configs}, roots=self.roots
        )

    def server_of(self, tool_name: str) -> str | None:
        """从工具名（`{server}_{tool}`）反推所属服务端名。

        用已知服务端名做「最长前缀匹配」，即使服务端名本身含下划线也不会拆错；
        单服务端不加前缀时返回 None。
        """
        for name in sorted(self.server_configs, key=len, reverse=True):
            if tool_name.startswith(f"{name}_"):
                return name
        return None

    # ---------- 连接管理 ----------
    async def connect(self) -> "MCPClientManager":
        """连接（并初始化）所有服务端；任一失败会抛出异常。"""
        await self._client.__aenter__()
        return self

    async def close(self) -> None:
        """断开并释放所有服务端连接。"""
        await self._client.__aexit__(None, None, None)

    async def __aenter__(self) -> "MCPClientManager":
        return await self.connect()

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    # ---------- 工具操作 ----------
    async def list_tools(self) -> list[dict[str, Any]]:
        """列出所有服务端的工具。

        Returns:
            每个工具为 dict：name / description / input_schema。
            多服务端时 name 形如 `{server_name}_{tool_name}`。
        """
        tools = await self._client.list_tools()
        return [
            {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.input_schema or {},
            }
            for tool in tools
        ]

    async def list_tools_by_server(self) -> dict[str | None, list[dict[str, Any]]]:
        """按服务端名分组返回工具：{server_name: [tool_dict, ...]}。

        单服务端不加前缀时 server_of() 返回 None，会归到 None 键下。
        """
        grouped: dict[str | None, list[dict[str, Any]]] = {
            name: [] for name in self.server_configs
        }
        for tool in await self.list_tools():
            grouped.setdefault(self.server_of(tool["name"]), []).append(tool)
        return grouped

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        """调用工具。

        Args:
            name: 工具名。多服务端时为 `{server_name}_{tool_name}`，
                  单服务端时为 fastmcp 原生的工具名。
            arguments: 工具入参。

        Returns:
            fastmcp 解析后的结果：有 output_schema 时是 dataclass/dict，
            否则是内容块列表；工具报错时抛出 ToolError。
        """
        return await self._client.call_tool(name, arguments or {})


# 多种服务端配置
SERVER_CONFIGS = {
    "filesystem": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
    },
    # "fetch": {"command": "uvx", "args": ["mcp-server-fetch"]},
    # "custom": {"command": "python", "args": ["my_mcp_server.py"]},
    "remote": {
        "transport": "http",
        "url": "http://127.0.0.1:8010/mcp_server",
    },
}


async def main():
    async with MCPClientManager(SERVER_CONFIGS) as manager:
        tools = await manager.list_tools()
        print(f"共 {len(tools)} 个工具：{[t['name'] for t in tools]}")
        print(json.dumps(tools, indent=2, ensure_ascii=False))
       

        # 调用示例：
        result = await manager.call_tool("remote_get_weather", {"city":"上海"})
        print("调用结果:", result)


if __name__ == "__main__":
    asyncio.run(main())
