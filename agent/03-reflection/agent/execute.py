"""Execute 步骤 — Reflection Agent Loop 的「生成/执行」阶段（function calling 版）。

Reflection 范式：Execute(生成候选答案) -> Reflection(评审) -> 带着反馈重新 Execute ...
本文件负责前半段：把任务 + 上一轮反思反馈拼进提示词，然后跑一个 function-calling
循环 —— 模型自主决定调用工具还是直接给出答案：
  1) 返回 tool_calls  -> 逐个执行工具，把结果以 role=tool 回传，继续循环；
  2) 不再返回 tool_calls -> 这就是本轮的候选答案，返回给 AgentLoop 交给 Reflection 评审。

工具执行采用三段式：解析 JSON -> Pydantic 契约校验 -> 调用 handler，
任何一步失败都返回结构化 ToolResult，不向循环抛异常（错误会回传给模型自行修正）。
"""

import json
import os
import threading

from openai.types.chat import ChatCompletionMessageToolCall
from pydantic import ValidationError

from agent.prompt import build_execute_prompt
from llm.llm_deepseek import LlmApiClient
from tools import TOOL_FUNCTIONS
from tools.tool_result import ToolResult, ToolSpec

_lock = threading.Lock()

# 单次 execute 内允许的最大 tool-call 往返轮数（兜底，防死循环）
MAX_EXECUTE_TURNS = int(os.environ.get("max_execute_turns", "10"))


def _execute_tool(tool_call: ChatCompletionMessageToolCall) -> ToolResult:
    """三段式执行单个工具调用，返回结构化 ToolResult（永不抛异常）。"""
    tool_name = tool_call.function.name
    args_text = tool_call.function.arguments or "{}"

    spec: ToolSpec | None = TOOL_FUNCTIONS.get(tool_name)
    if spec is None:
        return ToolResult(
            success=False,
            code="UNKNOWN_TOOL",
            error=ValueError(f"未注册的工具 {tool_name}"),
        )

    # 1) 解析模型返回的 JSON 参数
    try:
        raw_args = json.loads(args_text)
    except json.JSONDecodeError as e:
        return ToolResult(success=False, code="INVALID_JSON", error=e)

    # 2) Pydantic 契约校验（模型传参错误在这里被拦下，给出字段级明细）
    try:
        validated = spec.args_model.model_validate(raw_args)
    except ValidationError as e:
        return ToolResult(success=False, code="VALIDATION_ERROR", error=e)

    # 3) 调用工具 handler
    try:
        data = spec.handler(**validated.model_dump())
        return ToolResult(success=True, data=data)
    except Exception as e:
        return ToolResult(success=False, code="EXECUTION_ERROR", error=e)


class Execute:

    _instance = None

    def __init__(self, llmclient: LlmApiClient):
        self.llmclient = llmclient

    @classmethod
    def instance(cls, llmclient: LlmApiClient):
        if cls._instance is None:
            with _lock:
                if cls._instance is None:
                    cls._instance = cls(llmclient)
        return cls._instance

    def run(self, task: str, reflections:list, session_lessons:list, history:str) -> str:
        """执行一版候选答案（function calling 循环）。

        Args:
            task:           当前用户任务
            reflections:    本轮任务此前失败的评审/修改指引（第一轮为空）
            session_lessons:历史任务沉淀的教训（参考）
            history:        会话上下文（历史+滚动摘要，供引用）
        Returns:
            str: 本轮候选答案，交由 AgentLoop 的 Reflection 步骤评审
        """
        messages: list[dict] = [
            {"role": "system",
             "content": "你是任务执行智能体，处于一段多轮会话中。"
                        "会话历史仅作参考，你只负责完成当前 Task；"
                        "需要信息时调用工具，完成后直接给出最终答案。"},
            {"role": "user",
             "content": build_execute_prompt(task, reflections, session_lessons, history)},
        ]

        for turn in range(1, MAX_EXECUTE_TURNS + 1):
            print(f"\n{'=' * 10} Execute turn {turn}/{MAX_EXECUTE_TURNS} {'=' * 10}")
            msg = self.llmclient.think(messages)

            # 没有 tool_calls -> 模型给出了最终答案，execute 阶段结束
            if not msg.tool_calls:
                answer = (msg.content or "").strip()
                print(f"Execute final answer:\n{answer}")
                return answer or "（模型未返回内容）"

            # 有 tool_calls -> 按 OpenAI 协议把 assistant 的调用原样回填
            messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            })

            # 逐个执行工具，结果以 role=tool 回传，模型看到结果后继续推理
            for tc in msg.tool_calls:
                result = _execute_tool(tc)
                print(f"  Tool[{tc.function.name}] -> {result}")
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": str(result)}
                )

        # 轮数上限兜底：返回可评审的失败说明，而不是抛异常
        return "Failed: max execute turns exceeded."
