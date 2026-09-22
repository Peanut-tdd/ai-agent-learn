"""工具契约与结果：Pydantic 参数校验 + 结构化 ToolResult。"""
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Type

from pydantic import BaseModel, ValidationError


@dataclass
class ToolSpec:
    """工具契约：处理器 + Pydantic 参数模型（单一事实来源，schema 也从它生成）。"""

    name: str
    handler: Callable[..., Any]
    args_model: Type[BaseModel]
    description: str = ""


def make_tool_schema(spec: ToolSpec) -> dict:
    """从 Pydantic 参数模型生成 OpenAI function calling 的 JSON Schema。"""
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.args_model.model_json_schema(),
        },
    }


class ToolResult:
    """工具执行结果（结构化）：success + code + data/error。

    code 取值：
      - INVALID_JSON      参数不是合法 JSON
      - VALIDATION_ERROR  参数未通过 Pydantic 契约校验（模型传参错误）
      - EXECUTION_ERROR   工具执行时抛出异常（网络/外部依赖/业务异常）
      - UNKNOWN_TOOL      模型调用了未注册的工具
    """

    def __init__(
        self,
        success: bool,
        code: str = "",
        data: Any = None,
        error: Exception | None = None,
        retry_class: Any = None,
    ):
        self.success = success
        self.code = code if code else ("EXECUTION_ERROR" if not success and error else "")
        self.data = data
        self.error = error
        # 错误状态确定性分类（RetryClass），供熔断回填等使用；不设则按 error 现场分类
        self.retry_class = retry_class

    def error_message(self) -> str:
        """把异常转成给模型的单行描述（ValidationError 扁平化为字段明细）。"""
        if self.error is None:
            return ""
        if isinstance(self.error, ValidationError):
            parts = [
                f"{'.'.join(map(str, err['loc']))}: {err['msg']}"
                for err in self.error.errors()
            ]
            return "; ".join(parts) or str(self.error)
        return f"{type(self.error).__name__}: {self.error}"

    def __str__(self):
        if self.success:
            return str(self.data) if self.data is not None else ""
        return f"Error({self.code}): {self.error_message()}"

    def __repr__(self):
        return (
            f"ToolResult(success={self.success}, code={self.code!r}, "
            f"data={self.data!r}, error={self.error!r})"
        )
