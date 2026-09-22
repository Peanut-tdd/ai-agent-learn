"""Tool: basic_calculator — 支持加减乘除的简单计算器。"""

from typing import Literal

from pydantic import BaseModel, Field

Operation = Literal["add", "subtract", "multiply", "divide"]


class Args(BaseModel):
    """参数契约：非法 operation 在校验层即被拒绝（VALIDATION_ERROR）。"""

    operation: Operation = Field(..., description="运算类型：add/subtract/multiply/divide")
    a: float = Field(..., description="第一个操作数")
    b: float = Field(..., description="第二个操作数")


def basic_calculator(operation: str, a: float, b: float) -> str:
    """执行四则运算，返回带算式的结果字符串。运行时错误（除零）抛 ValueError。"""
    if operation == "add":
        result = a + b
        expr = f"{a} + {b} = {result}"
    elif operation == "subtract":
        result = a - b
        expr = f"{a} - {b} = {result}"
    elif operation == "multiply":
        result = a * b
        expr = f"{a} × {b} = {result}"
    else:  # divide
        if b == 0:
            raise ValueError("division by zero")
        result = a / b
        expr = f"{a} ÷ {b} = {result}"

    return expr
