"""工具注册中心 — 统一导出 TOOL_FUNCTIONS（ToolSpec 注册表）与 TOOLS（schema 列表）。"""

from tools.basic_calculator import Args as CalcArgs, basic_calculator
from tools.get_attraction import Args as AttrArgs, get_attraction
from tools.get_weather import Args as WeatherArgs, get_weather
from tools.simulate_payment import (
    ChargeArgs,
    CreateOrderArgs,
    QueryArgs,
    charge_payment,
    create_order,
    query_payment,
)

from tools.tool_result import SideEffect, ToolResult, ToolSpec, make_tool_schema

TOOL_FUNCTIONS: dict[str, ToolSpec] = {
    "basic_calculator": ToolSpec(
        name="basic_calculator",
        handler=basic_calculator,
        args_model=CalcArgs,
        description="执行两个数的基本四则运算（加减乘除）。operation 可选值：add/subtract/multiply/divide。",
    ),
    "get_weather": ToolSpec(
        name="get_weather",
        handler=get_weather,
        args_model=WeatherArgs,
        description="通过调用 wttr.in API 查询指定城市的实时天气。",
    ),
    "get_attraction": ToolSpec(
        name="get_attraction",
        handler=get_attraction,
        args_model=AttrArgs,
        description="根据城市和天气，使用 Tavily Search API 搜索并返回优化后的景点推荐。",
    ),
    "create_order": ToolSpec(
        name="create_order",
        handler=create_order,
        args_model=CreateOrderArgs,
        description="创建订单（幂等写）。order_id 是幂等键，调用方生成；失败时请保持 order_id 不变。",
        side_effect=SideEffect.IDEMPOTENT,
    ),
    "charge_payment": ToolSpec(
        name="charge_payment",
        handler=charge_payment,
        args_model=ChargeArgs,
        description="对用户扣款（非幂等写！）。失败后操作可能已执行但结果未知，禁止重复调用，请用 query_payment 确认。",
        side_effect=SideEffect.NON_IDEMPOTENT,
    ),
    "query_payment": ToolSpec(
        name="query_payment",
        handler=query_payment,
        args_model=QueryArgs,
        description="查询用户的扣款记录，用于确认扣款是否真的发生。",
    ),
}

TOOLS: list[dict] = [make_tool_schema(spec) for spec in TOOL_FUNCTIONS.values()]
