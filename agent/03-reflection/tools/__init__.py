"""工具注册中心 — 统一导出 TOOL_FUNCTIONS（ToolSpec 注册表）与 TOOLS（schema 列表）。"""

from tools.basic_calculator import Args as CalcArgs, basic_calculator
from tools.current_time import Args as CurrentTimeArgs,current_time
from tools.word_count import Args as WordCountArgs,word_count
from tools.get_weather import Args as WeatherArgs,get_weather
from tools.get_attraction import Args as AttrArgs,get_attraction
from tools.tool_result import  ToolResult, ToolSpec, make_tool_schema

TOOL_FUNCTIONS: dict[str, ToolSpec] = {
    "basic_calculator": ToolSpec(
        name="basic_calculator",
        handler=basic_calculator,
        args_model=CalcArgs,
        description="执行两个数的基本四则运算（加减乘除）。operation 可选值：add/subtract/multiply/divide。",
    ),
    "current_time":ToolSpec(
        name="current_time",
        handler=current_time,
        args_model=CurrentTimeArgs,
        description="返回指定时区的当前时间，默认 Asia/Shanghai。"
    ),
    "word_count":ToolSpec(
        name="word_count",
        handler=word_count,
        args_model=WordCountArgs,
        description="统计文本字符数与词数（按空白分隔）。"
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
   
}

TOOLS: list[dict] = [make_tool_schema(spec) for spec in TOOL_FUNCTIONS.values()]
