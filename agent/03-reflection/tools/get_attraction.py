"""Tool: get_attraction — 通过 Tavily Search API 搜索景点推荐。"""

import os

from pydantic import BaseModel, Field
from tavily import TavilyClient

# Tavily 请求超时（秒）
HTTP_TIMEOUT = 10


class Args(BaseModel):
    city: str = Field(..., min_length=1, description="城市名，如 北京")
    weather: str = Field(..., min_length=1, description="该城市当前的天气信息（来自 get_weather 的结果）")


def get_attraction(city: str, weather: str) -> str:
    """
    根据城市和天气，使用Tavily Search API搜索并返回优化后的景点推荐。
    配置/API 错误时抛出异常，由上层统一捕获并标记为失败。
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("未配置TAVILY_API_KEY环境变量")

    tavily = TavilyClient(api_key=api_key, timeout=HTTP_TIMEOUT)

    query = f"'{city}' 在'{weather}'天气下最值得去的旅游景点推荐及理由"

    try:
        response = tavily.search(query=query, search_depth="basic", include_answer=True)
    except Exception as e:
        raise RuntimeError(f"执行Tavily搜索时出现问题: {e}") from e

    # response['answer'] 是一个基于所有搜索结果的总结性回答
    if response.get("answer"):
        return response["answer"]

    formatted_results = []
    for result in response.get("results", []):
        formatted_results.append(f"- {result['title']}: {result['content']}")

    if not formatted_results:
        return "抱歉，没有找到相关的旅游景点推荐。"

    return "根据搜索，为您找到以下信息:\n" + "\n".join(formatted_results)
