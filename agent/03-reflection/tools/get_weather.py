"""Tool: get_weather — 通过 wttr.in API 查询实时天气。"""

import requests
from pydantic import BaseModel, Field

# wttr.in 请求超时（秒）
HTTP_TIMEOUT = 10


class Args(BaseModel):
    city: str = Field(..., min_length=1, description="城市名，如 北京")


def get_weather(city: str) -> str:
    """
    通过调用 wttr.in API 查询真实的天气信息。
    出错时抛出异常，由上层统一捕获并标记为失败。
    """
    url = f"https://wttr.in/{city}?format=j1"

    try:
        response = requests.get(url, timeout=HTTP_TIMEOUT)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"查询 {city} 天气时网络请求失败: {e}") from e

    try:
        current_condition = data["current_condition"][0]
        weather_desc = current_condition["weatherDesc"][0]["value"]
        temp_c = current_condition["temp_C"]
    except (KeyError, IndexError) as e:
        raise ValueError(f"解析 {city} 天气数据失败，可能是城市名称无效") from e

    return f"{city}当前天气:{weather_desc}，气温{temp_c}摄氏度"
