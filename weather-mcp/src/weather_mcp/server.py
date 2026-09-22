from mcp.server.mcpserver import MCPServer
import httpx
import requests

mcp=MCPServer("CalculatorServer")


# wttr.in 请求超时（秒）
HTTP_TIMEOUT = 10

@mcp.tool()
def add(a:int,b:int) ->int:
    """Add two numbers"""
    return a+b

@mcp.tool()
def multiply(a:int,b:int)->int:
    """Multiply two numbers"""
    return a*b


@mcp.tool()
def get_alerts(state: str) -> str:
    """获取美国某个州的天气警报。用两位州代码，比如 CA 或 NY。"""
    url = f"https://api.weather.gov/alerts/active?area={state.upper()}"
    headers = {"User-Agent": "weather-mcp-demo/1.0"}
    
    with httpx.Client(timeout=HTTP_TIMEOUT) as client:
        resp = client.get(url, headers=headers)
        resp.raise_for_status()
        data = resp.json()

    features = data.get("features", [])
    if not features:
        return f"{state} 当前没有活跃的天气警报。"

    # 简单格式化一下结果
    lines = []
    for f in features[:5]:  # 只取前5条演示
        props = f["properties"]
        lines.append(f"- {props.get('event')}: {props.get('areaDesc')}")
    return "\n".join(lines)



@mcp.tool()
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



if __name__=="__main__":
    import inspect  
    print(inspect.signature(MCPServer.run_streamable_http_async))   
    mcp.run(
        transport="streamable-http", 
        host="0.0.0.0",          # 允许外部访问
        port=8010,
        streamable_http_path="/mcp_server",   # 端点路径

    )

