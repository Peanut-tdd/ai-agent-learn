# 04 · MCP Calling Agent

一个「按需检索 MCP 工具」的对话 Agent 学习项目。

它把多个 MCP（Model Context Protocol）服务端的工具汇总成一个大工具池，
**每轮对话只把与用户问题最相关的少数工具交给 LLM**，从而在工具数量增长时
控制 token 开销、提升工具选择准确率，再通过标准的 function-calling 循环
自动调用 MCP 工具并给出最终回答。

核心链路：**MCP 工具池 → 工具注册表 → 向量检索召回 → 裁剪后喂给 LLM → 执行工具 → 回答**。

---

## 1. 特性

- **多 MCP 服务端统一接入**：stdio（`npx ...`）和远程 HTTP 服务端混用；多服务端时工具名自动带 `{server}_` 前缀。
- **工具注册表** `ToolRegistry`：把 MCP 工具转成 OpenAI function-calling 定义，并按名字分发调用。
- **工具向量检索** `ToolRetriever`：TF-IDF 混合索引（英文词 + 中文字符 n-gram），按自然语言问题召回相关工具。
- **零重依赖**：检索器纯 Python 实现，不需要 `scikit-learn` / `numpy`，可离线运行。
- **标准 Agent 循环**：`tool_calls` → 执行 MCP 工具 → `role=tool` 回传 → 继续，直到模型给出最终回答或到达轮数上限。

---



## 2. 目录结构

```
04-mcp-calling-agent/
├── main.py                    # 交互式入口：读入问题并调用 agent.chat
├── requirement.txt            # openai / python-dotenv / requests / fastmcp
├── .env                       # api_key、base_url（不入库）
├── agent/
│   ├── agent.py               # agent：chat 主循环、工具检索接入、工具执行
│   └── prompt.py              # system_prompt
├── llm/
│   └── llm.py                 # llmapi：OpenAI 兼容客户端单例（invoke）
└── mcp_server/
    ├── client.py              # MCPClientManager：连接/列举/调用 MCP
    ├── tool_registry.py       # ToolRegistry：工具索引 + LLM 定义 + 分发
    ├── tool_retriever.py      # ToolRetriever：TF-IDF 向量检索
    └── demo.py                # 检索方案的设计草稿（片段，不直接运行）
```

> `mcp_server/demo.py` 是一段设计参考稿（里面的 `servers`、`TfidfVectorizer`
> 等并未定义），用来演示「工具注册 + 向量检索」的思路，实际实现以
> `tool_registry.py` / `tool_retriever.py` 为准。

---



## 3. 快速开始



### 3.1 准备环境

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirement.txt          # fastmcp==4.0.5 / openai==2.9.0 等
```

`.env`（参考）：

```dotenv
api_key=sk-xxxxxxxx
base_url=https://api.deepseek.com/v1
# MODEL=deepseek-v4-pro        # 可选，llm.py 默认值
```



### 3.2 启动 MCP 服务端

`mcp_server/client.py` 里的默认配置包含两个服务端：

```python
SERVER_CONFIGS = {
    "filesystem": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "."]},
    "remote":     {"transport": "http", "url": "http://127.0.0.1:8010/mcp_server"},
}
```

- `filesystem`：本地文件系统服务端，`npx` 会按需下载，无需额外启动；
- `remote`：**需要你自己先在本机 8010 端口启动**一个 HTTP MCP 服务端（示例里是天气/计算器服务）。没启动的话，连接会失败——可先在 `SERVER_CONFIGS` 里把它注释掉。



### 3.3 运行

```bash
# 交互式对话
python main.py

# 只验证 MCP 连接与工具列举
python mcp_server/client.py
```

`main.py` 里输入问题即可，输入 `exit` 或 `退出` 结束。

---



## 4. 整体架构

```
┌──────────────┐   query   ┌──────────────────────────────────────────────┐
│   用户输入    │ ────────► │ agent.chat(query)                            │
└──────────────┘           │                                              │
                           │ 1) _ensure_tools()  ← 首次懒加载              │
                           │      MCPClientManager.connect()              │
                           │      list_tools() → ToolRegistry             │
                           │                  → ToolRetriever             │
                           │                                              │
                           │ 2) select_tools(query)                       │
                           │      ToolRetriever.search(query, k=8)        │
                           │      → [工具名...] → ToolRegistry.to_llm_tools│
                           │                                              │
                           │ 3) 循环：llmapi.invoke(messages, tools)       │
                           │      ├─ 无 tool_calls → 返回最终回答          │
                           │      └─ 有 tool_calls → _execute_tool()       │
                           │             ToolRegistry.dispatch()           │
                           │             → MCPClientManager.call_tool()    │
                           │             → role=tool 回传，继续循环         │
                           └──────────────────────────────────────────────┘
```

一次 `chat` 的关键点：

- **工具只加载一次**：`_ensure_tools()` 懒加载并缓存连接、注册表、检索器；后续轮次直接复用。
- **检索每个用户轮次做一次**：同一个用户问题在整个 function-calling 循环里使用同一批候选工具（候选集稳定，模型不会中途“发现”没被检索到的工具）。
- **模型能调用的工具 = 检索出来的工具**：没检索到的工具根本不在 `tools` 参数里，模型无法调用。

---



## 5. MCP 接入层：`MCPClientManager`

文件：`mcp_server/client.py`

对 `fastmcp.Client` 的薄封装，负责连接生命周期和工具 I/O。


| 方法                                     | 作用                                                  |
| -------------------------------------- | --------------------------------------------------- |
| `connect()` / `close()` / `async with` | 建立、释放所有服务端连接（任一服务端失败即抛异常）                           |
| `list_tools()`                         | 返回 `list[dict]`：`{name, description, input_schema}` |
| `list_tools_by_server()`               | 按服务端名分组：`{server_name: [tool_dict, ...]}`           |
| `call_tool(name, args)`                | 调用工具，返回 fastmcp 解析后的结果                              |
| `server_of(name)`                      | 从工具名反推所属服务端                                         |




### 5.1 多服务端与工具命名

`fastmcp.Client` 原生支持 `{"mcpServers": {...}}`：

- **≥ 2 个服务端**：挂载成一个组合服务器，工具名自动加前缀 → `filesystem_read_file`、`remote_get_weather`；
- **只有 1 个服务端**：直连，**不加前缀** → `read_file`。

`server_of()` 用「按服务端名长度降序做最长前缀匹配」来反推归属，这样即使服务端名本身含下划线（如 `my_api`）也不会拆错。

### 5.2 roots

构造函数默认把当前工作目录作为 `roots` 上报（绝对 `file://` URI）。filesystem 类服务端会用它来决定可访问目录。

---



## 6. 工具注册层：`ToolRegistry`

文件：`mcp_server/tool_registry.py`

把「一堆工具」变成「可按名检索、按名调用」的索引。

```python
registry = ToolRegistry(await manager.list_tools(), manager)
registry.keys                               # ['filesystem_read_file', ...]
registry.to_llm_tools(["filesystem_read_file"])   # OpenAI tools 参数
await registry.dispatch("filesystem_read_file", {"path": "requirement.txt"})
```


| 成员                        | 作用                                                           |
| ------------------------- | ------------------------------------------------------------ |
| `index`                   | `{工具名: 工具 dict}`，工具名即 fastmcp 的完整名（多服务端带前缀）                  |
| `keys`                    | 全部工具名                                                        |
| `get(name)`               | 取工具原始定义                                                      |
| `server_of(name)`         | 工具所属服务端（委托给 `MCPClientManager`）                              |
| `to_llm_tools(keys=None)` | 转成 `{"type": "function", "function": {...}}`；不传 `keys` 则导出全部 |
| `dispatch(name, args)`    | 调用 MCP 工具（会校验工具已注册）                                          |


`to_llm_tools` 的输出示例：

```json
{
  "type": "function",
  "function": {
    "name": "filesystem_read_file",
    "description": "Read the complete contents of a file ...",
    "parameters": { "type": "object", "properties": { "path": {"type": "string"} }, "required": ["path"] }
  }
}
```

`parameters` 直接来自 MCP 工具的 `input_schema`（JSON Schema）。

---



## 7. 工具检索层：`ToolRetriever`（重点）

文件：`mcp_server/tool_retriever.py`

### 7.1 为什么要检索

把全部工具都塞给 LLM 有两个问题：

1. **贵**：每个工具的名字 + 描述 + 参数 Schema 都要占 token，且每轮都发；
2. **不准**：候选太多时模型容易选错或漏选。

当工具池只有十几个时无所谓，几十上百个时就需要「先召回、再精选」。
`ToolRetriever` 就是召回器：把用户问题当成查询，从工具池里挑出最相关的 top-k。

### 7.2 匹配是怎么实现的（总览）

匹配分三步：**建文档 → 建索引 → 查询打分**。

```
工具池
  │  ① 为每个工具拼一份加权“文档” tokens
  ▼
文档 tokens ──② TF-IDF 向量化──► 归一化稀疏向量矩阵
                                      │
用户问题 ──③ 分词+改写──► 查询向量 ──► 余弦相似度 ──► 过滤 min_score ──► top-k 工具名
```



### 7.3 第 ① 步：构建工具文档（分字段加权）

每个工具的“文档”是一串 token，按字段重复不同次数来加权：


| 字段               | 权重     | 说明                                 |
| ---------------- | ------ | ---------------------------------- |
| 工具名（`_` 换成空格后分词） | **×3** | 名字最能表意，如 `read_file` → `read file` |
| 服务端名             | ×2     | 让同类服务端的工具更容易被同类问题命中                |
| 工具描述             | ×1     | 主要语义来源                             |
| 参数名              | ×1     | 提供线索，如 `path`、`city`、`encoding`    |


例如 `filesystem_read_file` 的文档大致是：

```
read file read file read file      # 名字 ×3
filesystem filesystem              # server ×2
read the complete contents of a file from the file system ...   # 描述
path tail head                     # 参数名
```

> 直觉：名字里出现查询词的工具，权重被放大 3 倍，因此在同分情况下更靠前。



### 7.4 第 ① 步的分词

```python
_WORD_RE   = re.compile(r"[a-z0-9]+")              # 英文/数字词
_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")       # 连续中文段
```

- 英文：转小写后按连续字母/数字切词；
- 中文：对每个连续中文段补充 **2~4 字符 n-gram**（如 `读取文件` → `读取`、`取文`、`文件`、`读取文`、`取文件`、`读取文件`）。

这样**不需要 jieba 之类的分词器**，也能让中文和中文描述产生部分重叠。
（注意：中文 n-gram 只有在工具文档本身含中文时才会贡献分数；当前工具描述是英文，所以中文查询主要靠下面的查询改写来桥接。）

### 7.5 第 ② 步：TF-IDF 向量化

对每份文档 tokens 统计词频，并乘以 IDF：

```
tf(t, d)  = 1 + ln( count(t, d) )
idf(t)    = ln( (1 + N) / (1 + df(t)) ) + 1        # 平滑，N = 工具数，df = 含 t 的工具数
w(t, d)   = tf(t, d) * idf(t)
```

然后把每个文档向量做 **L2 归一化**，得到单位向量，存成稀疏字典 `{term: weight}`。

- 出现在**越少工具**里的词，IDF 越大 → 区分度越高（比如 `weather` 比 `file` 更有辨识度）；
- 用 `1 + ln(count)` 而不是原始词频，避免某个词重复太多而压倒其它特征。



### 7.6 第 ③ 步：查询匹配

查询走同一条分词链路，并额外做一次**查询改写**（见 7.7），然后：

1. 计算查询的 TF-IDF 向量（未出现在工具语料里的词权重按 0 处理，天然被忽略），并 L2 归一化；
2. 与每个工具向量求**余弦相似度**（两个向量都已归一化，因此点积即余弦）：

```
score(query, tool) = Σ_t  q(t) * d_tool(t)
```

1. 过滤 `score < min_score`（默认 **0.05**）；
2. 按分数降序排序，取前 **k** 个（默认 **k = 8**）。

```python
picked = retriever.search("帮我看看目录里有哪些文件", k=8, min_score=0.05)
# [('filesystem_list_directory', 0.475), ('filesystem_write_file', 0.331), ...]
```

复杂度：查询一次是 O(工具数 × 查询特征数)，几十~几千个工具都很快；索引只在初始化时构建一次。

### 7.7 查询改写 `QUERY_EXPAND`

工具名/描述是英文（`read_file`、`weather`），用户往往说中文（“读取文件”“天气”）。
中文 n-gram 和英文 token 不会重合，所以这里加了一层**关键词表**：

```python
"文件": "file files content",
"读":   "read view open cat",
"目录": "directory folder list",
"天气": "weather temperature forecast",
"删除": "delete remove",
...
```

`_expand()` 用小写子串匹配，命中就把对应英文词追加到查询串末尾：

```
"读取 requirement.txt 的内容"
  → 命中 “读” “内容”
  → "读取 requirement.txt 的内容 read view open cat content file"
```

追加的英文词再参与 TF-IDF 打分，从而把中文问题“翻译”到英文工具空间。
这张表就是当前方案里**唯一的跨语言桥梁**，也是后续最值得替换/增强的部分（见 7.9）。

### 7.8 一个直观例子

```
query: "读取 requirement.txt 的内容"
改写后 tokens: [读取, 取..., requirement, txt, 内容, read, view, open, cat, content, file]

候选（真实运行结果）:
  0.308  fs_a_read_file
  0.295  fs_b_read_file
  0.247  fs_a_read_media_file
  0.243  fs_b_read_media_file
  0.180  fs_a_read_multiple_files
  0.172  fs_a_read_text_file
  ...
```

`read_file` 胜出，因为它的名字 token `read`（×3）和扩展词 `read` 重合，且 `file` 在名字与描述中都出现。
两个服务端各有一套同名工具，所以 `fs_a_*` 与 `fs_b_*` 分数接近（实际业务里通常靠提示词或后处理约定选哪个服务端）。

无关问题会被阈值挡掉：

```
query: "你好呀"
  → 没有任何工具词命中，分数全为 0 → search() 返回 []
  → agent 把 tools=None 传给 LLM，模型直接闲聊
```



### 7.9 可替换/增强点

- **换 embedding**：把 `_vector()` 换成向量模型（对工具描述建库 + 对 query 编码），即可获得真正的语义与跨语言匹配；当前 DeepSeek 未提供 embeddings 接口，所以先用 TF-IDF。
- **扩充** `QUERY_EXPAND`：新增 MCP 服务端后，把该领域的常见中文说法补进表里。
- **调参**：`k` 与 `min_score` 在 `agent/agent.py` 顶部（`TOOL_TOP_K = 8`、`TOOL_MIN_SCORE = 0.05`）。
- **加业务规则**：如“危险工具（写/删/发布）默认不召回或需二次确认”，可仿照 demo 的 `RISKY_VERBS` 在 `search()` 后做过滤。

---



## 8. Agent 主流程：`agent/agent.py`

```python
class agent:
    async def chat(self, query):
        await self._ensure_tools()                 # 懒加载：连接 MCP + 建 registry/retriever
        tools = self.select_tools(query)            # 检索 top-k → LLM 工具定义

        messages = [system] + self.history_msg + [user: query]
        llm_retries = 0
        for i in range(MAX_TURNS):                  # MAX_TURNS = 10
            tool_msg = self.llmclient.invoke(messages, tools or None)
            if not tool_msg.tool_calls:
                self._remember("user", query)
                self._remember("assistant", tool_msg.content or "")
                return tool_msg                      # 最终回答
            messages.append(assistant tool_calls)    # 回填 assistant 的 tool_calls
            for tc in tool_msg.tool_calls:
                result = await self._execute_tool(tc)  # 调 MCP 工具
                messages.append({role: "tool", tool_call_id: tc.id, content: result})
        return self._fallback(query)                 # 硬上限兜底，不抛异常
```

要点：

- `_ensure_tools()` **只连一次**：连接与事件循环绑定；如果换了事件循环（例如再次 `asyncio.run`），会自动重连，避免复用失效会话。
- `_execute_tool()`：解析 `tc.function.arguments`（JSON），调用 `ToolRegistry.dispatch()`，把结果转成文本；参数非法或工具报错都返回错误文本交给模型，而不是中断整轮。
- **工具结果转文本**：优先用 fastmcp 解析出的 `.data`（dataclass 会自动转 dict 再 JSON 序列化），否则拼接 `content` 里的文本块。
- **历史管理**：`history_msg` 按 user/assistant 成对保存，滑动窗口 `HISTORY_WINDOW_TURNS = 10`，下一轮拼进 `messages`。
- **可测试性**：`agent(server_configs=...)` 可注入自定义服务端配置，便于用假 LLM 做离线测试。

---



## 9. LLM 层：`llm/llm.py`

`llmapi` 是 OpenAI 兼容客户端的**进程内单例**：

- 从 `.env` 读取 `api_key` / `base_url`，`MODEL` 默认 `deepseek-v4-pro`；
- `invoke(messages, tools=None)` 调用 `chat.completions.create`，打印请求/响应，返回 `assistant` message（含 `content` 与 `tool_calls`）；
- `tools` 为空时不传该参数，避免部分服务端对空数组报错。

---



## 10. 已知限制与注意事项

1. **单服务端不加前缀**：这是 fastmcp 的原生行为（多服务端才 mount 加前缀）。代码按此语义实现，`server_of()` 在单服务端且无前缀时返回 `None`。
2. **跨语言靠关键词表**：TF-IDF 是字面匹配，中文 query 命中英文工具描述依赖 `QUERY_EXPAND`。要更强语义需要 embedding。
3. **同名工具歧义**：多个服务端有同名工具时（`fs_a_read_file` / `fs_b_read_file`）分数接近，检索无法区分业务归属，需要额外规则或提示词。
4. `remote` **服务端要自己启动**：默认配置指向 `http://127.0.0.1:8010/mcp_server`，未启动会导致连接失败。
5. `demo.py` **不是可运行程序**：它是检索方案的设计草稿（依赖的 `servers`、`TfidfVectorizer` 等未定义），仅作参考。
6. **版本敏感**：`fastmcp==4.0.5`。MCP SDK v2 把 `Tool.inputSchema` 改名为 `input_schema`，本项目的 `client.py` 使用新字段名。

---



## 11. 验证方式

```bash
# 1) 语法检查
python -m py_compile mcp_server/*.py agent/*.py llm/*.py

# 2) 连接 MCP 并列出工具（需要 remote 已启动，或先注释掉）
python mcp_server/client.py

# 3) 交互式对话
python main.py
```

无真实 LLM 时，也可以注入假客户端验证 Agent 循环：

```python
import asyncio
from types import SimpleNamespace
from agent.agent import agent

CONF = {  # 用两个 stdio 服务端模拟“多服务端带前缀”
    "fs_a": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "."]},
    "fs_b": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "."]},
}

class FakeLLM:
    def __init__(self): self.calls = 0
    def invoke(self, messages, tools=None):
        self.calls += 1
        if self.calls == 1:
            tc = SimpleNamespace(id="c1", function=SimpleNamespace(
                name="fs_a_read_file", arguments='{"path": "requirement.txt"}'))
            return SimpleNamespace(content=None, tool_calls=[tc])
        return SimpleNamespace(content="done", tool_calls=None)

async def main():
    a = agent(server_configs=CONF)
    a.llmclient = FakeLLM()
    print((await a.chat("读取 requirement.txt 的内容")).content)
    await a.aclose()

asyncio.run(main())
```

典型输出：

```
[MCP] 已加载 28 个工具，来自 2 个服务端
[检索] '读取 requirement.txt 的内容' → 命中 8/28 个工具
    0.308  fs_a_read_file
    0.295  fs_b_read_file
    ...
循环1次 → 模型请求 fs_a_read_file → 读到文件内容
循环2次 → 最终回答
```

---



## 12. 扩展指南

**新增一个 MCP 服务端**

```python
# mcp_server/client.py
SERVER_CONFIGS = {
    "filesystem": {...},
    "remote": {...},
    "github": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-github"],
               "env": {"GITHUB_TOKEN": "..."}},
}
```

工具会自动进入注册表与检索器；建议同时在 `ToolRetriever.QUERY_EXPAND` 里补充该领域中文词。

**只给 LLM 全量工具（关闭检索）**

```python
tools = registry.to_llm_tools()          # keys=None → 全部工具
```

