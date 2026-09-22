"""工具向量检索：用 TF-IDF 混合索引按用户语句召回最相关的工具。

思路参考 demo.py 的 ToolRetriever：
- 分字段加权拼「工具文档」（工具名×3、server×2、描述×1、参数名×1）；
- 词级 token + 中文字符 n-gram（2~4）两种特征混合，兼顾英文标识符和中文口语；
- 用余弦相似度排序，取 top-k，并用 min_score 过滤掉明显不相关的工具。

与原 demo 的区别：这里不依赖 sklearn / numpy，纯 Python 稀疏向量实现。
工具规模在几百个以内时速度足够，也方便离线运行。
需要跨语言语义匹配（例如中文 query 对英文描述）时，可以再补 QUERY_EXPAND，
或把 _vector() 换成 embedding 模型。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp_server.tool_registry import ToolRegistry


class ToolRetriever:
    """按用户语句从 ToolRegistry 中召回相关工具。

    用法::

        retriever = ToolRetriever(registry)
        picked = retriever.search("帮我看下目录里有哪些文件")   # [(工具名, 分数), ...]
        keys = retriever.search_keys("上海今天天气怎么样", k=4)
    """

    # 查询改写：把常见中文需求补成工具描述里会出现的英文词（demo 里 QUERY_EXPAND 的思路）。
    # key 用小写子串匹配，命中就把 value 里的词追加到 query 后面。
    QUERY_EXPAND: dict[str, str] = {
        "文件": "file files content",
        "内容": "content file",
        "读": "read view open cat",
        "查看": "read view list show",
        "看一下": "read view list show",
        "打开": "open read",
        "写": "write save create",
        "保存": "write save",
        "修改": "edit write modify",
        "编辑": "edit modify",
        "替换": "edit replace",
        "目录": "directory folder list",
        "文件夹": "directory folder",
        "列表": "list directory",
        "搜索": "search find glob",
        "查找": "search find glob",
        "搜": "search find glob",
        "找": "find search",
        "移动": "move rename",
        "重命名": "move rename",
        "改名": "rename move",
        "创建": "create mkdir new",
        "新建": "create mkdir new",
        "删除": "delete remove",
        "删": "delete remove",
        "信息": "info metadata stat",
        "详情": "info metadata detail",
        "大小": "size info",
        "权限": "allowed permissions access",
        "允许": "allowed permissions access",
        "根目录": "allowed directories root",
        "树": "tree directory recursive",
        "天气": "weather temperature forecast",
        "气温": "weather temperature",
        "温度": "weather temperature",
        "预报": "weather forecast",
        "警报": "alerts weather warning",
        "预警": "alerts weather warning",
        "计算": "add sum multiply calculate",
        "相加": "add sum",
        "求和": "add sum",
        "相乘": "multiply product",
        "乘积": "multiply product",
        "加": "add sum",
        "乘": "multiply product",
    }

    # 词级 token：连续英文字母/数字
    _WORD_RE = re.compile(r"[a-z0-9]+")
    # 中文字符连续段
    _CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")

    def __init__(self, registry: "ToolRegistry"):
        self.keys: list[str] = registry.keys
        docs = [self._document(registry, key) for key in self.keys]
        self._idf = self._build_idf(docs)
        self._vectors = [self._vector(tokens) for tokens in docs]

    # ---------- 分词 ----------
    @classmethod
    def _tokenize(cls, text: str) -> list[str]:
        """英文按词切分，中文额外补充 2~4 字符 n-gram（无需分词器）。"""
        text = text.lower()
        tokens = cls._WORD_RE.findall(text)
        for run in cls._CJK_RUN_RE.findall(text):
            for n in (2, 3, 4):
                if len(run) >= n:
                    tokens.extend(run[i : i + n] for i in range(len(run) - n + 1))
        return tokens

    @classmethod
    def _expand(cls, query: str) -> str:
        low = query.lower()
        extra = [words for key, words in cls.QUERY_EXPAND.items() if key in low]
        return query + " " + " ".join(extra)

    # ---------- 向量化 ----------
    def _build_idf(self, docs: list[list[str]]) -> dict[str, float]:
        df: Counter[str] = Counter()
        for tokens in docs:
            df.update(set(tokens))
        n = max(1, len(docs))
        # 平滑 IDF，避免除零；未登录词在 _vector 里按 0 处理
        return {term: math.log((1 + n) / (1 + count)) + 1.0 for term, count in df.items()}

    def _vector(self, tokens: list[str]) -> dict[str, float]:
        tf = Counter(tokens)
        vec = {
            term: (1.0 + math.log(count)) * self._idf.get(term, 0.0)
            for term, count in tf.items()
        }
        norm = math.sqrt(sum(value * value for value in vec.values()))
        if not norm:
            return {}
        return {term: value / norm for term, value in vec.items()}

    @staticmethod
    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        if len(a) > len(b):
            a, b = b, a
        return sum(value * b.get(term, 0.0) for term, value in a.items())

    @classmethod
    def _document(cls, registry: "ToolRegistry", key: str) -> list[str]:
        tool = registry.get(key) or {}
        server = registry.server_of(key) or ""
        schema = tool.get("input_schema") or {}
        params = " ".join(schema.get("properties", {}).keys())
        description = tool.get("description") or ""
        # 分字段加权：工具名×3 + server×2 + 描述×1 + 参数名×1
        return (
            cls._tokenize(key.replace("_", " ")) * 3
            + cls._tokenize(server) * 2
            + cls._tokenize(description)
            + cls._tokenize(params)
        )

    # ---------- 检索 ----------
    def search(
        self, query: str, k: int = 8, min_score: float = 0.05
    ) -> list[tuple[str, float]]:
        """返回 [(工具名, 余弦相似度), ...]，按分数从高到低，最多 k 个。"""
        query_vec = self._vector(self._tokenize(self._expand(query)))
        scored = [
            (self.keys[i], self._cosine(query_vec, self._vectors[i]))
            for i in range(len(self.keys))
        ]
        scored = [item for item in scored if item[1] >= min_score]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:k]

    def search_keys(
        self, query: str, k: int = 8, min_score: float = 0.05
    ) -> list[str]:
        """只返回工具名列表，方便直接丢给 ToolRegistry.to_llm_tools()。"""
        return [key for key, _ in self.search(query, k=k, min_score=min_score)]
