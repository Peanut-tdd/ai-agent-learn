SEP = "__"
RISKY_VERBS = ("create", "write", "send", "move", "transition", "upload", "delete")


class ToolRegistry:
    def __init__(self, servers):
        self.index = {}
        self.clients = servers
        for srv_name, srv in servers.items():
            if srv_name == "__core__":
                continue
            for t in srv._tools.values():
                self.index[f"{srv_name}{SEP}{t['name']}"] = (srv_name, t["name"], t)

    def to_llm_tools(self, keys):
        out = []
        for k in keys:
            _, _, t = self.index[k]
            out.append({"type": "function", "function": {
                "name": k,                       # 带前缀，只给 LLM 看
                "description": t["description"],
                "parameters": t["inputSchema"],
            }})
        return out

    async def dispatch(self, llm_name, args):
        srv, real, _ = self.index[llm_name]      # 拆前缀
        return await self.clients[srv].call_tool(real, args)


registry = ToolRegistry(servers)
ALL_KEYS = list(registry.index)

# =====================================================================
# 3. token 统计（离线可用）
# =====================================================================
_CJK = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")


def ntok(obj) -> int:
    s = json.dumps(obj, ensure_ascii=False)
    return len(_CJK.findall(s)) + max(1, round(len(_CJK.sub("", s)) / 4))


CORE_TOOL = {"type": "function", "function": {
    "name": "finish", "description": SERVERS["__core__"][0][1],
    "parameters": SERVERS["__core__"][0][2]}}

full_tools = registry.to_llm_tools(ALL_KEYS) + [CORE_TOOL]
FULL_COST = ntok(full_tools)

print(f"工具总数：{len(ALL_KEYS)}（来自 {len(servers) - 1} 个 server）")
print(f"[基线] 全量加载工具定义 = {FULL_COST} tokens/轮\n")

# =====================================================================
# 4. 检索器：TF-IDF 混合索引（词级 + 字符 n-gram），分字段加权
# =====================================================================

# 极简查询改写：把缩写/口语补成描述里会出现的词
QUERY_EXPAND = {
    "pr": "PR pull request 合并请求",
    "mr": "MR pull request 合并请求",
    "工单": "工单 issue",
    "sql": "SQL 数据库 查询",
    "表结构": "表结构 字段 describe",
    "频道": "频道 channel",
}


def expand(q: str) -> str:
    low = q.lower()
    extra = [v for k, v in QUERY_EXPAND.items() if k in low]
    return q + " " + " ".join(extra)


class ToolRetriever:
    def __init__(self, registry):
        self.keys = list(registry.index)
        self.risky = {k for k in self.keys
                      if any(w in registry.index[k][1] for w in RISKY_VERBS)}
        # 分字段加权拼文档：名字×3 + 别名×3 + 描述×1 + 参数名×1 + server×2
        docs = []
        for k in self.keys:
            srv, real, t = registry.index[k]
            params = " ".join(t["inputSchema"].get("properties", {}).keys())
            human = k.replace(SEP, " ")
            docs.append(" ".join([human] * 3 + [t["tags"]] * 3 +
                                 [t["description"], params] + [srv, real] * 2))
        self.vw = TfidfVectorizer(analyzer="word", lowercase=True)
        self.vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), lowercase=True)
        self.M = np.hstack([self.vw.fit_transform(docs).toarray() * 0.4,
                            self.vc.fit_transform(docs).toarray()])

    def _qvec(self, q):
        q = expand(q)
        return np.hstack([self.vw.transform([q]).toarray() * 0.4,
                          self.vc.transform([q]).toarray()])

    def search(self, query, k=6, min_score=0.06):
        sims = cosine_similarity(self._qvec(query), self.M)[0]
        out = []
        for i in np.argsort(-sims):
            if sims[i] < min_score or len(out) >= k:
                break
            out.append((self.keys[i], float(sims[i])))
        return out


retriever = ToolRetriever(registry)

print("=" * 76)
print("策略 A：检索器自动召回（对用户和 LLM 都透明，无额外往返）")
print("=" * 76)

QUERIES = [
    "帮我看看 backend-api 这个仓库最近三天有哪些 PR 还没合并",
    "查一下 orders 表昨天订单量是多少，顺便看下这条 SQL 走没走索引",
    "把刚才那份对账结果发到 #finance 频道",
    "PROJ-1024 这个工单推进到 Done，并且加一条评论说明原因",
]

for q in QUERIES:
    picked = retriever.search(q, k=6)
    cost = ntok(registry.to_llm_tools([k for k, _ in picked]) + [CORE_TOOL])
    print(f"\nquery: {q}")
    for kk, s in picked:
        print(f"    {s:.3f}  {kk}{'  ⚠️副作用' if kk in retriever.risky else ''}")
    print(f"    → {len(picked) + 1} 个工具 / {cost} tokens = 全量的 {cost/FULL_COST:.0%}")

# =====================================================================
# 5. 策略 B：元工具 find_tools
# =====================================================================

FIND_TOOLS_META = {"type": "function", "function": {
    "name": "find_tools",
    "description": ("检索当前可用的扩展工具。当你需要的能力不在上面这几个工具里时，"
                    "用自然语言描述需求调用它；它会返回相关工具的完整定义，之后你才能调用它们。"
                    "一次没搜到就换个更具体的说法再搜一次。"),
    "parameters": schema({"query": {"type": "string"}, "limit": {"type": "integer"}}, ["query"]),
}}

print("\n" + "=" * 76)
print("策略 B：元工具 find_tools（多一轮往返，工具池可以上千）")
print("=" * 76)

META_COST = ntok([CORE_TOOL, FIND_TOOLS_META])
print(f"\n第 1 轮只挂元工具：{META_COST} tokens（vs 全量 {FULL_COST}）\n")
