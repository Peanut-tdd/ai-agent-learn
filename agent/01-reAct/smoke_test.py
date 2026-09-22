"""冒烟测试：验证多轮记忆、死循环检测、Pydantic 契约校验、结构化错误（全部 mock，不调真实 LLM/网络）。"""
import sys
from types import SimpleNamespace

import requests
import time

sys.path.insert(0, ".")

import agent.agent as agent_mod
from agent.agent import (
    RetryClass,
    _classify_error,
    agent,
    MAX_TURNS,
    STRATEGY_HINT,
    FALLBACK_ANSWER,
)
from circuit_breaker import BreakerState, CircuitBreaker
from tools import TOOL_FUNCTIONS, ToolResult, ToolSpec
from tools.basic_calculator import Args as CalcArgs, basic_calculator
from tools.get_weather import Args as WeatherArgs
from tools.get_attraction import Args as AttrArgs
from tools.simulate_payment import ChargeArgs, CreateOrderArgs, _LEDGER
from tools.tool_result import SideEffect


agent_mod.RETRY_BACKOFF_SECONDS = 0.01  # 加速退避，避免测试 sleep 太久


def tc(name, args):
    return SimpleNamespace(id="call_x", function=SimpleNamespace(name=name, arguments=args))


def mk(name, handler, args_model, side_effect=SideEffect.READ_ONLY):
    return ToolSpec(name=name, handler=handler, args_model=args_model, side_effect=side_effect)


class StubLLM:
    """按脚本依次返回 tool_calls 或最终回答，并记录每次收到的 messages。"""

    def __init__(self, script):
        self.script = script
        self.i = 0
        self.calls = []

    def generate(self, messages):
        self.calls.append(messages)
        step = self.script[self.i]
        self.i += 1
        if step[0] == "raise":
            raise step[1]
        if step[0] == "answer":
            return SimpleNamespace(content=step[1], tool_calls=None)
        return SimpleNamespace(content=None, tool_calls=[tc(n, a) for n, a in step[1]])


def make_agent(stub, specs):
    a = agent.__new__(agent)  # 跳过 __init__（避免读 .env / 建真实 client）
    a.llmclient = stub
    a.history_msg = []
    a._blocked_writes = set()
    a._breakers = {}
    a._breakers_lock = __import__("threading").Lock()
    agent_mod.TOOL_FUNCTIONS = specs
    return a


def last_tool_content(stub):
    return [m["content"] for m in stub.calls[-1] if m.get("role") == "tool"][-1]


def check(name, cond):
    print(f"{'✅' if cond else '❌'} {name}")
    if not cond:
        sys.exit(1)


def boom(city):
    raise ValueError("boom")


# ── 1. ToolResult 结构化输出 ──────────────────────────────────────────────────
r = ToolResult(success=False, error=ValueError("division by zero"))
check("默认 code 为 EXECUTION_ERROR", str(r) == "Error(EXECUTION_ERROR): ValueError: division by zero")
r2 = ToolResult(success=False, code="VALIDATION_ERROR", error=ValueError("bad city"))
check("显式 code 输出", str(r2) == "Error(VALIDATION_ERROR): ValueError: bad city")
check("成功结果只输出 data", str(ToolResult(success=True, data="晴 20度")) == "晴 20度")

# ── 2. 工具运行时错误：handler 抛异常 ────────────────────────────────────────
try:
    basic_calculator("divide", 1, 0)
    check("除零应抛异常", False)
except ValueError as e:
    check("除零抛 ValueError", "division by zero" in str(e))

# ── 3. Pydantic 契约校验（真实注册表）───────────────────────────────────────
stub = StubLLM([
    ("tools", [("basic_calculator", '{"operation": "mod", "a": 1, "b": 2}')]),
    ("answer", "好了"),
])
a = make_agent(stub, TOOL_FUNCTIONS)
a.chat("算一下")
c = last_tool_content(stub)
check("非法 operation → VALIDATION_ERROR", c.startswith("Error(VALIDATION_ERROR)"))
check("校验错误含字段明细", "operation" in c)

stub = StubLLM([("tools", [("get_weather", "{}")]), ("answer", "好了")])
a = make_agent(stub, TOOL_FUNCTIONS)
a.chat("查天气")
c = last_tool_content(stub)
check("缺必填字段 → VALIDATION_ERROR", c.startswith("Error(VALIDATION_ERROR)") and "city" in c)

stub = StubLLM([("tools", [("get_weather", '{"city": 123}')]), ("answer", "好了")])
a = make_agent(stub, TOOL_FUNCTIONS)
a.chat("查天气")
c = last_tool_content(stub)
check("类型错误 → VALIDATION_ERROR", c.startswith("Error(VALIDATION_ERROR)"))

stub = StubLLM([("tools", [("get_weather", '{"city":')]), ("answer", "好了")])
a = make_agent(stub, TOOL_FUNCTIONS)
a.chat("查天气")
check("非法 JSON → INVALID_JSON", last_tool_content(stub).startswith("Error(INVALID_JSON)"))

stub = StubLLM([("tools", [("fly_to_moon", "{}")]), ("answer", "好了")])
a = make_agent(stub, TOOL_FUNCTIONS)
a.chat("飞")
check("未注册工具 → UNKNOWN_TOOL", last_tool_content(stub).startswith("Error(UNKNOWN_TOOL)"))

# ── 4. 多轮记忆：第二轮能看到第一轮的 user/assistant ─────────────────────────
stub = StubLLM([
    ("tools", [("get_weather", '{"city": "北京"}')]),
    ("answer", "北京多云，21度"),
    ("tools", [("get_attraction", '{"city": "北京", "weather": "多云"}')]),
    ("answer", "推荐颐和园"),
])
a = make_agent(stub, {
    "get_weather": mk("get_weather", lambda city: "晴 20度", WeatherArgs),
    "get_attraction": mk("get_attraction", lambda city, weather: "故宫", AttrArgs),
})
check("第一轮返回回答", a.chat("北京天气怎么样").content == "北京多云，21度")
check("第二轮返回回答", a.chat("那景点呢").content == "推荐颐和园")
check("持久化 2 对 user/assistant",
      [m["role"] for m in a.history_msg] == ["user", "assistant", "user", "assistant"])
check("第二轮 context 包含第一轮历史", stub.calls[2][1]["content"] == "北京天气怎么样")

# ── 5. 死循环检测：同工具同参连续失败 2 次 → 注入换策略提示 ─────────────────
stub = StubLLM([
    ("tools", [("get_weather", '{"city": "北京"}')]),
    ("tools", [("get_weather", '{"city": "北京"}')]),
    ("tools", [("get_weather", '{"city": "北京"}')]),
    ("answer", "我放弃了"),
])
a = make_agent(stub, {"get_weather": mk("get_weather", boom, WeatherArgs)})
a.chat("查天气")
hint_msgs = [m for m in stub.calls[2] if (m.get("content") or "") and "换一种完全不同的策略" in m.get("content")]
check("连续失败 2 次后注入换策略提示", len(hint_msgs) == 1)
check("提示中 n=2", STRATEGY_HINT.format(n=2) in hint_msgs[0]["content"])

# ── 5b. 换策略提示的位置：多 tool_call 时不能插在 tool 结果中间 ─────────────
stub = StubLLM([
    ("tools", [("get_weather", '{"city": "北京"}')]),
    ("tools", [("get_weather", '{"city": "北京"}'), ("get_weather", '{"city": "北京"}')]),  # 同轮两次同参数调用，第2次触发阈值
    ("answer", "结束"),
])
a = make_agent(stub, {"get_weather": mk("get_weather", boom, WeatherArgs)})
a.chat("查天气")
msgs = stub.calls[-1]
hint_idx = next(i for i, m in enumerate(msgs) if "换一种完全不同的策略" in (m.get("content") or ""))
after_hint_tools = [m for m in msgs[hint_idx + 1:] if m.get("role") == "tool"]
check("hint 注入后不再有 tool 结果（消息顺序合法）", len(after_hint_tools) == 0)
check("hint 之前的 tool 结果完整", sum(1 for m in msgs[:hint_idx] if m.get("role") == "tool") == 3)

# ── 6. 失败计数跨 chat() 重置：每轮各失败 1 次不触发提示 ─────────────────────
stub = StubLLM([
    ("tools", [("get_weather", '{"city": "北京"}')]),
    ("answer", "放弃1"),
    ("tools", [("get_weather", '{"city": "北京"}')]),
    ("answer", "放弃2"),
])
a = make_agent(stub, {"get_weather": mk("get_weather", boom, WeatherArgs)})
a.chat("查天气1")
a.chat("查天气2")
hints = [m for m in stub.calls[3] if "换一种完全不同的策略" in (m.get("content") or "")]
check("跨轮各失败 1 次不触发死循环", len(hints) == 0)

# ── 7. 成功一次重置计数 ───────────────────────────────────────────────────────
stub = StubLLM([
    ("tools", [("get_weather", '{"city": "北京"}')]),
    ("tools", [("get_weather", '{"city": "北京"}')]),   # 第2次失败→提示
    ("tools", [("get_weather", '{"city": "上海"}')]),   # 换参数成功 → 重置
    ("tools", [("get_weather", '{"city": "上海"}')]),   # 又失败1次 → 不触发
    ("answer", "完了"),
])
a = make_agent(stub, {
    "get_weather": mk("get_weather",
                      lambda city: boom(city) if city == "北京" else "上海晴", WeatherArgs),
})
a.chat("查天气")
# messages 是同一可变列表，calls 各元素均为其引用；直接数最终列表中的提示消息条数
final_msgs = stub.calls[-1]
hint_msgs = [m for m in final_msgs if "换一种完全不同的策略" in (m.get("content") or "")]
check("换参数成功后重新计数，整个对话仅注入 1 次提示", len(hint_msgs) == 1)

# ── 8. 硬上限兜底：10 轮全失败 → 返回兜底回答而非抛异常 ─────────────────────
script = [("tools", [("get_weather", '{"city": "北京"}')])] * MAX_TURNS
stub = StubLLM(script)
a = make_agent(stub, {"get_weather": mk("get_weather", boom, WeatherArgs)})
try:
    out = a.chat("查天气")
    check("硬上限返回兜底回答", out.content == FALLBACK_ANSWER)
except Exception as e:
    check(f"硬上限不应抛异常（实际: {e}）", False)

# ── 9. LLM 超时重试：失败一次后注入提示继续 ──────────────────────────────────
stub = StubLLM([
    ("raise", TimeoutError("LLM timeout")),
    ("answer", "重试成功"),
])
a = make_agent(stub, {"get_weather": mk("get_weather", lambda city: "晴", WeatherArgs)})
check("LLM 超时后重试成功", a.chat("查天气").content == "重试成功")
retry_hint = [m for m in stub.calls[-1] if "上一次模型响应超时" in (m.get("content") or "")]
check("重试前注入了超时提示", len(retry_hint) == 1)

# ── 10. LLM 连续超时超过重试上限 → 兜底回答，不崩 ────────────────────────────
from agent.agent import MAX_LLM_RETRIES

stub = StubLLM([("raise", TimeoutError("x"))] * (MAX_LLM_RETRIES + 1))
a = make_agent(stub, {"get_weather": mk("get_weather", lambda city: "晴", WeatherArgs)})
try:
    out = a.chat("查天气")
    check("LLM 重试耗尽返回兜底回答", out.content == FALLBACK_ANSWER)
except Exception as e:
    check(f"LLM 重试耗尽不应抛异常（实际: {e}）", False)

# ── 11. 总时间预算：超预算立即兜底 ───────────────────────────────────────────
agent_mod.MAX_TOTAL_SECONDS = 0
stub = StubLLM([("tools", [("get_weather", '{"city": "北京"}')])] * 10)
a = make_agent(stub, {"get_weather": mk("get_weather", lambda city: "晴", WeatherArgs)})
out = a.chat("查天气")
check("总时间预算耗尽返回兜底回答", out.content == FALLBACK_ANSWER)
agent_mod.MAX_TOTAL_SECONDS = 180  # 恢复预算，避免影响后续测试

# ── 12. 非幂等写：at-most-once，重复调用被拦截，查询工具确认真相 ────────────
stub = StubLLM([
    ("tools", [("charge_payment", '{"user": "u1", "amount": 100}')]),
    ("tools", [("charge_payment", '{"user": "u1", "amount": 100}')]),  # 应被拦截
    ("tools", [("query_payment", '{"user": "u1"}')]),
    ("answer", "确认完毕"),
])
a = make_agent(stub, TOOL_FUNCTIONS)
a.chat("扣款")
tool_msgs = [m["content"] for m in stub.calls[-1] if m.get("role") == "tool"]
check("首次扣款返回 EXECUTION_ERROR（超时）", tool_msgs[0].startswith("Error(EXECUTION_ERROR): TimeoutError"))
check("重复扣款被拦截 BLOCKED_WRITE", tool_msgs[1].startswith("Error(BLOCKED_WRITE)"))
check("query_payment 确认钱确实被扣了", "累计扣款: 100.0" in tool_msgs[2])
check("扣款 handler 实际只执行了一次", _LEDGER.get("u1") == 100.0)

# ── 13. 幂等写：代码层自动重试，第二次幂等命中 ───────────────────────────────
stub = StubLLM([
    ("tools", [("create_order", '{"order_id": "ord-1", "item": "书", "qty": 2}')]),
    ("answer", "下单成功"),
])
a = make_agent(stub, TOOL_FUNCTIONS)
check("幂等下单成功", a.chat("下单").content == "下单成功")
tool_msgs = [m["content"] for m in stub.calls[-1] if m.get("role") == "tool"]
check("自动重试后幂等命中", "幂等命中" in tool_msgs[0])

# ── 14. 幂等写重试耗尽：如实报错，handler 调用次数 = 首次 + 3 次重试 ─────────
counter = {"n": 0}

def always_timeout(**kw):
    counter["n"] += 1
    raise TimeoutError("always")

stub = StubLLM([
    ("tools", [("create_order", '{"order_id": "x", "item": "a", "qty": 1}')]),
    ("answer", "done"),
])
a = make_agent(stub, {"create_order": mk("create_order", always_timeout, CreateOrderArgs, SideEffect.IDEMPOTENT)})
a.chat("下单")
tool_msgs = [m["content"] for m in stub.calls[-1] if m.get("role") == "tool"]
check("幂等重试耗尽如实报错", "含自动重试 3 次" in tool_msgs[0])
check("handler 共被调用 4 次（首次+3重试）", counter["n"] == 4)

# ── 14b. 幂等写不可重试错误：不自动重试，错误信息反映真实次数 ─────────────────
counter2 = {"n": 0}

def always_business(**kw):
    counter2["n"] += 1
    raise ValueError("业务错误")

stub = StubLLM([
    ("tools", [("create_order", '{"order_id": "x2", "item": "a", "qty": 1}')]),
    ("answer", "done"),
])
a = make_agent(stub, {"create_order": mk("create_order", always_business, CreateOrderArgs, SideEffect.IDEMPOTENT)})
a.chat("下单")
tool_msgs = [m["content"] for m in stub.calls[-1] if m.get("role") == "tool"]
check("不可重试错误如实报实际次数（执行1次含0重试）", "执行 1 次（含自动重试 0 次）" in tool_msgs[0])
check("不可重试错误 handler 只调用 1 次", counter2["n"] == 1)

# ── 15. 非幂等拦截跨 chat() 持续生效 ─────────────────────────────────────────
stub1 = StubLLM([("tools", [("charge_payment", '{"user": "u2", "amount": 50}')]), ("answer", "a")])
a = make_agent(stub1, TOOL_FUNCTIONS)
a.chat("扣款")
stub2 = StubLLM([("tools", [("charge_payment", '{"user": "u2", "amount": 50}')]), ("answer", "b")])
a.llmclient = stub2
a.chat("再扣")
tool_msgs = [m["content"] for m in stub2.calls[-1] if m.get("role") == "tool"]
check("跨 chat() 依然拦截", tool_msgs[0].startswith("Error(BLOCKED_WRITE)"))

# ── 16. 参数校验失败不触发拦截（根本没发出去）────────────────────────────────
stub = StubLLM([
    ("tools", [("charge_payment", '{"user": "u3", "amount": -5}')]),   # 负数金额 → 校验失败
    ("tools", [("charge_payment", '{"user": "u3", "amount": 10}')]),  # 合法 → 真正执行
    ("answer", "done"),
])
a = make_agent(stub, TOOL_FUNCTIONS)
a.chat("扣款")
tool_msgs = [m["content"] for m in stub.calls[-1] if m.get("role") == "tool"]
check("参数校验失败返回 VALIDATION_ERROR", tool_msgs[0].startswith("Error(VALIDATION_ERROR)"))
check("校验失败后仍可正常调用（未被拦截）", tool_msgs[1].startswith("Error(EXECUTION_ERROR)"))

# ── 17. 非幂等写：确定未执行的错误（连接失败）→ 代码层自动重试 ────────────────
calls = {"n": 0}

def flaky(**kw):
    calls["n"] += 1
    if calls["n"] == 1:
        raise requests.exceptions.ConnectionError("conn refused")
    return "扣款成功"

stub = StubLLM([("tools", [("charge_payment", '{"user": "u4", "amount": 10}')]), ("answer", "done")])
a = make_agent(stub, {"charge_payment": mk("charge_payment", flaky, ChargeArgs, SideEffect.NON_IDEMPOTENT)})
a.chat("扣款")
tool_msgs = [m["content"] for m in stub.calls[-1] if m.get("role") == "tool"]
check("非幂等连接失败自动重试后成功", "扣款成功" in tool_msgs[0])
check("handler 共调用 2 次（首次+1次重试）", calls["n"] == 2)

# ── 18. 非幂等写：SAFE 重试耗尽 → 不拦截，可换参数再试 ───────────────────────
def always_conn(**kw):
    raise requests.exceptions.ConnectionError("refused")

stub = StubLLM([
    ("tools", [("charge_payment", '{"user": "u5", "amount": 10}')]),
    ("tools", [("charge_payment", '{"user": "u5", "amount": 20}')]),  # 换参数，应未被拦截
    ("answer", "done"),
])
a = make_agent(stub, {"charge_payment": mk("charge_payment", always_conn, ChargeArgs, SideEffect.NON_IDEMPOTENT)})
a.chat("扣款")
tool_msgs = [m["content"] for m in stub.calls[-1] if m.get("role") == "tool"]
check("连接失败重试耗尽返回 EXECUTION_ERROR（非 BLOCKED）", tool_msgs[0].startswith("Error(EXECUTION_ERROR)"))
check("错误消息说明确定未执行", "确定未执行" in tool_msgs[0])
check("SAFE 耗尽后未拦截，换参数可再调", tool_msgs[1].startswith("Error(EXECUTION_ERROR)"))

# ── 19. 幂等写：业务错误（NO）不自动重试 ─────────────────────────────────────
counter = {"n": 0}

def biz_err(**kw):
    counter["n"] += 1
    raise ValueError("余额不足")

stub = StubLLM([("tools", [("create_order", '{"order_id": "y", "item": "a", "qty": 1}')]), ("answer", "done")])
a = make_agent(stub, {"create_order": mk("create_order", biz_err, CreateOrderArgs, SideEffect.IDEMPOTENT)})
a.chat("下单")
tool_msgs = [m["content"] for m in stub.calls[-1] if m.get("role") == "tool"]
check("业务错误不自动重试（handler 只调 1 次）", counter["n"] == 1)
check("业务错误直接返回", "余额不足" in tool_msgs[0])

# ── 20. 错误分类器单元测试：状态确定性判定 ───────────────────────────────────
check("ConnectTimeout → SAFE（连接未建立）", _classify_error(requests.exceptions.ConnectTimeout()) == RetryClass.SAFE)
check("ReadTimeout → UNKNOWN（已发出）", _classify_error(requests.exceptions.ReadTimeout()) == RetryClass.UNKNOWN)
check("TimeoutError → UNKNOWN（保守）", _classify_error(TimeoutError("x")) == RetryClass.UNKNOWN)
check("ConnectionError → SAFE（未送达）", _classify_error(requests.exceptions.ConnectionError()) == RetryClass.SAFE)
check("ConnectionResetError → UNKNOWN（可能已执行）", _classify_error(ConnectionResetError()) == RetryClass.UNKNOWN)
resp = requests.Response(); resp.status_code = 429
check("HTTP 429 → SAFE", _classify_error(requests.exceptions.HTTPError("x", response=resp)) == RetryClass.SAFE)
resp = requests.Response(); resp.status_code = 504
check("HTTP 504 → UNKNOWN", _classify_error(requests.exceptions.HTTPError("x", response=resp)) == RetryClass.UNKNOWN)
resp = requests.Response(); resp.status_code = 404
check("HTTP 404 → NO", _classify_error(requests.exceptions.HTTPError("x", response=resp)) == RetryClass.NO)
check("业务 ValueError → NO", _classify_error(ValueError("余额不足")) == RetryClass.NO)

# ── 21. 熔断：连续失败触发 OPEN → 后续快速失败，handler 不再被调用 ───────────
agent_mod.CIRCUIT_FAILURE_THRESHOLD = 2
agent_mod.CIRCUIT_COOLDOWN_SECONDS = 30  # 保持 OPEN
counter = {"n": 0}

def fail_always(**kw):
    counter["n"] += 1
    raise TimeoutError("db down")

script = [("tools", [("get_weather", '{"city": "北京"}')])] * 6 + [("answer", "done")]
stub = StubLLM(script)
a = make_agent(stub, {"get_weather": mk("get_weather", fail_always, WeatherArgs)})
a.chat("查天气")
tool_msgs = [m["content"] for m in stub.calls[-1] if m.get("role") == "tool"]
check("连续失败 2 次后触发熔断", tool_msgs[0].startswith("Error(EXECUTION_ERROR)") and tool_msgs[1].startswith("Error(EXECUTION_ERROR)"))
check("熔断后返回 CIRCUIT_OPEN", tool_msgs[2].startswith("Error(CIRCUIT_OPEN)"))
check("熔断后 handler 不再被调用", counter["n"] == 2)

# ── 22. 半开探测成功 → 恢复 CLOSED ───────────────────────────────────────────
agent_mod.CIRCUIT_COOLDOWN_SECONDS = 0  # 立即进入半开（测试用）
counter2 = {"n": 0}

def recover(**kw):
    counter2["n"] += 1
    if counter2["n"] <= 2:
        raise TimeoutError("db down")
    return "恢复了"

stub = StubLLM([
    ("tools", [("get_weather", '{"city": "北京"}')]),
    ("tools", [("get_weather", '{"city": "北京"}')]),   # 第2次失败 → 熔断
    ("tools", [("get_weather", '{"city": "北京"}')]),   # 半开探测 → 成功 → 恢复
    ("tools", [("get_weather", '{"city": "北京"}')]),   # 已恢复，正常
    ("answer", "ok"),
])
a = make_agent(stub, {"get_weather": mk("get_weather", recover, WeatherArgs)})
a.chat("查天气")
tool_msgs = [m["content"] for m in stub.calls[-1] if m.get("role") == "tool"]
check("半开探测成功恢复", "恢复了" in tool_msgs[2] and "恢复了" in tool_msgs[3])

# ── 23. 熔断器状态机单元测试 ─────────────────────────────────────────────────
cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=30)
cb.record_failure(); cb.record_failure()
check("连续失败触发 OPEN", cb.state == BreakerState.OPEN)
check("OPEN 期间拒绝调用", cb.is_open() is True)
cb._opened_at = time.monotonic() - 31  # 模拟冷却结束
check("冷却结束后半开放行探测", cb.is_open() is False)
cb.record_failure()  # 探测失败
check("探测失败重新熔断", cb.state == BreakerState.OPEN and cb.is_open() is True)
cb._opened_at = time.monotonic() - 31
check("再次半开放行", cb.is_open() is False)
cb.record_success()
check("探测成功恢复 CLOSED", cb.state == BreakerState.CLOSED and cb.is_open() is False)

# ── 24. 业务错误（NO）不计入熔断 ─────────────────────────────────────────────
agent_mod.CIRCUIT_FAILURE_THRESHOLD = 2
agent_mod.CIRCUIT_COOLDOWN_SECONDS = 30
counter4 = {"n": 0}

def biz_fail(**kw):
    counter4["n"] += 1
    raise ValueError("余额不足")

script = [("tools", [("create_order", '{"order_id": "z", "item": "a", "qty": 1}')])] * 4 + [("answer", "ok")]
stub = StubLLM(script)
a = make_agent(stub, {"create_order": mk("create_order", biz_fail, CreateOrderArgs, SideEffect.IDEMPOTENT)})
a.chat("下单")
tool_msgs = [m["content"] for m in stub.calls[-1] if m.get("role") == "tool"]
check("业务错误不触发熔断", all("CIRCUIT_OPEN" not in m for m in tool_msgs))
check("业务错误下 handler 一直被调用", counter4["n"] == 4)

# ── 25. 熔断按工具名独立：A 熔断不影响 B ─────────────────────────────────────
agent_mod.CIRCUIT_FAILURE_THRESHOLD = 2
agent_mod.CIRCUIT_COOLDOWN_SECONDS = 30
counterA = {"n": 0}
counterB = {"n": 0}

def failA(**kw):
    counterA["n"] += 1
    raise TimeoutError("A down")

def okB(**kw):
    counterB["n"] += 1
    return "B 正常"

stub = StubLLM([
    ("tools", [("get_weather", '{"city": "北京"}')]),
    ("tools", [("get_weather", '{"city": "北京"}')]),          # A 第2次失败 → 熔断
    ("tools", [("basic_calculator", '{"operation": "add", "a": 1, "b": 2}')]),  # B 正常
    ("tools", [("get_weather", '{"city": "北京"}')]),          # A 被熔断拦截
    ("tools", [("basic_calculator", '{"operation": "add", "a": 1, "b": 2}')]),  # B 正常
    ("answer", "done"),
])
a = make_agent(stub, {
    "get_weather": mk("get_weather", failA, WeatherArgs),
    "basic_calculator": mk("basic_calculator", okB, CalcArgs),
})
a.chat("测试")
tool_msgs = [m["content"] for m in stub.calls[-1] if m.get("role") == "tool"]
check("工具 A 熔断被拦截", tool_msgs[3].startswith("Error(CIRCUIT_OPEN)"))
check("工具 B 不受影响", counterB["n"] == 2 and "B 正常" in tool_msgs[4])

print("\n全部通过 ✅")
