import json
import threading
import time
from enum import Enum
from types import SimpleNamespace

import requests
from agent.prompt import system_prompt
from llm.llmapi import llmapi
from openai.types.chat import ChatCompletionMessageToolCall
from pydantic import ValidationError

from tools import TOOL_FUNCTIONS
from tools.tool_result import SideEffect, ToolResult, ToolSpec
from circuit_breaker import CircuitBreaker

_lock = threading.Lock()

# ── 可调参数 ──────────────────────────────────────────────────────────────────
MAX_TURNS = 10                 # 单次 chat() 的兜底硬上限（正常推理轮 + 修复轮）
MAX_CONSECUTIVE_FAILURES = 2   # 同工具+同参数连续失败 N 次 → 判定死循环
HISTORY_WINDOW_TURNS = 10      # 跨轮持久化滑动窗口（user/assistant 对数）
MAX_LLM_RETRIES = 2            # 单次 chat() 内模型调用失败/超时的重试次数
MAX_TOTAL_SECONDS = 180        # 单次 chat() 的总时间预算（秒）
IDEMPOTENT_MAX_RETRIES = 3     # 幂等写失败时的代码层自动重试次数（不含首次）
NON_IDEMPOTENT_MAX_RETRIES = 2  # 非幂等写：仅对"确定未执行"的错误自动重试的次数
RETRY_BACKOFF_SECONDS = 1      # 指数退避基数：1s / 2s / 4s
CIRCUIT_FAILURE_THRESHOLD = 5  # 同工具连续失败 N 次触发熔断（跨对话累计）
CIRCUIT_COOLDOWN_SECONDS = 30  # 熔断持续时间（秒），之后进入半开探测

STRATEGY_HINT = (
    "提示：你已连续 {n} 次用完全相同的参数调用同一工具且均失败，"
    "请放弃当前思路，换一种完全不同的策略。"
)
FALLBACK_ANSWER = "抱歉，我尝试了多种方式仍未解决你的问题，请换个问法或稍后再试。"
LLM_RETRY_HINT = "提示：上一次模型响应超时或出错。请根据当前对话历史继续作答，必要时重新调用工具。"


def _normalize_args(args: dict) -> str:
    """规范化参数：按 key 排序后 JSON 编码，用于精确比较"同参数"。"""
    return json.dumps(args, sort_keys=True, ensure_ascii=False)


class RetryClass(str, Enum):
    """错误按状态确定性分类——决定能否安全重试。

    SAFE     请求确定未执行（连接失败/DNS/ConnectTimeout/429/503）：重试安全。
    UNKNOWN  结果未知（读超时/连接重置/504）：请求可能已执行，非幂等绝不能重试。
    NO       确定失败或业务拒绝（4xx/业务异常）：重试无意义。
    """

    SAFE = "safe"
    UNKNOWN = "unknown"
    NO = "no"


def _classify_error(e: Exception) -> RetryClass:
    """按异常类型/HTTP 状态码判定状态确定性（显式匹配 + 名字启发式兜底）。"""
    # 1. 带响应状态码的错误（如 requests.HTTPError）
    resp = getattr(e, "response", None)
    if resp is not None:
        status = resp.status_code
        if status in (429, 503):
            return RetryClass.SAFE       # 明确拒绝，请求未执行
        if status == 504:
            return RetryClass.UNKNOWN    # 网关超时，上游可能已处理
        if status >= 500:
            return RetryClass.SAFE       # 5xx 服务端错误，一般未执行
        return RetryClass.NO             # 4xx 请求本身有问题

    # 2. 超时类（先子类后父类）
    if isinstance(e, requests.exceptions.ConnectTimeout):
        return RetryClass.SAFE           # 连接阶段超时，连接未建立
    if isinstance(e, requests.exceptions.ReadTimeout):
        return RetryClass.UNKNOWN        # 读超时，请求已发出
    if isinstance(e, (TimeoutError, requests.exceptions.Timeout)):
        return RetryClass.UNKNOWN        # 无法区分阶段，保守按未知

    # 3. 连接类（先子类后父类：ConnectionResetError/BrokenPipeError 继承自内置 ConnectionError）
    if isinstance(e, (ConnectionResetError, BrokenPipeError, requests.exceptions.ChunkedEncodingError)):
        return RetryClass.UNKNOWN        # 连接中途断开，可能已执行
    if isinstance(e, (requests.exceptions.ConnectionError, ConnectionError)):
        return RetryClass.SAFE           # 连接失败/DNS，请求未送达

    # 4. 名字启发式兜底（timeout 必须先于 connection：APITimeoutError 名字同时含两者）
    name = type(e).__name__.lower()
    if "timeout" in name or "timed out" in str(e).lower():
        return RetryClass.UNKNOWN
    if "connection" in name or "network" in name or "dns" in name:
        return RetryClass.SAFE
    if "reset" in name or "broken" in name or "chunked" in name:
        return RetryClass.UNKNOWN
    return RetryClass.NO


class agent:

    _agent_client = None

    def __init__(self):
        self.llmclient: llmapi = llmapi.instance()
        # 跨 chat() 调用的持久化历史：只保存 {user, assistant} 最终回答对
        self.history_msg: list[dict] = []
        # 非幂等写失败后（结果未知）永久拦截的工具名，本 agent 生命周期内有效
        self._blocked_writes: set[str] = set()
        # 熔断器注册表：按工具名粒度，跨 chat() 累计下游故障
        self._breakers: dict[str, CircuitBreaker] = {}
        self._breakers_lock = threading.Lock()

    def _get_breaker(self, name: str) -> CircuitBreaker:
        """懒创建熔断器（按工具名）。"""
        with self._breakers_lock:
            breaker = self._breakers.get(name)
            if breaker is None:
                breaker = CircuitBreaker(
                    failure_threshold=CIRCUIT_FAILURE_THRESHOLD,
                    cooldown_seconds=CIRCUIT_COOLDOWN_SECONDS,
                )
                self._breakers[name] = breaker
            return breaker

    def _record_breaker(self, tool_name: str, result: ToolResult):
        """按调用结果回填熔断器：只统计下游故障类失败（SAFE/UNKNOWN）。

        NO（业务拒绝）与校验/拦截类错误与服务无关，不计入熔断。
        """
        if result.success:
            self._get_breaker(tool_name).record_success()
            return
        if result.code != "EXECUTION_ERROR" or result.error is None:
            return  # VALIDATION_ERROR/INVALID_JSON/BLOCKED_WRITE/UNKNOWN_TOOL/CIRCUIT_OPEN
        cls = result.retry_class or _classify_error(result.error)
        if cls in (RetryClass.SAFE, RetryClass.UNKNOWN):
            self._get_breaker(tool_name).record_failure()

    @classmethod
    def instance(cls):
        if not cls._agent_client:
            with _lock:
                if not cls._agent_client:
                    cls._agent_client = cls()
        return cls._agent_client

    def _execute_tool(self, tool_call: ChatCompletionMessageToolCall) -> ToolResult:
        """三段式执行：解析 JSON → Pydantic 契约校验 → 按副作用类型分派执行。

        任何一步失败都返回结构化的 ToolResult，不向调用方抛异常。
        """
        tool_name = tool_call.function.name
        args_text = tool_call.function.arguments or "{}"

        spec = TOOL_FUNCTIONS.get(tool_name)
        if not spec:
            return ToolResult(
                success=False, code="UNKNOWN_TOOL",
                error=ValueError(f"未注册的工具 {tool_name}"),
            )

        # 熔断检查：下游持续故障时快速失败，不发起真实调用
        if self._get_breaker(tool_name).is_open():
            return ToolResult(
                success=False, code="CIRCUIT_OPEN",
                error=RuntimeError(
                    f"{tool_name} 持续失败已触发熔断，暂时不可用。"
                    "请稍后再试、改用其他工具，或告知用户服务暂时不可用。"
                ),
            )

        # 非幂等写拦截：结果未知的操作绝不允许重复执行
        if spec.side_effect == SideEffect.NON_IDEMPOTENT and tool_name in self._blocked_writes:
            return ToolResult(
                success=False, code="BLOCKED_WRITE",
                error=ValueError(
                    f"{tool_name} 之前的调用结果未知（可能已执行），禁止重复调用。"
                    "请使用查询类工具（如 query_payment）确认操作状态。"
                ),
            )

        try:
            raw_args = json.loads(args_text)
        except json.JSONDecodeError as e:
            return ToolResult(success=False, code="INVALID_JSON", error=e)

        try:
            validated = spec.args_model.model_validate(raw_args)
        except ValidationError as e:
            return ToolResult(success=False, code="VALIDATION_ERROR", error=e)

        # 幂等写：SAFE/UNKNOWN 都自动重试（幂等键保证同参重试安全）
        if spec.side_effect == SideEffect.IDEMPOTENT:
            return self._execute_with_retry(
                spec, validated,
                retryable={RetryClass.SAFE, RetryClass.UNKNOWN},
                max_retries=IDEMPOTENT_MAX_RETRIES,
            )

        # 非幂等写：按状态确定性分派——UNKNOWN 立即拦截，SAFE 代码层重试，NO 返回
        if spec.side_effect == SideEffect.NON_IDEMPOTENT:
            try:
                data = spec.handler(**validated.model_dump())
                return ToolResult(success=True, data=data)
            except Exception as e:
                cls = _classify_error(e)
                if cls == RetryClass.UNKNOWN:
                    self._blocked_writes.add(spec.name)  # 结果未知 → 永久拦截
                    # 首次失败保留真实异常给模型；之后所有调用都会被 BLOCKED_WRITE 拦截
                    return ToolResult(
                        success=False, code="EXECUTION_ERROR", error=e, retry_class=cls,
                    )
                if cls == RetryClass.SAFE:
                    # 确定未执行 → 代码层自动重试（同参数安全）
                    return self._execute_with_retry(
                        spec, validated,
                        retryable={RetryClass.SAFE},
                        max_retries=NON_IDEMPOTENT_MAX_RETRIES,
                    )
                # NO：请求未执行，返回错误，模型可换参数
                return ToolResult(
                    success=False, code="EXECUTION_ERROR", error=e, retry_class=cls,
                )

        # 只读工具：失败直接返回，由模型决定重试
        try:
            data = spec.handler(**validated.model_dump())
            return ToolResult(success=True, data=data)
        except Exception as e:
            return ToolResult(success=False, code="EXECUTION_ERROR", error=e)

    def _execute_with_retry(
        self,
        spec: ToolSpec,
        validated,
        retryable: set[RetryClass],
        max_retries: int,
    ) -> ToolResult:
        """同参数自动重试（指数退避），只重试状态允许的错误类别。

        重试耗尽后按最后一次错误的确定性，如实告知模型：
        - SAFE：请求确定未执行，可换参数重试
        - UNKNOWN：结果未知，必须查询确认
        """
        last_error = None
        last_class = RetryClass.NO
        attempts = 0  # 实际执行次数（含首次）
        for attempt in range(max_retries + 1):  # 首次 + N 次重试
            attempts += 1
            try:
                data = spec.handler(**validated.model_dump())
                return ToolResult(success=True, data=data)
            except Exception as e:
                last_error = e
                last_class = _classify_error(e)
                if attempt < max_retries and last_class in retryable:
                    time.sleep(RETRY_BACKOFF_SECONDS * (2 ** attempt))
                    continue
                break  # 状态不允许重试，或重试次数耗尽

        if last_class == RetryClass.SAFE:
            msg = (
                f"{spec.name} 执行 {attempts} 次（含自动重试 {attempts - 1} 次）仍失败"
                f"（{last_error}）。请求确定未执行，可换参数重试。"
            )
        elif last_class == RetryClass.UNKNOWN:
            msg = (
                f"{spec.name} 执行 {attempts} 次（含自动重试 {attempts - 1} 次）仍失败"
                f"（{last_error}）。操作结果未知，请使用查询工具确认状态。"
            )
        else:
            msg = (
                f"{spec.name} 执行 {attempts} 次（含自动重试 {attempts - 1} 次）仍失败"
                f"（{last_error}）。请求未执行，可换参数重试。"
            )
        return ToolResult(
            success=False, code="EXECUTION_ERROR", error=RuntimeError(msg), retry_class=last_class,
        )

    def _remember(self, role: str, content: str):
        """持久化一条消息，并按 turn 边界整对裁剪到窗口内。"""
        self.history_msg.append({"role": role, "content": content})
        while len(self.history_msg) > HISTORY_WINDOW_TURNS * 2:
            self.history_msg.pop(0)  # 弹出最早的 user
            self.history_msg.pop(0)  # 及其对应的 assistant

        print(f'持久化消息：{self.history_msg}')

    def _fallback(self, query: str, content: str = FALLBACK_ANSWER):
        """兜底：持久化本轮问答并返回最终回答对象（不抛异常）。"""
        self._remember("user", query)
        self._remember("assistant", content)
        return SimpleNamespace(content=content, tool_calls=None)

    def chat(self, query):
        messages: list[dict] = [{"role": "system", "content": system_prompt}]
        messages.extend(self.history_msg)
        messages.append({"role": "user", "content": query})

        # 死循环检测状态：仅本轮有效，跨 chat() 不累计
        consecutive_failures = 0
        last_failed_sig = None
        hint_injected = False
        llm_retries = 0
        deadline = time.monotonic() + MAX_TOTAL_SECONDS

        for i in range(MAX_TURNS):
            # 总时间预算检查：超时直接兜底，避免最长 10 轮的干等
            if time.monotonic() > deadline:
                return self._fallback(query)

            print(f"\n{'=' * 10}循环{i + 1}次{'=' * 10}")
            try:
                tool_msg = self.llmclient.generate(messages)
            except Exception as e:
                print(f"模型调用出错：{e}")
                if llm_retries < MAX_LLM_RETRIES:
                    # 注入提示让模型继续，而不是整轮崩溃
                    llm_retries += 1
                    messages.append({"role": "user", "content": LLM_RETRY_HINT})
                    continue
                return self._fallback(query)

            # 模型调用可能耗时（最长 LLM_TIMEOUT），返回后立即复查时间预算
            if time.monotonic() > deadline:
                if not tool_msg.tool_calls:
                    # 已有最终回答就直接交付，不兜底
                    self._remember("user", query)
                    self._remember("assistant", tool_msg.content or "")
                    return tool_msg
                return self._fallback(query)

            if not tool_msg.tool_calls:
                # 模型给出最终回答：持久化本轮的 user/assistant 对
                self._remember("user", query)
                self._remember("assistant", tool_msg.content or "")
                return tool_msg

            # 把 assistant 的 tool_calls 按 API 要求格式加入本轮历史
            assistant_tool_calls: list = []
            for tc in tool_msg.tool_calls:
                assistant_tool_calls.append(
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                )
            messages.append(
                {
                    "role": "assistant",
                    "content": tool_msg.content,
                    "tool_calls": assistant_tool_calls,
                }
            )

            # 逐个执行工具，把结果以 role=tool 回传模型，并做死循环检测
            for tc in tool_msg.tool_calls:
                if time.monotonic() > deadline:
                    return self._fallback(query)

                result = self._execute_tool(tc)
                self._record_breaker(tc.function.name, result)
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": str(result)}
                )

                if not result.success:
                    try:
                        sig = (
                            tc.function.name,
                            _normalize_args(json.loads(tc.function.arguments or "{}")),
                        )
                    except json.JSONDecodeError:
                        sig = (tc.function.name, "<malformed-args>")

                    if sig == last_failed_sig:
                        consecutive_failures += 1
                    else:
                        consecutive_failures = 1
                        last_failed_sig = sig
                else:
                    consecutive_failures = 0
                    last_failed_sig = None

            # 换策略提示必须等所有 role=tool 结果都追加完后再注入：
            # OpenAI 要求 tool 消息紧跟 assistant(tool_calls) 之后，
            # 中途插入 user 消息会被严格实现返回 400。
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES and not hint_injected:
                hint_injected = True
                messages.append(
                    {
                        "role": "user",
                        "content": STRATEGY_HINT.format(n=consecutive_failures),
                    }
                )

        # 硬上限兜底：返回最终回答而不是抛异常
        return self._fallback(query)
