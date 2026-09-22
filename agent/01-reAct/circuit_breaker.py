"""熔断器：跨对话保护下游（DB/外部 API）。

核心思想：下游持续性故障时快速失败（fail fast），给下游恢复时间，而不是反复重试。
与 per-chat 的死循环检测互补——死循环管"模型重复调用"，熔断管"下游真的挂了"。

状态机：
    CLOSED ──连续失败 ≥ N 次──▶ OPEN ──冷却 T 秒──▶ HALF_OPEN ──探测成功──▶ CLOSED
      ▲                          │                   │──探测失败──▶ OPEN
      └──────────────────────────┘

线程安全：内部持有锁。
"""
import threading
import time
from enum import Enum


class BreakerState(str, Enum):
    CLOSED = "closed"        # 正常：全部放行
    OPEN = "open"            # 熔断中：快速失败
    HALF_OPEN = "half_open"  # 半开：放行有限探测请求，验证下游是否恢复


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, cooldown_seconds: float = 30.0,
                 probe_max: int = 1):
        self.failure_threshold = failure_threshold   # 连续失败 N 次触发熔断
        self.cooldown_seconds = cooldown_seconds     # 熔断持续时间，之后进半开
        self.probe_max = probe_max                   # 半开阶段的探测请求数
        self._lock = threading.Lock()
        self._state = BreakerState.CLOSED
        self._consecutive_failures = 0
        self._opened_at = 0.0
        self._probe_remaining = probe_max

    @property
    def state(self) -> BreakerState:
        with self._lock:
            return self._state

    def is_open(self) -> bool:
        """是否应拒绝调用（True = 熔断中，调用方应快速失败）。"""
        with self._lock:
            if self._state == BreakerState.CLOSED:
                return False
            if self._state == BreakerState.OPEN:
                # 冷却结束 → 进入半开，放行探测请求
                if time.monotonic() - self._opened_at >= self.cooldown_seconds:
                    self._state = BreakerState.HALF_OPEN
                    self._probe_remaining = self.probe_max
                else:
                    return True
            # HALF_OPEN：只放行有限的探测请求
            if self._probe_remaining > 0:
                self._probe_remaining -= 1
                return False
            return True

    def record_success(self):
        """调用成功（含半开探测成功）→ 关闭熔断。"""
        with self._lock:
            self._state = BreakerState.CLOSED
            self._consecutive_failures = 0

    def record_failure(self):
        """调用失败（仅下游故障类，业务拒绝不计入）→ 累计或重新熔断。"""
        with self._lock:
            if self._state == BreakerState.HALF_OPEN:
                # 探测失败 → 重新熔断，重启冷却
                self._state = BreakerState.OPEN
                self._opened_at = time.monotonic()
                self._probe_remaining = 0
                return
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.failure_threshold:
                self._state = BreakerState.OPEN
                self._opened_at = time.monotonic()
