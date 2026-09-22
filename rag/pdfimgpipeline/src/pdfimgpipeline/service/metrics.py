"""极简 Prometheus 指标：计数器 + 延迟累加。事件循环单线程，无需加锁。"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict


class Metrics:
    def __init__(self) -> None:
        self._counters: Dict[str, float] = defaultdict(float)
        self._lat_sum: Dict[str, float] = defaultdict(float)
        self._lat_count: Dict[str, int] = defaultdict(int)

    def inc(self, name: str, n: float = 1.0) -> None:
        self._counters[name] += n

    def observe(self, name: str, seconds: float) -> None:
        self._lat_sum[name] += seconds
        self._lat_count[name] += 1

    def render(self) -> str:
        lines = []
        for name in sorted(self._counters):
            lines.append(f"# TYPE pdfimg_{name}_total counter")
            lines.append(f"pdfimg_{name}_total {self._counters[name]}")
        for name in sorted(self._lat_sum):
            count = self._lat_count[name]
            avg = self._lat_sum[name] / count if count else 0.0
            lines.append(f"# TYPE pdfimg_{name}_seconds gauge")
            lines.append(f"pdfimg_{name}_seconds {round(avg, 4)}")
        lines.append("")
        return "\n".join(lines)


METRICS = Metrics()
