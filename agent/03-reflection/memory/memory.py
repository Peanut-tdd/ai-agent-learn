from enum import StrEnum


class Memory:
    """会话级记忆，由 AgentLoop 在 __init__ 创建一次，跨多轮对话持久。

    两级结构：
      - records:         当前任务的反思轨迹（start_task 时清空，单任务内重试用）；
      - session_lessons: 跨任务的策略教训（end_task 时从本轮反思中抽取一条沉淀，
                          有界保留，供后续任务参考——显式决策，避免把每个任务的噪音带进会话）。
    """

    class RecordType(StrEnum):
        Execute = "execute"
        Reflect = "reflect"

    def __init__(self, max_session_lessons: int = 3) -> None:
        # 记录格式：{"record_type": "execute" | "reflect", "content": str}
        self.records: list[dict] = []
        # 跨任务教训（历史任务的"最终修改指引"），默认最多保留 3 条
        self.session_lessons: list[str] = []
        self.max_session_lessons = max_session_lessons

    # ---------- 任务作用域 ----------

    def start_task(self) -> None:
        """开始一个新任务：清空单任务反思轨迹（会话教训保留）。"""
        self.records = []

    def end_task(self) -> None:
        """任务结束：把本轮最后一条修改指引沉淀为会话级教训。

        只沉淀"最终指引"一条——任务第一轮就通过（无反思记录）说明无需修正，
        不留教训，避免每个任务的噪音都进入会话。
        """
        lessons = [r["content"] for r in self.records if r["record_type"] == "reflect"]
        if not lessons:
            return
        self.session_lessons.append(lessons[-1])
        self.session_lessons = self.session_lessons[-self.max_session_lessons:]

    # ---------- 记录 ----------

    def add_record(self, record_type: "Memory.RecordType", content: str):
        if not isinstance(record_type, self.RecordType):
            raise ValueError("record type must be an Memory.RecordType")
        self.records.append({"record_type": record_type.value, "content": content})
        print(f"\n\n新增一条{record_type}记录")

    def get_reflections(self) -> list[str]:
        """返回当前任务的所有评审/修正反馈（下一轮 Execute 作为策略记忆）。"""
        return [r["content"] for r in self.records if r["record_type"] == "reflect"]

    def get_session_lessons(self) -> list[str]:
        """返回跨任务沉淀的策略教训（供当前任务参考）。"""
        return list(self.session_lessons)

    # ---------- 调试 ----------

    def get_trajectory(self) -> str:
        """拼接整条轨迹，用于调试打印。"""
        parts = []
        for r in self.records:
            if r["record_type"] == "execute":
                parts.append(f"--- 上一轮尝试 ---\n{r['content']}")
            elif r["record_type"] == "reflect":
                parts.append(f"--- 评审员反馈 ---\n{r['content']}")
        return "\n\n".join(parts)

    def get_last_execution(self) -> str | None:
        for r in reversed(self.records):
            if r["record_type"] == "execute":
                return r["content"]
        return None
