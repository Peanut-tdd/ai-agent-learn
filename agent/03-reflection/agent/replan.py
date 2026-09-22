"""Replan 步骤 — Reflection Agent Loop 的「修正」阶段（纯文本，不使用 function calling）。

Execute 产出候选答案 -> Reflect 评审未通过时，Replan 把评审反馈转成
下一轮 Execute 的具体修改指引，存入 Memory 作为策略记忆，驱动重试改进。

修正同样是纯分析任务，不涉及外部数据，因此不调用任何工具（think 传 tools=None）。
"""

import threading

from agent.prompt import build_replan_prompt
from llm.llm_deepseek import LlmApiClient

_lock = threading.Lock()


class Replan:

    _instance = None

    def __init__(self, llmclient: LlmApiClient):
        self.llmclient = llmclient

    @classmethod
    def instance(cls, llmclient: LlmApiClient):
        if cls._instance is None:
            with _lock:
                if cls._instance is None:
                    cls._instance = cls(llmclient)
        return cls._instance

    def run(self, task: str, output: str, feedback: str, history: str = "") -> str:
        """把评审反馈转成下一轮 Execute 的修改指引（无工具，纯文本）。

        Args:
            task:     当前用户任务
            output:   上一轮候选答案
            feedback: 评审反馈
            history:  会话上下文（历史+滚动摘要），用于理解任务引用/前后一致性

        生成失败时退回原始 feedback，保证修正链路不断。
        """
        print(f"\n{'='*10} Replan Start {'='*10}")
        messages = [
            {"role": "system", "content": "你是修正规划器，为下一轮执行生成具体修改指引，会结合会话历史理解上下文。"},
            {"role": "user", "content": build_replan_prompt(task, output, feedback, history)},
        ]

        msg = self.llmclient.think(messages, tools=None)
        hint = (msg.content or "").strip()
        print(f"\n--- Replan End---\n{hint}")
        return hint or feedback
