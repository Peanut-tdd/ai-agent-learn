"""Reflection 步骤 — Reflection Agent Loop 的「评审」阶段（纯文本评审，不使用 function calling）。

对 Execute 产出的候选答案做质量评审，返回结构化结论：
  - passed=True  -> 答案达标，循环可以结束；
  - passed=False -> 给出可操作的修改意见（feedback），
                    由 AgentLoop 存入 Memory，作为下一轮 Execute 的"策略记忆"重新生成。

评审是纯分析任务，不涉及外部数据，因此不调用任何工具（think 传 tools=None）。
"""

import json
import re
import threading
from dataclasses import dataclass

from agent.prompt import build_reflection_prompt
from llm.llm_deepseek import LlmApiClient

_lock = threading.Lock()


@dataclass
class ReflectionResult:
    """评审结论：passed=是否通过；feedback=评审意见（传给下一轮 Execute）。"""

    passed: bool
    feedback: str




class Reflect:

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


    def _rule_check(self, task: str, output: str) -> ReflectionResult:
        """
            可验证任务的规则评估（示例：21+21 且统计答案字符串字符数）。
            命中规则时返回确定结果，否则返回 None 交给模型评估。

            注意：多轮对话中任务常被改写（如"现在算一下两个 21 的和"），
            字面规则不命中属预期，会自然回退到 LLM 结合会话历史评审。
        """
        if "21+21" not in task.replace(" ", "") and "21 + 21" not in task:
            return None

        has_sum = "42" in output
        char_ok = bool(
            re.search(r"字符数\s*[:：]\s*2\b", output)
            or re.search(r"\b2\s*个字符", output)
            or re.search(r"字符数.{0,20}2", output)
        )
        if has_sum and char_ok:
            return ReflectionResult(
                passed=True,
                feedback="规则校验通过：包含 42 且明确给出答案字符串字符数为 2。",
            )
        missing = []
        if not has_sum:
            missing.append("未给出正确计算结果 42")
        if not char_ok:
            missing.append("未用 word_count 统计答案字符串「42」的字符数（应为 2）")
        return ReflectionResult(
            passed=False,
            feedback="规则校验未通过：" + "；".join(missing),
        )
    

    def run(self, task: str, output: str, history: str = "", use_rules: bool = True) -> ReflectionResult:
        """评审 Execute 的候选答案，返回结构化结论。

        Args:
            task:    当前用户任务
            output:  候选答案
            history: 会话上下文（历史+滚动摘要），多轮任务可能引用前文，评审需结合上下文

        模型需返回 JSON {"passed": bool, "feedback": str}；
        解析失败时保守按"未通过"处理并保留原文作为反馈（宁可重试，不误判完成）。
        """

        """Evaluation：规则优先，否则由 LLM 给出成功标志、分数与批评。"""
        if use_rules:
            ruled = self._rule_check(task, output)
            if ruled is not None:
                print(f"Rule check passed: {ruled.passed}, feedback: {ruled.feedback}")
                return ruled

        messages = [
            {"role": "system", "content": "你是一个严格、客观的评审智能体，评审时会结合会话历史理解任务上下文。"},
            {"role": "user", "content": build_reflection_prompt(task, output, history)},
        ]

        print(f"\n{'=' * 10} Reflection {'=' * 10}")

        # 纯文本评审：tools=None 关闭 function calling
        msg = self.llmclient.think(messages, tools=None)
        content = (msg.content or "").strip()
        print(f"\n--- Reflection ---\n{content}")

        passed, feedback = self._parse(content)
        print(f"Reflection verdict: passed={passed}")
        return ReflectionResult(passed=passed, feedback=feedback)

    @staticmethod
    def _parse(content: str) -> tuple[bool, str]:
        """从模型输出解析 (passed, feedback)，容忍 markdown 代码块等多余内容。"""
        cleaned = content.strip()
        # 去掉可能的 ```json ... ``` 围栏
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`").strip()
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()

        # 优先整体解析；失败则退而取第一个 {...} 对象
        data = None
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if m:
                try:
                    data = json.loads(m.group(0))
                except json.JSONDecodeError:
                    pass

        if not isinstance(data, dict):
            # 解析失败：保守判未通过，原文即反馈
            return False, content

        passed = bool(data.get("passed", False))
        feedback = data.get("feedback") or ""
        return passed, str(feedback).strip() or content
