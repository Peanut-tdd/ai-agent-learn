from llm.llm_deepseek import LlmApiClient
import os
from agent.execute import Execute
from agent.reflect import Reflect
from agent.replan import Replan
from memory.memory import Memory

class AgentLoop:

    def __init__(self,llmClient:LlmApiClient) -> None:
        self.llmclient=llmClient
        self.execute_agent=Execute.instance(llmClient)
        self.reflection_agent=Reflect.instance(llmClient)
        self.replan_agent=Replan.instance(llmClient)
        self.max_trials=int(os.environ.get("max_trials",3))       #最大尝试次数
        self.debug=os.environ.get("debug")
        # 会话级历史：记录 (task, answer)，跨多轮对话持久
        self.history_messages:list[tuple[str,str]]=[]
        # 旧轮次的滚动摘要（超预算部分压缩成摘要，避免上下文爆炸）
        self._summary:str=""
        self._summary_until:int=0   # 已并入摘要的轮数（history_messages 前缀索引）
        self.max_memory_rounds=int(os.environ.get("max_memory_rounds",20)) # 近期保留的最大轮数
        self.max_history_chars=int(os.environ.get("max_history_chars",4000)) # 会话上下文字符预算
        # 会话级记忆（跨任务沉淀教训），由本实例持有，不被单次 run 重建
        self.memory=Memory(int(os.environ.get("max_session_lessons",3)))

    # ------------------------------------------------------------------
    # 会话上下文：近期轮次按预算保留 + 更早的轮次做滚动摘要
    # ------------------------------------------------------------------

    def _session_context(self) -> str:
        """生成当前任务的会话上下文（摘要 + 近期轮次），带字符/轮数双重预算。

        - 近期轮次：从最新一轮往前取，直到字符预算（budget*0.7）或轮数上限；
        - 更早的轮次：增量滚动摘要（每次有新轮次溢出时才调用一次 LLM 压缩），
          摘要本身也受预算约束（budget*0.3）。
        """
        turns=self.history_messages
        if not turns:
            return "（无历史）"

        recent_budget=int(self.max_history_chars*0.7)
        recent:list[tuple[str,str]]=[]
        used=0
        for t in reversed(turns):
            cost=len(t[0])+len(t[1])+20   # 每轮固定开销（标签、换行等）
            # 至少保留最新一轮；之后受预算/轮数双重约束
            if recent and (used+cost>recent_budget or len(recent)>=self.max_memory_rounds):
                break
            recent.append(t)
            used+=cost
        recent=list(reversed(recent))

        overflow_start=len(turns)-len(recent)
        # 有新轮次溢出 -> 并入滚动摘要（只发生一次，不重复压缩旧内容）
        if overflow_start>self._summary_until:
            new_turns=turns[self._summary_until:overflow_start]
            self._summary=self._summarize(self._summary,new_turns)
            self._summary_until=overflow_start

        parts=[]
        if self._summary:
            parts.append(f"早期对话摘要：\n{self._summary}")
        if recent:
            parts.append("近期对话：\n"+ "\n".join(
                f"Task: {task}\nAnswer: {answer}" for task,answer in recent))
        return "\n\n".join(parts)

    def _summarize(self, old_summary:str, new_turns:list[tuple[str,str]]) -> str:
        """把已有摘要 + 新增轮次合并压缩成新摘要；失败时退回截断文本，不中断会话。"""
        blocks=[]
        if old_summary:
            blocks.append(f"已有摘要：\n{old_summary}")
        blocks.append("新增对话：\n"+ "\n".join(
            f"Task: {task}\nAnswer: {answer}" for task,answer in new_turns))
        text="\n\n".join(blocks)

        summary_budget=int(self.max_history_chars*0.3)
        if len(text)>summary_budget:
            text=text[:summary_budget]+"…"

        prompt=(
            "你是会话记忆整理器。下面是一段多轮对话（用户任务/助手答案），"
            "其中可能包含一条已有的会话摘要。\n"
            "请把已有摘要与新增对话合并压缩成一份新的会话摘要（不超过 150 字），"
            "保留：每轮任务要点、关键结果、对后续对话有用的信息（用户偏好、未完成事项、结论）。"
            "直接输出摘要文本，不要其他任何内容。\n\n"
            f"{text}"
        )
        try:
            msg=self.llmclient.think(
                [{"role":"system","content":"你是会话记忆整理器。"},
                 {"role":"user","content":prompt}],
                tools=None,
            )
            return (msg.content or "").strip() or "（摘要生成失败）"
        except Exception as e:
            print(f"!!! 会话摘要生成失败，改用截断文本：{e}")
            return text[:summary_budget]

    # ------------------------------------------------------------------
    # 单任务执行
    # ------------------------------------------------------------------

    def run(self,task:str) -> str:
        """
            reflection 循环：execute（执行）-> reflect（反思）-> replan（修正重试）

            每轮：Execute 产出候选答案 -> Reflect 评审；
            通过则结束返回；未通过则把反馈修正为修改指引存入 Memory，
            下一轮 Execute 带着全部指引（策略记忆）重新生成。

            多轮会话特性：
            - 会话上下文（历史+滚动摘要）贯穿 Execute/Reflect/Replan 三环；
            - 本轮失败教训在任务结束时沉淀为会话级教训，供后续任务参考；
            - 单轮异常会重试并在下一轮携带异常信息，不会静默丢失任务。
        """
        self.memory.start_task()
        session_ctx=self._session_context()
        last_output=""
        last_error:str|None=None

        for trial in range (1,self.max_trials+1):

            if self.debug:
                print(f"\n========== Trial {trial}/{self.max_trials} ==========")
                trajectory=self.memory.get_trajectory()
                if trajectory:
                    print(f"Strategy memory:\n{trajectory}")

            try:
                # 1) Execute：带着会话上下文 + 历史教训 + 本轮评审/修正指引生成候选答案
                reflections=self.memory.get_reflections()
                session_lessons=self.memory.get_session_lessons()
                last_output=self.execute_agent.run(task,reflections,session_lessons,session_ctx)
                self.memory.add_record(self.memory.RecordType.Execute,last_output)

                if self.debug:
                    print(f"\n--- Evaluation ---")
                    print(f"Output: {last_output}")

                # 2) Reflect：结合会话历史评审候选答案
                result=self.reflection_agent.run(task,last_output,session_ctx)

                if result.passed:
                    print(f"\n>>> 第 {trial} 轮通过评审，循环结束")
                    self.history_messages.append((task,last_output))
                    self.memory.end_task()
                    return last_output

                # 3) Replan：未通过 -> 结合会话历史把反馈修正为修改指引，存入策略记忆
                hint=self.replan_agent.run(task,last_output,result.feedback,session_ctx)
                self.memory.add_record(self.memory.RecordType.Reflect,hint)

            except Exception as e:
                last_error=str(e)
                print(f"\n!!! 第 {trial} 轮执行异常：{e}")
                if trial<self.max_trials:
                    # 非最后一轮：把异常信息作为指引存入记忆，下一轮重试
                    print("    将携带异常信息进入下一轮重试")
                    self.memory.add_record(
                        self.memory.RecordType.Reflect,
                        f"上一轮执行异常（系统级）：{e}，请重试并避免再次触发。",
                    )
                    continue
                # 最后一轮也失败：如实返回错误并记入历史，保证会话不断裂
                msg=f"（执行失败：{last_error}）"
                self.history_messages.append((task,msg))
                self.memory.end_task()
                return msg

        else:
            # 达到最大轮数仍未通过：返回最后一轮答案，如实告知未达标
            self.history_messages.append((task,last_output))
            self.memory.end_task()
            print(f"\n>>> 达到最大尝试次数 {self.max_trials}，未通过评审，返回最后答案")
            return last_output
