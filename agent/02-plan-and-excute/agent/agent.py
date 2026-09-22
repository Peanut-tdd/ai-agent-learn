from dotenv import load_dotenv
from plan.plan import planner
from execute.execute import execute
import os
from  tools.types  import Tool
from llm.llm_deepseek import LlmClient

class agent:

    def __init__(self,llm:LlmClient,tools:dict[str:Tool]) -> None:
        
        load_dotenv()
        self.max_plans:int=int(os.environ.get("max_plans", "5"))
        self.debug=os.environ.get("debug", "false").lower() == "true"
        self.llmclient=llm
        self.tools=tools

        self.planner=planner(llm)
        self.executer=execute(llm)

        # 多轮记忆：保存每一轮 (用户问题, 最终回答)，最多保留最近 20 轮
        self.memory: list[tuple[str, str]] = []
        self.max_memory_rounds = 20

    def _memory_str(self) -> str:
        """把最近的对话历史拼成提示词片段"""
        recent = self.memory[-self.max_memory_rounds:]
        return "\n".join(f"用户: {q}\n助手: {a}" for q, a in recent)
    





    def plan_and_execute(self,task):

        # 带上历史记忆规划（第一轮为空）
        history = self._memory_str()
        steps = self.planner.plan(task, history)
        if self.debug:
            print("\n=== Plan ===")
            for i, st in enumerate(steps, 1):
                print(f"  {i}. {st}")
            print()

        state:dict[str,any]={}
        for attempt in range(self.max_plans+1):
            for i,st in enumerate(steps):
                if self.debug:
                    print(f"\n-- Excuting step {i+1}:{st} --")

              #  try:
                    out=self.executer.execute_step(st,state,self.tools,history)
                    state[f"step_{i}"]=out
                    if self.debug:
                        print(f"Result:{out}")
                """ except Exception as e:
                    if self.debug:
                       print(f"Step failed: {e}. Replanning...\n")

                    replan_prompt=(
                        f"Task: {task}\n"
                        f"Failed step: {st}\n"
                        f"Error: {e}\n"
                        "Give a new plan."
                    )

                    text=self.llmclient.think(replan_prompt)
                    steps = self._parse_steps(text)
                    if self.debug:
                        print("\n=== Replanned ===")
                        for j, new_st in enumerate(steps, 1):
                            print(f"  {j}. {new_st}")
                        print()
                    break """
            else:
                # 整轮对话完成，写入记忆，供下一轮使用
                self.memory.append((task, out))
                return out
                

        return "Failed after replanning."





    def _parse_steps(self,text: str) -> list[str]:
        return [s.strip("- ").strip() for s in text.splitlines() if s.strip()]


        
            
