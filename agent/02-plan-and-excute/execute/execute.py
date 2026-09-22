from  tools.types  import Tool
from llm.llm_deepseek import LlmClient
from dotenv import load_dotenv
import os
from execute.prompt import REACT_PROMPT_TEMPLATE
import json
import re

class execute:
    

    def __init__(self,llm_client:LlmClient) -> None:
        load_dotenv()
        self.llm_client=llm_client
        self.debug=os.environ.get("debug", "false").lower() == "true"
        self.max_react_steps=int(os.environ.get("max_react_steps", "4"))
        self.history:list[tuple:(str,str,str)]=[]



    

    def execute_step(self, step: str, stat: dict[str:any], tools: dict[str, Tool], history: str = ""):
        """以 ReAct 作为单步执行器，结合已有 state 完成当前步骤。"""
        context="\n".join(f'- {k}:{v}' for k,v in stat.items()) or "(none)"
        if history:
            context += f"\n\n# 历史对话(与用户之前的问答):\n{history}"
        question = (
            f"Complete this step only:\n{step}\n\n"
            f"Prior step results:\n{context}\n\n"
            "Use tools if needed, then give Final Answer succinctly. "
         )
        return self.react_loop(question,tools)






    def react_loop(self,question:str,tools:dict[str,Tool],)->str:
        
        history:list[tuple(str,str,str)]=[]
        
        for step in range(self.max_react_steps + 1):
            if self.debug:
                print(f"\n-- ReAct Step:{step+1} Start --")
            
            prompt=self.build_prompt(question,tools,history)

            response_text=self.llm_client.think(prompt, stateless=True)

            thought, action = self._parse_output(response_text)
            if thought:
                print(f"思考: {thought}")

            if not action:
                print("警告:未能解析出有效的Action，流程终止。")
                break

            # 4. 执行Action
            if action.startswith("Finish"):
                # 如果是Finish指令，提取最终答案并结束
                final_answer = re.match(r"Finish\[(.*)\]", action).group(1)
                if self.debug:
                    print(f"🎉 最终答案: {final_answer}")
                return final_answer
            
            action, action_input = self._parse_action(action)
            if not action or action not in tools:
                continue
            else:
                obs = tools[action].run(action_input or "")

            self._print_tool_execution(action or "", action_input or "", obs)
            history.append((thought, f"{action}[{action_input}]", obs))
    
        return "Failed: max steps exceeded."





    def build_prompt(self,question:str,tools:dict[str:Tool],history:list[tuple:(str,str,str)]) ->str:
        
        tools_desc=self._GetAvailabeTools(tools)
        history_str=self._GetConvertHistiory(history)

        prompt=REACT_PROMPT_TEMPLATE.format(
            tools=tools_desc,
            question=question,
            history=history_str,
        )

        return prompt



    def _GetAvailabeTools(self,tools):

        return "\n".join([f"- {info.name}:{info.description}" for info in tools.values()]) or ("none")

    

    def _GetConvertHistiory(self,history):
        lines=[]
        for thought, action, obs in history:
            lines += [
                f"Thought: {thought}",
                f"Action: {action}",
                f"Observation: {obs}",
                "",
            ]

        return "\n".join(lines)


    # (这些方法是 ReActAgent 类的一部分)
    def _parse_output(self, text: str):
        """解析LLM的输出，提取Thought和Action。
        """
        # Thought: 匹配到 Action: 或文本末尾
        thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", text, re.DOTALL)
        # Action: 匹配到文本末尾
        action_match = re.search(r"Action:\s*(.*?)$", text, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        return thought, action

    def _parse_action(self, action_text: str):
        """解析Action字符串，提取工具名称和输入。
        """
        match = re.match(r"(\w+)\[(.*)\]", action_text, re.DOTALL)
        if match:
            return match.group(1), match.group(2)
        return None, None


    def _print_tool_execution(self,action: str, action_input: str, observation: str) -> None:
        print("\n=======>>> Tool execution (local function call) ======\n")
        print(
            json.dumps(
                {"action": action, "input": action_input, "observation": observation},
                indent=2,
                ensure_ascii=False,
            )
        )