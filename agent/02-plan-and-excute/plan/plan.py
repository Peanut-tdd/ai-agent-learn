import ast
from plan.prompt import PLANNER_PROMPT_TEMPLATE

class planner:



    def __init__(self,llm_client):
        self.llm_client=llm_client




    def plan(self, question: str, history: str = ""):

        prompt = PLANNER_PROMPT_TEMPLATE.format(question=question, history=history or "无")

        response_text = self.llm_client.think(prompt, stateless=True)

           # 解析LLM输出的列表字符串
        try:
            # 找到```python和```之间的内容
            plan_str = response_text.split("```python")[1].split("```")[0].strip()
            # 使用ast.literal_eval来安全地执行字符串，将其转换为Python列表
            plan = ast.literal_eval(plan_str)

            print(f'plan result:{plan}')
            return plan if isinstance(plan, list) else []
        except (ValueError, SyntaxError, IndexError) as e:
            print(f"❌ 解析计划时出错: {e}")
            print(f"原始响应: {response_text}")
            return []
        except Exception as e:
            print(f"❌ 解析计划时发生未知错误: {e}")
            return []






