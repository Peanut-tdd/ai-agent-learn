import re
from tools import build_default_tools
from llm.llm_deepseek import LlmClient
from agent.agent import agent



def main():
    tools=build_default_tools()
    llm=LlmClient.instance(
        system_prompt=(           
            "You are a Plan-and-Execute agent assistant. "
            "When planning, output numbered steps one per line. "
            "When executing or summarizing, use the same language as the task.")
    )



    agent_instance=agent(llm,tools)

    while True:
        query=input("请输入：")
        if not query.strip():
            print("输入不能为空")
            continue
        if query.lower() in ["exit","退出"]:
            break
    
        result=agent_instance.plan_and_execute(query)
        print(f"最终回答是：{result}")

   
    



if __name__=="__main__":
    main()