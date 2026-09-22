from llm.llm_deepseek import LlmApiClient
from agent.agent_loop import  AgentLoop
 
 
def main():
    llm=LlmApiClient.instance(
        system_prompt=(
            "You are a Reflection agent assistant. "
            "Follow the requested output format exactly. "
            "Use the same language as the task."
        )
    )

    agent_loop=AgentLoop(llm)
    while True:
        query=input("请输入：")
        if not query.strip():
            print("输入不能为空")
            continue
        if query.lower() in ["exit","退出"]:
            break
        try:
            result=agent_loop.run(query)
            print(result)
        except Exception as e:
            print(f"Error: {e}")



if __name__ == "__main__":
    main()
