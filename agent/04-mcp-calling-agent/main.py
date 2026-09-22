"""项目入口：交互式运行 ReAct Agent。"""
import asyncio
from agent.agent import agent


async def main():
    cli = agent.instance()

    while True:
        query = input("请输入：")

        if not query.strip():
            print("输入不能为空")
            continue
        if query.lower() in ["exit", "退出"]:
            break

        try:
            result = await cli.chat(query.strip())
            print(f"最终回答：{result.content}")
        except Exception as e:
            print(f"出错：{e}，请稍后再试。")


if __name__ == "__main__":
    asyncio.run(main()) 
