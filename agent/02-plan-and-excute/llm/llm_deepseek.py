from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import datetime
import threading
from llm.log  import print_message
from llm.log import print_messages


_lock = threading.Lock()


class LlmClient:

    _instance = None

    def __init__(self,system_prompt:str="You are a helpful assistant") -> None:
        load_dotenv()
        self.api_key = os.environ.get("api_key")
        self.base_url = os.environ.get("base_url")
        self.model=os.environ.get("model")
        # 最多保留最近多少轮对话（1 轮 = 1 条 user + 1 条 assistant）
        self.max_rounds = 10
        self.messages=[
            {"role": "system", "content": f'{system_prompt}'}
        ]

        if not self.api_key:
            raise ValueError('.env中缺少api_key')

        if not self.base_url:
            raise ValueError('.env中缺少base_url')
        

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        # 每轮对话日志：默认写到项目根目录 logs/conversations.jsonl
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.log_file = os.environ.get(
            "LOG_FILE",
            os.path.join(project_root, "logs", "conversations.jsonl"),
        )
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

    @classmethod
    def instance(cls,system_prompt:str="You are a helpful assistant"):
        if cls._instance is None:
            with _lock:
                if cls._instance is None:
                    cls._instance = cls(system_prompt)
        return cls._instance


    

    def _log_round(self, prompt: str, result: str, stateless: bool) -> None:
        """把每一轮对话（用户输入 + 模型输出）追加记录到日志文件（JSONL，每行一条）"""
        record = {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "model": self.model,
            "stateless": stateless,  # False=普通对话，True=plan/ReAct 内部调用
            "prompt": prompt,
            "response": result,
        }
        with _lock:  # 多线程安全追加
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _trim_history(self):
        """超出 max_rounds 时，丢掉最早的对话轮次，保留 system 消息"""
        # system 之后每 2 条是一轮（user + assistant），若未成对则多留 1 条
        history = self.messages[1:]
        max_msgs = self.max_rounds * 2 + (len(history) % 2)
        if len(history) > max_msgs:
            self.messages = self.messages[:1] + history[-max_msgs:]

    def think(self, prompt: str, stateless: bool = False) -> str:

        # stateless=True：不累积到共享历史，每条 prompt 自带完整上下文（plan/ReAct 用）
        if stateless:
            messages = [self.messages[0], {"role": "user", "content": prompt}]
        else:
            self.messages.append({"role": "user", "content": prompt})
            messages = self.messages
           

        print("\n====>>>> LLM 输入\n")
        print_messages(messages)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            stream=True,
        )
        result=""
        for chunk in response:
            # 流式返回的每个 chunk 是 ChatCompletionChunk 对象，不是字符串
            content=chunk.choices[0].delta.content
            if content:
                result+=content
                #print(content,end='',flush=True)

        print(f'\n=====<<<<LLM 输出：\n')
        print_message(result)


        # 记录 AI 的回复，形成 user/assistant 交替的历史（stateless 模式不记录）
        if not stateless:
            self.messages.append({"role": "assistant", "content": result})
            # 超出上限则裁剪历史，防止上下文无限增长
            self._trim_history()

        # 无论是否 stateless，都记录这一轮对话
        #self._log_round(prompt, result, stateless)

        #print(f'\n{'='*50}')
        return result





if __name__=='__main__':

    

    while True:

        query=input("请输入：")
        if not query.strip():
            print("不能输入为空")
            continue
        if query.lower() in ["exit","退出"]:
            break

       
        

        client=LlmClient.instance()
        result=client.think(query)
        print(f'最终输出：{result}')
