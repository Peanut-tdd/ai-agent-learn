from concurrent.futures import thread
from pickle import NONE
from tkinter import NO
from  dotenv  import load_dotenv
from openai import OpenAI
import os
import json

from tools import TOOLS

import threading

_lock=threading.Lock()

class LlmApiClient:

    _instance=None

    def __init__(self,system_prompt):
        load_dotenv()
        self.api_key=os.environ.get("api_key")
        self.base_url=os.environ.get("base_url")
        self.model=os.environ.get("MODEL", "deepseek-chat")
        self.messages=[
            {"role": "system", "content": f'{system_prompt}'}
        ]

        if not self.api_key:
            raise ValueError('.env中缺少api_key')

        if not self.base_url:
            raise ValueError('.env中缺少base_url')
        
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)



    @classmethod
    def instance(cls,system_prompt:str="You are a helpful assistant"):
        if cls._instance is None:
            with _lock:
                if cls._instance is None:
                    cls._instance=cls(system_prompt)
        
        return cls._instance






    
    def think(self, messages:list, tools:list|None=TOOLS):
        """与模型对话。默认携带 tools（function calling）；
        纯文本评审（如 Reflection）时传 tools=None 关闭工具调用。"""

        print(f"\n====>>>LLM request\n")
        print_messages(messages)

        try:
            response=self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools
            )

            """jsrp=json.dumps(response.dict(), indent=2, ensure_ascii=False)
            print(f'模型返回结果是：{jsrp}')"""
            msg= response.choices[0].message

            print("\n====<<< LLM response\n")
            print_message(msg)
            
            print("\n\n")
            print(f"Model>\t {msg.content or ''}")
            print("\n\n")
            return msg
        except Exception as e:
            print(f"调用模型出错了：{e}")
            raise
            

            
    


def print_message(message) -> None:
    """格式化打印单条消息（dict 或 ChatCompletionMessage），自动去掉 None 字段。"""
    if hasattr(message, "model_dump"):
        data = message.model_dump(exclude_none=True)
    elif isinstance(message, dict):
        data = {k: v for k, v in message.items() if v is not None}
    else:
        data = message
    print(json.dumps(data, indent=2, ensure_ascii=False))


def print_messages(messages: list) -> None:
    for i, msg in enumerate(messages):
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", "?")
        print(f"--- [{i}] {role} ---")
        print_message(msg)


