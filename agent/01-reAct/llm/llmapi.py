import os
from openai import OpenAI
from dotenv import load_dotenv
import threading
from  tools import TOOLS
import json

_lock = threading.Lock()

# LLM 调用超时与客户端重试
LLM_TIMEOUT = 60.0
LLM_MAX_RETRIES = 2

class llmapi:

    _client=None

    


    def __init__(self) -> None:
        load_dotenv()
        self.api_key=os.environ.get("api_key")
        # 兼容 .env 中的旧变量名 api_url
        self.base_url=os.environ.get("base_url") or os.environ.get("api_url")
        if not self.api_key:
            raise ValueError(".env 中缺少 api_key")
        if not self.base_url:
            raise ValueError(".env 中缺少 base_url（或 api_url）")
        self.model=os.environ.get("MODEL", "deepseek-v4-pro")
        
        print(f"config:{self.base_url}-{self.api_key}-{self.model}")
        self.client=OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=LLM_TIMEOUT,
            max_retries=LLM_MAX_RETRIES,
        )

    @classmethod
    def instance(cls):
        if not cls._client:
            with _lock:
                cls._client=cls()
        return cls._client



    def generate(self, messages:list):

        print(f'模型入参：{messages}')

        try:
            response=self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOLS
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



