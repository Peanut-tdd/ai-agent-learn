import os
import json
import threading

from dotenv import load_dotenv
from openai import OpenAI

# LLM 调用超时与客户端重试
LLM_TIMEOUT = 60.0
LLM_MAX_RETRIES = 2

_lock = threading.Lock()


class llmapi:

    _client = None

    def __init__(self) -> None:
        load_dotenv()
        self.api_key = os.environ.get("api_key")
        if not self.api_key:
            raise ValueError(".env 中缺少 api_key")
        self.base_url = os.environ.get("base_url")
        if not self.base_url:
            raise ValueError(".env中缺少base_url")
        self.model = os.environ.get("MODEL", "deepseek-v4-pro")
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=LLM_TIMEOUT,
            max_retries=LLM_MAX_RETRIES,
        )

    @classmethod
    def instance(cls) -> "llmapi":
        if cls._client is None:
            with _lock:
                if cls._client is None:
                    cls._client = cls()
        return cls._client

    def invoke(self, messages: list, tools: list | None = None):
        """调用 chat.completions，返回 assistant message（含 content / tool_calls）。"""
        print("\n====<<< LLM request\n")
        print_messages(messages=messages)

        kwargs: dict = {"model": self.model, "messages": messages}
        if tools:
            kwargs["tools"] = tools

        try:
            response = self.client.chat.completions.create(**kwargs)
            msg = response.choices[0].message
            print("\n====<<< LLM response\n")
            print_message(msg)

            print("\n\n")
            print(f"Model>\t {msg.content or ''}")
            print("\n\n")
            return msg
        except Exception as e:
            print(f"模型报错了{e}")
            raise


def print_message(message) -> None:
    """格式化打印单条消息（dict 或 ChatCompletionMessage），自动去掉 None 字段。"""
    if hasattr(message, "model_dump"):
        data = message.model_dump(exclude_none=True)
    elif isinstance(message, dict):
        data = {k: v for k, v in message.items() if v is not None}
    else:
        data = message
    print(json.dumps(data, indent=2, ensure_ascii=False, default=str))


def print_messages(messages: list) -> None:
    for i, msg in enumerate(messages):
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", "?")
        print(f"--- [{i}] {role} ---")
        print_message(msg)
