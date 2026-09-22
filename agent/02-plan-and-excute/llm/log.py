
import json
from pyexpat.errors import messages
from tkinter import NO

def _dict(message)->dict:
    #优先处理 Pydantic 模型
    if hasattr(message,"model_dump"):
       return message.model_dump(exclude_none=None) 
    if isinstance(message,dict):
        return {k:v for k,v in message.items() if v is not None}
    return {"value":message}



def print_message(message)->None:

    if isinstance(message,dict):
        data=_dict(message)
        content=data.pop("content",None)

        extras={k:v for k,v in data.items()if k!="role"}
        if extras:
            print(json.dumps(extras,indent=2,ensure_ascii=False))
    if isinstance(message,str):
        content=message

    if  content is None:
        return
    
    print("content:")
    text=content if isinstance(content,str) else json.dumps(content,ensure_ascii=False)
    for line in text.splitlines():
        print(f'{line}')



def print_messages(messages:list):
    for i,msg in enumerate(messages):
        role=msg.get("role") if isinstance(msg,dict) else  getattr(msg,"role","?")
        print(f'----[{i}] {role} ----')
        print_message(msg)
