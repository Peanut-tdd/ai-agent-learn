## plan-and-excutor 先规划再每逐步执行（注入式记忆实现多轮对话）

Plan-and-Solve:我们实现了一个先规划后执行的 Plan-and-Solve 智能体，并利用它解决了需要多步推理的数学应用题。它将复杂的任务分解为清晰的步骤，然后逐一执行。其核心优势在于**结构性**和**稳定性**，特别适合处理逻辑路径确定、内部推理密集的任务。

### 创建环境

```
python -m venv .venv

source .venv/bin/activate
```

### 扩展下载

```
pip install -r erquirement.txt
```

### 运行

```
进入项目根目录
python -m main或者python main.py
```

