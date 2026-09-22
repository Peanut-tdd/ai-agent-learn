"""各环节提示词。多轮对话优化：历史上下文贯穿 Execute/Reflect/Replan 三环，
并在提示词里明确告诉模型"历史是参考，只作答当前任务"，避免把历史当噪音。
"""


def build_execute_prompt(task, reflections, session_lessons, history):
    """Execute 提示词：任务 + 会话历史 + 跨任务教训 + 本轮反思指引。

    Args:
        task:           当前用户任务
        reflections:    本轮任务此前失败的评审/修改指引（必须遵守）
        session_lessons:历史任务沉淀的教训（参考）
        history:        会话前文（已完成的任务/答案）
    """
    lines = [
        f"Task: {task}",
        "",
        "以下是本次会话的历史记录（前文），仅作参考。历史中可能包含已完成的任务；",
        "你只负责完成上面的当前 Task，不要重复作答历史中已完成的任务，",
        "但如果当前任务需要引用历史内容（如“按上面的格式”“把刚才的结果改一下”），请直接使用。",
        "",
        f"History: {history or '（无历史）'}",
    ]

    if session_lessons:
        lines += ["", "过去任务沉淀的经验教训（遇到类似情形时参考，不要照搬到不相关任务）："]
        for i, lesson in enumerate(session_lessons, 1):
            lines.append(f"{i}. {lesson}")

    if reflections:
        lines += ["", "本轮任务此前失败的评审/修改指引（必须遵守）："]
        for i, r in enumerate(reflections, 1):
            lines.append(f"{i}. {r}")

    return "\n".join(lines)


def build_reflection_prompt(task, output, history):
    """评审员提示词：结合会话历史判断候选答案是否达标，输出 JSON 结论。"""
    return f"""你是严格的评审员，判断下面的任务是否已被候选答案正确完成。

会话历史（前文，任务可能引用其中的内容，评审时作为上下文）:
{history or "（无历史）"}

任务:
{task}

候选答案:
{output}

评审要求:
1. 先结合会话历史理解任务：如果任务引用了历史中的内容（如“刚才的答案”“上一轮的结果”），以历史为准；
2. 若答案正确、完整且直接回应了任务，passed 为 true；
3. 否则指出具体问题（遗漏、错误、不完整、与历史事实不符），并给出可执行的修改建议，passed 为 false。
只输出一个 JSON 对象，不要输出其他任何内容:
{{"passed": true, "feedback": "..."}}
"""


def build_replan_prompt(task, output, feedback, history):
    """修正提示词：结合会话历史，把评审反馈转成下一轮 Execute 的具体修改指引。"""
    return f"""你是修正规划器。根据评审员的反馈，给出下一轮执行时必须修改的具体要点。

会话历史（前文，用于理解任务上下文）:
{history or "（无历史）"}

任务:
{task}

上一轮候选答案:
{output}

评审员反馈:
{feedback}

要求:
1. 列出 2-4 条具体的修改要点（改什么、怎么改），直接可执行；
2. 如果问题与历史相关（任务引用了历史内容、答案与历史前后不一致），明确指出应如何结合历史修正；
3. 不要复述答案本身，只输出修改指引，每行一条，以 "- " 开头。
"""
