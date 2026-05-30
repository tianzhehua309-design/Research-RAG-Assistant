from src.app.errors import AppError

# 定义模型的身份和规则
SYSTEM_PROMPT = (
    "你是一个研究资料与实验记录问答助手。"
    "你必须只基于给定的 context 回答问题。"
    "如果 context 中没有足够依据，请明确说明：没有找到相关依据。"
    "不要编造不存在的论文、实验结果、指标或结论。"
    "回答要简洁、准确，并尽量指出依据来自哪些 chunk。"
)


def build_qa_messages(
    question: str,
    context: str,
) -> list[dict[str, str]]:
    cleaned_question = question.strip()
    cleaned_context = context.strip()

    if not cleaned_question:
        raise AppError(
            code="EMPTY_QUESTION",
            message="question 不能为空",
            retryable=False,
        )

    if not cleaned_context:
        cleaned_context = "未检索到相关上下文。"

    user_prompt = (
        "请根据下面的 context 回答 question。\n\n"
        "要求：\n"
        "1. 只能使用 context 中的信息回答。\n"
        "2. 如果 context 中没有答案，请说：没有找到相关依据。\n"
        "3. 不要编造 context 中不存在的信息。\n"
        "4. 回答中可以提到相关 chunk_id，方便用户追溯。\n\n"
        f"question:\n{cleaned_question}\n\n"
        f"context:\n{cleaned_context}"
    )

    # 返回的是 OpenAI-compatible chat messages 格式
    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]