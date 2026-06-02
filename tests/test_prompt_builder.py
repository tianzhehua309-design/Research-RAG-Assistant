import pytest

from src.app.errors import AppError
from src.app.services.prompt_builder import build_qa_messages


def test_build_qa_messages_success():
    messages = build_qa_messages(
        question="RAG 的基本流程是什么？",
        context="[chunk_id=doc_001_chunk_0001]\nRAG 包括检索和生成。",
    )

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "RAG 的基本流程是什么？" in messages[1]["content"]
    assert "doc_001_chunk_0001" in messages[1]["content"]


def test_build_qa_messages_rejects_empty_question():
    with pytest.raises(AppError) as exc_info:
        build_qa_messages(
            question="   ",
            context="some context",
        )

    assert exc_info.value.code == "EMPTY_QUESTION"
    assert exc_info.value.retryable is False


def test_build_qa_messages_handles_empty_context():
    messages = build_qa_messages(
        question="RAG 的基本流程是什么？",
        context="   ",
    )

    assert "未检索到相关上下文" in messages[1]["content"]