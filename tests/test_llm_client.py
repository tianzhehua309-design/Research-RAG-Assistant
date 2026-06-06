import httpx
import pytest

from src.app.clients.llm_client import LLMClient, MockQAClient
from src.app.errors import AppError


class FakeResponse:
    def __init__(
        self,
        status_code: int = 200,
        json_data: dict | None = None,
        text: str = "",
    ) -> None:
        self.status_code = status_code
        self._json_data = json_data
        self.text = text

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request(
                method="POST",
                url="https://example.com/chat/completions",
            )
            response = httpx.Response(
                status_code=self.status_code,
                text=self.text,
                request=request,
            )
            raise httpx.HTTPStatusError(
                message=f"HTTP {self.status_code}",
                request=request,
                response=response,
            )

    def json(self) -> dict:
        if self._json_data is None:
            raise ValueError("invalid json")

        return self._json_data


def test_mock_qa_client_success() -> None:
    client = MockQAClient()

    answer = client.generate_answer(
        question="RAG 的基本流程是什么？",
        context="RAG 包括文档解析、chunking、embedding、检索、上下文组装和回答生成。",
        request_id="test-request-id",
    )

    assert "RAG 的基本流程是什么" in answer
    assert "citations" in answer


def test_mock_qa_client_rejects_empty_question() -> None:
    client = MockQAClient()

    with pytest.raises(AppError) as exc_info:
        client.generate_answer(
            question="   ",
            context="some context",
        )

    assert exc_info.value.code == "EMPTY_QUESTION"
    assert exc_info.value.retryable is False


def test_mock_qa_client_empty_context_returns_no_evidence_answer() -> None:
    client = MockQAClient()

    answer = client.generate_answer(
        question="RAG 的基本流程是什么？",
        context="   ",
    )

    assert "没有找到相关依据" in answer


def test_llm_client_success(monkeypatch) -> None:
    def fake_post(*args, **kwargs):
        return FakeResponse(
            status_code=200,
            json_data={
                "choices": [
                    {
                        "message": {
                            "content": "RAG 的基本流程包括文档解析、切分、向量化、检索、上下文组装和回答生成。"
                        }
                    }
                ]
            },
        )

    monkeypatch.setattr(httpx, "post", fake_post)

    client = LLMClient(
        base_url="https://example.com",
        api_key="test-api-key",
        model="test-model",
    )

    answer = client.generate_answer(
        question="RAG 的基本流程是什么？",
        context="RAG 包括文档解析、chunking、embedding、检索和回答生成。",
        request_id="test-request-id",
    )

    assert "RAG" in answer
    assert "文档解析" in answer


def test_llm_client_rejects_empty_question() -> None:
    client = LLMClient(
        base_url="https://example.com",
        api_key="test-api-key",
        model="test-model",
    )

    with pytest.raises(AppError) as exc_info:
        client.generate_answer(
            question="   ",
            context="some context",
        )

    assert exc_info.value.code == "EMPTY_QUESTION"
    assert exc_info.value.retryable is False


def test_llm_client_timeout_is_retryable(monkeypatch) -> None:
    def fake_post(*args, **kwargs):
        raise httpx.TimeoutException("timeout")

    monkeypatch.setattr(httpx, "post", fake_post)

    client = LLMClient(
        base_url="https://example.com",
        api_key="test-api-key",
        model="test-model",
        max_retries=0,
    )

    with pytest.raises(AppError) as exc_info:
        client.generate_answer(
            question="RAG 的基本流程是什么？",
            context="some context",
        )

    assert exc_info.value.code == "UPSTREAM_TIMEOUT"
    assert exc_info.value.retryable is True


def test_llm_client_500_is_retryable(monkeypatch) -> None:
    def fake_post(*args, **kwargs):
        return FakeResponse(
            status_code=500,
            text="server error",
        )

    monkeypatch.setattr(httpx, "post", fake_post)

    client = LLMClient(
        base_url="https://example.com",
        api_key="test-api-key",
        model="test-model",
        max_retries=0,
    )

    with pytest.raises(AppError) as exc_info:
        client.generate_answer(
            question="RAG 的基本流程是什么？",
            context="some context",
        )

    assert exc_info.value.code == "UPSTREAM_SERVER_ERROR"
    assert exc_info.value.retryable is True


def test_llm_client_400_is_not_retryable(monkeypatch) -> None:
    def fake_post(*args, **kwargs):
        return FakeResponse(
            status_code=400,
            text="bad request",
        )

    monkeypatch.setattr(httpx, "post", fake_post)

    client = LLMClient(
        base_url="https://example.com",
        api_key="test-api-key",
        model="test-model",
        max_retries=2,
    )

    with pytest.raises(AppError) as exc_info:
        client.generate_answer(
            question="RAG 的基本流程是什么？",
            context="some context",
        )

    assert exc_info.value.code == "UPSTREAM_BAD_REQUEST"
    assert exc_info.value.retryable is False


def test_llm_client_invalid_json_is_retryable(monkeypatch) -> None:
    def fake_post(*args, **kwargs):
        return FakeResponse(
            status_code=200,
            json_data=None,
        )

    monkeypatch.setattr(httpx, "post", fake_post)

    client = LLMClient(
        base_url="https://example.com",
        api_key="test-api-key",
        model="test-model",
        max_retries=0,
    )

    with pytest.raises(AppError) as exc_info:
        client.generate_answer(
            question="RAG 的基本流程是什么？",
            context="some context",
        )

    assert exc_info.value.code == "UPSTREAM_INVALID_JSON"
    assert exc_info.value.retryable is True


def test_llm_client_invalid_response_structure_is_retryable(monkeypatch) -> None:
    def fake_post(*args, **kwargs):
        return FakeResponse(
            status_code=200,
            json_data={
                "wrong": "data",
            },
        )

    monkeypatch.setattr(httpx, "post", fake_post)

    client = LLMClient(
        base_url="https://example.com",
        api_key="test-api-key",
        model="test-model",
        max_retries=0,
    )

    with pytest.raises(AppError) as exc_info:
        client.generate_answer(
            question="RAG 的基本流程是什么？",
            context="some context",
        )

    assert exc_info.value.code == "UPSTREAM_INVALID_RESPONSE"
    assert exc_info.value.retryable is True