import pytest
from pydantic import ValidationError

from src.app.schemas import AskRequest, AskResponse, Citation


def test_ask_request_success():
    request = AskRequest(
        question="RAG 的基本流程是什么？",
        top_k=5,
        filters={
            "doc_type": "paper",
            "tag": "RAG",
        },
    )

    assert request.question == "RAG 的基本流程是什么？"
    assert request.top_k == 5
    assert request.filters == {
        "doc_type": "paper",
        "tag": "RAG",
    }


def test_ask_request_strips_question():
    request = AskRequest(
        question="  RAG 是什么？  ",
    )

    assert request.question == "RAG 是什么？"


def test_ask_request_rejects_empty_question():
    with pytest.raises(ValidationError):
        AskRequest(
            question="   ",
        )


def test_ask_request_rejects_invalid_top_k_too_small():
    with pytest.raises(ValidationError):
        AskRequest(
            question="RAG 是什么？",
            top_k=0,
        )


def test_ask_request_rejects_invalid_top_k_too_large():
    with pytest.raises(ValidationError):
        AskRequest(
            question="RAG 是什么？",
            top_k=100,
        )


def test_ask_request_rejects_invalid_filter_key():
    with pytest.raises(ValidationError):
        AskRequest(
            question="RAG 是什么？",
            filters={
                "unknown": "RAG",
            },
        )


def test_citation_success():
    citation = Citation(
        doc_id="doc_001",
        chunk_id="doc_001_chunk_0001",
        filename="rag_note.md",
        source="upload",
        doc_type="paper",
        tag="RAG",
        text_snippet="RAG 的流程包括检索和生成。",
        score=0.82,
    )

    assert citation.doc_id == "doc_001"
    assert citation.chunk_id == "doc_001_chunk_0001"
    assert citation.score == 0.82


def test_ask_response_success():
    citation = Citation(
        doc_id="doc_001",
        chunk_id="doc_001_chunk_0001",
        filename="rag_note.md",
        text_snippet="RAG 的流程包括检索和生成。",
        score=0.82,
    )

    response = AskResponse(
        answer="RAG 的基本流程包括文档解析、切分、向量化、检索和生成回答。",
        citations=[citation],
    )

    assert "RAG" in response.answer
    assert len(response.citations) == 1
    assert response.citations[0].chunk_id == "doc_001_chunk_0001"