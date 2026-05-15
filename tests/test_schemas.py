import pytest
from pydantic import ValidationError

from src.app.schemas import (
    AskRequest,
    Citation,
    DocumentInfo,
    DocumentMetadata,
    SearchRequest,
)

def test_document_info_schema_valid():
    metadata = DocumentMetadata(
        doc_type="paper",
        source="upload",
        tag="VLM",
    )

    document = DocumentInfo(
        doc_id="doc_001",
        filename="clip_robustness.md",
        metadata=metadata,
        created_at="2026-05-15T10:00:00Z",
        status="uploaded",
        chunk_count=0,
    )

    assert document.doc_id == "doc_001"
    assert document.metadata.doc_type == "paper"
    assert document.metadata.tag == "VLM"
    assert document.chunk_count == 0

def test_search_request_default_top_k():
    request = SearchRequest(
        query="CLIP 的对抗鲁棒性怎么样？",
    )

    assert request.query == "CLIP 的对抗鲁棒性怎么样？"
    assert request.top_k == 5
    assert request.filters is None

def test_search_request_rejects_invalid_top_k():
    with pytest.raises(ValidationError):
        SearchRequest(
            query="CLIP 的对抗鲁棒性怎么样？",
            top_k=0,
        )

def test_ask_request_rejects_empty_question():
    with pytest.raises(ValidationError):
        AskRequest(
            question="",
        )


def test_citation_schema_valid():
    citation = Citation(
        doc_id="doc_001",
        chunk_id="doc_001_chunk_0003",
        filename="clip_robustness.md",
        text_snippet="CLIP shows vulnerability under adversarial perturbations...",
        score=0.82,
    )

    assert citation.doc_id == "doc_001"
    assert citation.chunk_id == "doc_001_chunk_0003"
    assert citation.score == 0.82