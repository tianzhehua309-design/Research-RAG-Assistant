from src.app.schemas import SearchResult
from src.app.services.context_builder import build_context_and_citations


def test_build_context_and_citations_success():
    result = SearchResult(
        doc_id="doc_001",
        chunk_id="doc_001_chunk_0000",
        filename="rag_note.md",
        text="RAG 的基本流程包括文档上传、chunking、embedding、检索和回答生成。",
        score=0.91,
        metadata={
            "source": "upload",
            "doc_type": "paper",
            "tag": "RAG",
        },
    )

    context, citations = build_context_and_citations([result])

    assert "doc_001" in context
    assert "doc_001_chunk_0000" in context
    assert "rag_note.md" in context
    assert "upload" in context
    assert "RAG 的基本流程" in context

    assert len(citations) == 1
    assert citations[0].doc_id == "doc_001"
    assert citations[0].chunk_id == "doc_001_chunk_0000"
    assert citations[0].filename == "rag_note.md"
    assert "RAG 的基本流程" in citations[0].text_snippet
    assert citations[0].score == 0.91


def test_build_context_and_citations_empty_results():
    context, citations = build_context_and_citations([])

    assert context == ""
    assert citations == []


def test_build_context_and_citations_skips_empty_text():
    result = SearchResult(
        doc_id="doc_001",
        chunk_id="doc_001_chunk_0000",
        filename="empty.md",
        text="   ",
        score=0.5,
        metadata={
            "source": "upload",
            "doc_type": "note",
            "tag": "empty",
        },
    )

    context, citations = build_context_and_citations([result])

    assert context == ""
    assert citations == []


def test_build_context_and_citations_truncates_snippet():
    long_text = "a" * 500

    result = SearchResult(
        doc_id="doc_001",
        chunk_id="doc_001_chunk_0000",
        filename="long.md",
        text=long_text,
        score=0.88,
        metadata={
            "source": "upload",
            "doc_type": "paper",
            "tag": "long",
        },
    )

    context, citations = build_context_and_citations(
        [result],
        max_snippet_chars=100,
    )

    assert len(citations) == 1
    assert len(citations[0].text_snippet) == 103
    assert citations[0].text_snippet.endswith("...")
    assert long_text in context