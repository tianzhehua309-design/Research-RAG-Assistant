import pytest

from src.app.errors import AppError
from src.app.services.chunker import chunk_text, split_long_text


def test_chunk_text_success():
    text = (
        "第一段：CLIP 是视觉语言模型。\n\n"
        "第二段：PGD 是常见对抗攻击方法。\n\n"
        "第三段：RAG 可以返回 citations。"
    )

    chunks = chunk_text(
        text=text,
        doc_id="doc_001",
        metadata={
            "doc_type": "paper",
            "tag": "VLM",
            "source": "upload",
        },
        chunk_size=200,
        overlap=20,
    )

    assert len(chunks) == 1
    assert chunks[0]["doc_id"] == "doc_001"
    assert chunks[0]["chunk_id"] == "doc_001_chunk_0000"
    assert chunks[0]["chunk_index"] == 0
    assert "CLIP" in chunks[0]["text"]
    assert chunks[0]["metadata"]["doc_type"] == "paper"
    assert chunks[0]["metadata"]["tag"] == "VLM"
    assert chunks[0]["metadata"]["source"] == "upload"


def test_chunk_text_splits_multiple_chunks():
    text = (
        "第一段内容很长。" * 20
        + "\n\n"
        + "第二段内容也很长。" * 20
        + "\n\n"
        + "第三段内容也很长。" * 20
    )

    chunks = chunk_text(
        text=text,
        doc_id="doc_002",
        metadata={"doc_type": "experiment", "tag": "PGD", "source": "upload"},
        chunk_size=120,
        overlap=20,
    )

    assert len(chunks) > 1
    assert chunks[0]["chunk_id"] == "doc_002_chunk_0000"
    assert chunks[1]["chunk_id"] == "doc_002_chunk_0001"

    for chunk in chunks:
        assert len(chunk["text"]) <= 120
        assert chunk["metadata"]["doc_id"] == "doc_002"
        assert "chunk_id" in chunk["metadata"]


def test_split_long_text_overlap_works():
    text = "abcdefghijklmnopqrstuvwxyz"

    chunks = split_long_text(
        text=text,
        chunk_size=10,
        overlap=3,
    )

    assert chunks[0] == "abcdefghij"
    assert chunks[1] == "hijklmnopq"
    assert chunks[2] == "opqrstuvwx"
    assert chunks[3] == "vwxyz"


def test_chunk_text_rejects_empty_text():
    with pytest.raises(AppError) as exc_info:
        chunk_text(
            text="   \n\n   ",
            doc_id="doc_003",
            metadata={},
        )

    assert exc_info.value.code == "EMPTY_TEXT"


def test_chunk_text_rejects_empty_doc_id():
    with pytest.raises(AppError) as exc_info:
        chunk_text(
            text="hello",
            doc_id="   ",
            metadata={},
        )

    assert exc_info.value.code == "EMPTY_DOC_ID"


def test_chunk_text_rejects_invalid_chunk_size():
    with pytest.raises(AppError) as exc_info:
        chunk_text(
            text="hello",
            doc_id="doc_004",
            metadata={},
            chunk_size=0,
            overlap=0,
        )

    assert exc_info.value.code == "INVALID_CHUNK_SIZE"


def test_chunk_text_rejects_invalid_overlap():
    with pytest.raises(AppError) as exc_info:
        chunk_text(
            text="hello",
            doc_id="doc_005",
            metadata={},
            chunk_size=100,
            overlap=100,
        )

    assert exc_info.value.code == "INVALID_OVERLAP"