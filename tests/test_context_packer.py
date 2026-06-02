from src.app.services.context_packer import pack_context, truncate_text


def test_pack_context_sorts_by_score():
    results = [
        {
            "doc_id": "doc_1",
            "chunk_id": "chunk_low",
            "filename": "a.md",
            "text": "low score text",
            "score": 0.2,
        },
        {
            "doc_id": "doc_1",
            "chunk_id": "chunk_high",
            "filename": "a.md",
            "text": "high score text",
            "score": 0.9,
        },
    ]

    packed = pack_context(results)

    context = packed["context"]

    assert context.index("chunk_high") < context.index("chunk_low")


def test_pack_context_removes_duplicate_chunks():
    results = [
        {
            "doc_id": "doc_1",
            "chunk_id": "chunk_1",
            "filename": "a.md",
            "text": "same text",
            "score": 0.9,
        },
        {
            "doc_id": "doc_1",
            "chunk_id": "chunk_1",
            "filename": "a.md",
            "text": "same text duplicated",
            "score": 0.8,
        },
    ]

    packed = pack_context(results)

    assert len(packed["citations"]) == 1
    assert packed["context"].count("chunk_1") == 1


def test_pack_context_respects_max_chunks():
    results = []

    for index in range(5):
        results.append(
            {
                "doc_id": "doc_1",
                "chunk_id": f"chunk_{index}",
                "filename": "a.md",
                "text": f"text {index}",
                "score": 1.0 - index * 0.1,
            }
        )

    packed = pack_context(results, max_chunks=2)

    assert len(packed["citations"]) == 2


def test_pack_context_respects_max_context_chars():
    results = [
        {
            "doc_id": "doc_1",
            "chunk_id": "chunk_1",
            "filename": "a.md",
            "text": "a" * 1000,
            "score": 0.9,
        }
    ]

    packed = pack_context(results, max_context_chars=200)

    assert len(packed["context"]) <= 200


def test_pack_context_skips_invalid_result():
    results = [
        {
            "doc_id": "",
            "chunk_id": "chunk_1",
            "filename": "a.md",
            "text": "invalid",
            "score": 0.9,
        },
        {
            "doc_id": "doc_1",
            "chunk_id": "chunk_2",
            "filename": "a.md",
            "text": "valid",
            "score": 0.8,
        },
    ]

    packed = pack_context(results)

    assert len(packed["citations"]) == 1
    assert packed["citations"][0]["chunk_id"] == "chunk_2"


def test_truncate_text_short_text():
    text = "hello"

    assert truncate_text(text, 10) == "hello"


def test_truncate_text_long_text():
    text = "abcdefghijklmnopqrstuvwxyz"

    result = truncate_text(text, 10)

    assert result == "abcdefg..."
    assert len(result) == 10