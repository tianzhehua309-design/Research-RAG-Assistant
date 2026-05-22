import pytest

from src.app.clients.embedding_client import HashEmbeddingClient, MockEmbeddingClient
from src.app.errors import AppError


def test_mock_embedding_dimension():
    client = MockEmbeddingClient(dimension=8)

    vector = client.embed_query("This is a RAG note.")

    assert len(vector) == 8
    assert vector[0] == 1.0


def test_hash_embedding_dimension():
    client = HashEmbeddingClient(dimension=16)

    vector = client.embed_query("This is a RAG note.")

    assert len(vector) == 16


def test_hash_embedding_is_stable():
    client = HashEmbeddingClient(dimension=16)

    vector_1 = client.embed_query("Python FastAPI RAG")
    vector_2 = client.embed_query("Python FastAPI RAG")

    assert vector_1 == vector_2


def test_embed_texts_returns_one_vector_per_text():
    client = HashEmbeddingClient(dimension=16)

    vectors = client.embed_texts(
        [
            "Python FastAPI",
            "RAG Chroma",
            "CLIP adversarial robustness",
        ]
    )

    assert len(vectors) == 3
    assert all(len(vector) == 16 for vector in vectors)


def test_empty_text_raises_app_error():
    client = HashEmbeddingClient(dimension=16)

    with pytest.raises(AppError) as exc_info:
        client.embed_query("   ")

    assert exc_info.value.code == "EMPTY_TEXT"