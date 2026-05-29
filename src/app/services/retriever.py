from typing import Any

from src.app.clients.embedding_client import HashEmbeddingClient
from src.app.clients.vector_store import ChromaVectorStore
from src.app.errors import AppError

def build_metadata_filter(filters: dict[str, Any] | None) -> dict[str, Any] | None:
    if not filters:
        return None

    conditions: list[dict[str, Any]] = []

    for key in ("doc_type", "tag", "source"):
        value = filters.get(key)

        if isinstance(value, str) and value.strip():
            conditions.append(
                {
                    key: {
                        "$eq": value.strip(),
                    }
                }
            )

    if not conditions:
        return None

    if len(conditions) == 1:
        return conditions[0]

    return {
        "$and": conditions,
    }


def search_chunks(
    query: str,
    top_k: int = 5,
    filters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    cleaned_query = query.strip()

    if not cleaned_query:
        raise AppError(
            code="EMPTY_QUERY",
            message="query 不能为空",
            retryable=False,
        )

    if top_k <= 0:
        raise AppError(
            code="INVALID_TOP_K",
            message="top_k 必须大于 0",
            retryable=False,
        )

    metadata_filter = build_metadata_filter(filters)

    embedding_client = HashEmbeddingClient()
    vector_store = ChromaVectorStore()

    query_embedding = embedding_client.embed_query(cleaned_query)

    results = vector_store.query(
        query_embedding=query_embedding,
        top_k=top_k,
        filters=metadata_filter,
    )

    return results

def query(
    self,
    query_embedding: list[float],
    top_k: int = 5,
    filters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    results = self.collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=filters,
        include=["documents", "metadatas", "distances"],
    )

    return self._format_query_results(results)