from pathlib import Path
from typing import Any

import chromadb

from src.app.errors import AppError


class ChromaVectorStore:
    def __init__(
        self,
        persist_directory: str = "data/chroma",
        collection_name: str = "research_chunks",
    ) -> None:
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name
        )

    def add_chunks(
        self,
        chunks: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> int:
        if not chunks:
            raise AppError(
                code="EMPTY_CHUNKS",
                message="chunks 不能为空",
                retryable=False,
            )

        if len(chunks) != len(embeddings):
            raise AppError(
                code="EMBEDDING_COUNT_MISMATCH",
                message="chunks 数量和 embeddings 数量不一致",
                retryable=False,
            )

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for chunk in chunks:
            chunk_id = chunk.get("chunk_id")
            text = chunk.get("text")
            metadata = chunk.get("metadata", {})

            if not chunk_id or not isinstance(chunk_id, str):
                raise AppError(
                    code="INVALID_CHUNK",
                    message="chunk_id 缺失或不合法",
                    retryable=False,
                )

            if not text or not isinstance(text, str):
                raise AppError(
                    code="INVALID_CHUNK",
                    message="chunk text 缺失或不合法",
                    retryable=False,
                )

            clean_metadata = self._build_metadata(chunk, metadata)

            ids.append(chunk_id)
            documents.append(text)
            metadatas.append(clean_metadata)

        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        return len(ids)

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if not query_embedding:
            raise AppError(
                code="EMPTY_QUERY_EMBEDDING",
                message="query embedding 不能为空",
                retryable=False,
            )

        if top_k <= 0:
            raise AppError(
                code="INVALID_TOP_K",
                message="top_k 必须大于 0",
                retryable=False,
            )

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filters,
            include=["documents", "metadatas", "distances"],
        )

        return self._format_query_results(results)

    def delete_by_doc_id(self, doc_id: str) -> None:
        if not doc_id.strip():
            raise AppError(
                code="EMPTY_DOC_ID",
                message="doc_id 不能为空",
                retryable=False,
            )

        self.collection.delete(
            where={"doc_id": doc_id}
        )

    def _build_metadata(
        self,
        chunk: dict[str, Any],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        result = {
            **metadata,
            "doc_id": chunk.get("doc_id", metadata.get("doc_id", "")),
            "chunk_id": chunk.get("chunk_id", metadata.get("chunk_id", "")),
            "chunk_index": chunk.get("chunk_index", metadata.get("chunk_index", 0)),
        }

        clean_result = {}

        for key, value in result.items():
            if isinstance(value, (str, int, float, bool)):
                clean_result[key] = value
            elif value is None:
                continue
            else:
                clean_result[key] = str(value)

        return clean_result

    def _format_query_results(
        self,
        results: dict[str, Any],
    ) -> list[dict[str, Any]]:
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        formatted_results: list[dict[str, Any]] = []

        for chunk_id, text, metadata, distance in zip(
            ids,
            documents,
            metadatas,
            distances,
        ):
            score = 1 / (1 + distance)

            formatted_results.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": metadata.get("doc_id", ""),
                    "text": text,
                    "metadata": metadata,
                    "distance": distance,
                    "score": score,
                }
            )

        return formatted_results