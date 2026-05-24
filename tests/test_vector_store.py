from src.app.clients.vector_store import ChromaVectorStore


def test_vector_store_add_and_query(tmp_path):
    store = ChromaVectorStore(
        persist_directory=str(tmp_path / "chroma"),
        collection_name="test_chunks",
    )

    chunks = [
        {
            "doc_id": "doc_001",
            "chunk_id": "doc_001_chunk_0000",
            "chunk_index": 0,
            "text": "Python and FastAPI are useful for building APIs.",
            "metadata": {
                "doc_id": "doc_001",
                "chunk_id": "doc_001_chunk_0000",
                "doc_type": "note",
                "tag": "FastAPI",
                "source": "upload",
            },
        },
        {
            "doc_id": "doc_001",
            "chunk_id": "doc_001_chunk_0001",
            "chunk_index": 1,
            "text": "RAG uses retrieval and generation.",
            "metadata": {
                "doc_id": "doc_001",
                "chunk_id": "doc_001_chunk_0001",
                "doc_type": "note",
                "tag": "RAG",
                "source": "upload",
            },
        },
    ]

    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]

    count = store.add_chunks(chunks, embeddings)

    assert count == 2

    results = store.query(
        query_embedding=[1.0, 0.0, 0.0],
        top_k=1,
    )

    assert len(results) == 1
    assert results[0]["chunk_id"] == "doc_001_chunk_0000"
    assert "FastAPI" in results[0]["text"]


def test_vector_store_query_with_metadata_filter(tmp_path):
    store = ChromaVectorStore(
        persist_directory=str(tmp_path / "chroma"),
        collection_name="test_filter_chunks",
    )

    chunks = [
        {
            "doc_id": "doc_001",
            "chunk_id": "doc_001_chunk_0000",
            "chunk_index": 0,
            "text": "This is a paper about CLIP robustness.",
            "metadata": {
                "doc_id": "doc_001",
                "chunk_id": "doc_001_chunk_0000",
                "doc_type": "paper",
                "tag": "VLM",
                "source": "upload",
            },
        },
        {
            "doc_id": "doc_002",
            "chunk_id": "doc_002_chunk_0000",
            "chunk_index": 0,
            "text": "This is an experiment log about API tests.",
            "metadata": {
                "doc_id": "doc_002",
                "chunk_id": "doc_002_chunk_0000",
                "doc_type": "experiment",
                "tag": "API",
                "source": "upload",
            },
        },
    ]

    embeddings = [
        [1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],
    ]

    store.add_chunks(chunks, embeddings)

    results = store.query(
        query_embedding=[1.0, 0.0, 0.0],
        top_k=2,
        filters={"doc_type": "paper"},
    )

    assert len(results) == 1
    assert results[0]["doc_id"] == "doc_001"
    assert results[0]["metadata"]["doc_type"] == "paper"


def test_vector_store_delete_by_doc_id(tmp_path):
    store = ChromaVectorStore(
        persist_directory=str(tmp_path / "chroma"),
        collection_name="test_delete_chunks",
    )

    chunks = [
        {
            "doc_id": "doc_001",
            "chunk_id": "doc_001_chunk_0000",
            "chunk_index": 0,
            "text": "This chunk should be deleted.",
            "metadata": {
                "doc_id": "doc_001",
                "chunk_id": "doc_001_chunk_0000",
                "doc_type": "note",
                "tag": "delete",
                "source": "upload",
            },
        }
    ]

    embeddings = [[1.0, 0.0, 0.0]]

    store.add_chunks(chunks, embeddings)
    store.delete_by_doc_id("doc_001")

    results = store.query(
        query_embedding=[1.0, 0.0, 0.0],
        top_k=1,
    )

    assert results == []