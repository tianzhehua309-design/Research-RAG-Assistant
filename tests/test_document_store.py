from src.app.services.document_store import DocumentStore


def test_save_and_list_documents(tmp_path):
    store = DocumentStore(metadata_dir=tmp_path)

    store.save_document(
        {
            "doc_id": "doc_001",
            "filename": "note.md",
            "doc_type": "paper",
            "tag": "RAG",
            "source": "upload",
            "chunk_count": 2,
        }
    )

    documents = store.list_documents()

    assert len(documents) == 1
    assert documents[0]["doc_id"] == "doc_001"
    assert documents[0]["filename"] == "note.md"


def test_save_and_get_chunks_by_doc_id(tmp_path):
    store = DocumentStore(metadata_dir=tmp_path)

    chunks = [
        {
            "doc_id": "doc_001",
            "chunk_id": "doc_001_chunk_0000",
            "chunk_index": 0,
            "text": "This is chunk 0.",
            "metadata": {
                "doc_type": "paper",
                "tag": "RAG",
                "source": "upload",
            },
        }
    ]

    store.save_chunks("doc_001", chunks)

    saved_chunks = store.get_chunks_by_doc_id("doc_001")

    assert len(saved_chunks) == 1
    assert saved_chunks[0]["chunk_id"] == "doc_001_chunk_0000"
    assert saved_chunks[0]["text"] == "This is chunk 0."


def test_delete_document_removes_document_and_chunks(tmp_path):
    store = DocumentStore(metadata_dir=tmp_path)

    store.save_document(
        {
            "doc_id": "doc_001",
            "filename": "note.md",
            "doc_type": "paper",
            "tag": "RAG",
            "source": "upload",
        }
    )

    store.save_chunks(
        "doc_001",
        [
            {
                "doc_id": "doc_001",
                "chunk_id": "doc_001_chunk_0000",
                "chunk_index": 0,
                "text": "hello",
                "metadata": {},
            }
        ],
    )

    store.delete_document("doc_001")

    assert store.list_documents() == []
    assert store.get_chunks_by_doc_id("doc_001") == []