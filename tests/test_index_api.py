from pathlib import Path

from fastapi.testclient import TestClient

from src.app import main
from src.app.errors import AppError
from src.app.main import app
from src.app.services.document_store import DocumentStore


client = TestClient(app)


def setup_tmp_store(tmp_path, monkeypatch):
    metadata_dir = tmp_path / "metadata"
    upload_dir = tmp_path / "upload"

    metadata_dir.mkdir(parents=True, exist_ok=True)
    upload_dir.mkdir(parents=True, exist_ok=True)

    store = DocumentStore(metadata_dir=metadata_dir)

    monkeypatch.setattr(main, "document_store", store)
    monkeypatch.setattr(main, "UPLOAD_DIR", upload_dir)

    return store, upload_dir


def test_index_document_not_found(tmp_path, monkeypatch):
    setup_tmp_store(tmp_path, monkeypatch)

    response = client.post(
        "/documents/index",
        json={
            "doc_id": "doc_not_exists",
            "chunk_size": 800,
            "overlap": 100,
        },
    )

    assert response.status_code == 400

    body = response.json()
    assert body["error"]["code"] == "DOCUMENT_NOT_FOUND"
    assert body["error"]["retryable"] is False


def test_upload_then_index_success(tmp_path, monkeypatch):
    setup_tmp_store(tmp_path, monkeypatch)

    def fake_index_document(doc_id: str, chunk_size: int = 800, overlap: int = 100):
        return {
            "doc_id": doc_id,
            "indexed": True,
            "chunk_count": 1,
        }

    monkeypatch.setattr(main, "index_document", fake_index_document)

    upload_response = client.post(
        "/documents/upload",
        files={
            "file": (
                "note.md",
                b"# RAG Note\n\nThis is a test document for RAG.",
                "text/markdown",
            )
        },
        data={
            "doc_type": "paper",
            "tag": "RAG",
            "source": "upload",
        },
    )

    assert upload_response.status_code == 200

    doc_id = upload_response.json()["doc_id"]

    index_response = client.post(
        "/documents/index",
        json={
            "doc_id": doc_id,
            "chunk_size": 800,
            "overlap": 100,
        },
    )

    assert index_response.status_code == 200

    body = index_response.json()
    assert body["doc_id"] == doc_id
    assert body["indexed"] is True
    assert body["chunk_count"] == 1


def test_index_document_can_be_called_twice(tmp_path, monkeypatch):
    setup_tmp_store(tmp_path, monkeypatch)

    call_count = {"count": 0}

    def fake_index_document(doc_id: str, chunk_size: int = 800, overlap: int = 100):
        call_count["count"] += 1

        return {
            "doc_id": doc_id,
            "indexed": True,
            "chunk_count": 1,
        }

    monkeypatch.setattr(main, "index_document", fake_index_document)

    upload_response = client.post(
        "/documents/upload",
        files={
            "file": (
                "note.md",
                b"# RAG Note\n\nThis is a repeated index test.",
                "text/markdown",
            )
        },
        data={
            "doc_type": "paper",
            "tag": "RAG",
            "source": "upload",
        },
    )

    assert upload_response.status_code == 200

    doc_id = upload_response.json()["doc_id"]

    first_response = client.post(
        "/documents/index",
        json={
            "doc_id": doc_id,
            "chunk_size": 800,
            "overlap": 100,
        },
    )

    second_response = client.post(
        "/documents/index",
        json={
            "doc_id": doc_id,
            "chunk_size": 800,
            "overlap": 100,
        },
    )

    assert first_response.status_code == 200
    assert second_response.status_code == 200

    assert first_response.json()["indexed"] is True
    assert second_response.json()["indexed"] is True

    assert call_count["count"] == 2