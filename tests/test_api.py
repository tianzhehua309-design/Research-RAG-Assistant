# TestClient 是 FastAPI 提供的测试客户端。不用真的启动 uvicorn，也能模拟 HTTP 请求。
from fastapi.testclient import TestClient

from src.app.main import app
from src.app import main


client = TestClient(app)


def test_health_ok():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert "X-Request-ID" in response.headers


def test_debug_error_returns_unified_error():
    response = client.get("/debug/error")

    assert response.status_code == 400
    assert "X-Request-ID" in response.headers

    body = response.json()
    assert body["error"]["code"] == "DEBUG_ERROR"
    assert body["error"]["message"] == "这是一个用于测试统一错误返回的调试错误"
    assert body["error"]["retryable"] is False


def test_not_found_still_has_request_id():
    response = client.get("/not-found")

    assert response.status_code == 404
    assert "X-Request-ID" in response.headers


def test_search_chunks_success(monkeypatch):
    def fake_search_chunks(query, top_k=5, filters=None):
        return [
            {
                "chunk_id": "doc_test_chunk_0000",
                "doc_id": "doc_test",
                "text": "CLIP shows vulnerability under adversarial perturbations.",
                "metadata": {
                    "doc_id": "doc_test",
                    "chunk_id": "doc_test_chunk_0000",
                    "doc_type": "paper",
                    "tag": "VLM",
                    "source": "upload",
                },
                "distance": 0.25,
                "score": 0.8,
            }
        ]

    monkeypatch.setattr(main, "search_chunks", fake_search_chunks)

    response = client.post(
        "/search/chunks",
        json={
            "query": "CLIP 的对抗鲁棒性怎么样？",
            "top_k": 3,
            "filters": {
                "doc_type": "paper",
                "tag": "VLM",
            },
        },
    )

    assert response.status_code == 200
    assert "X-Request-ID" in response.headers

    body = response.json()

    assert body["query"] == "CLIP 的对抗鲁棒性怎么样？"
    assert body["top_k"] == 3
    assert body["filters"]["doc_type"] == "paper"
    assert len(body["results"]) == 1
    assert body["results"][0]["chunk_id"] == "doc_test_chunk_0000"
    assert body["results"][0]["score"] == 0.8


def test_search_chunks_passes_filters(monkeypatch):
    captured = {}

    def fake_search_chunks(query, top_k=5, filters=None):
        captured["query"] = query
        captured["top_k"] = top_k
        captured["filters"] = filters
        return []

    monkeypatch.setattr(main, "search_chunks", fake_search_chunks)

    response = client.post(
        "/search/chunks",
        json={
            "query": "PGD 参数在哪里？",
            "top_k": 5,
            "filters": {
                "doc_type": "experiment",
                "tag": "PGD",
                "source": "upload",
            },
        },
    )

    assert response.status_code == 200
    assert captured["query"] == "PGD 参数在哪里？"
    assert captured["top_k"] == 5
    assert captured["filters"] == {
        "doc_type": "experiment",
        "tag": "PGD",
        "source": "upload",
    }


def test_search_chunks_rejects_invalid_top_k():
    response = client.post(
        "/search/chunks",
        json={
            "query": "CLIP",
            "top_k": 0,
        },
    )

    assert response.status_code == 422
    assert "X-Request-ID" in response.headers

    body = response.json()
    assert body["error"]["code"] == "INVALID_REQUEST"

def test_search_chunks_success(monkeypatch):
    def fake_search_chunks(query, top_k=5, filters=None):
        return [
            {
                "chunk_id": "doc_test_chunk_0000",
                "doc_id": "doc_test",
                "text": "RAG includes document upload, chunking, embedding, retrieval, and citations.",
                "metadata": {
                    "doc_id": "doc_test",
                    "chunk_id": "doc_test_chunk_0000",
                    "doc_type": "paper",
                    "tag": "RAG",
                    "source": "upload",
                    "chunk_index": 0,
                },
                "distance": 1.0,
                "score": 0.5,
            }
        ]

    monkeypatch.setattr(main, "search_chunks", fake_search_chunks)

    response = client.post(
        "/search/chunks",
        json={
            "query": "RAG 的基本流程是什么？",
            "top_k": 5,
            "filters": {
                "doc_type": "paper",
                "tag": "RAG",
            },
        },
    )

    assert response.status_code == 200
    assert "X-Request-ID" in response.headers

    body = response.json()

    assert body["query"] == "RAG 的基本流程是什么？"
    assert body["top_k"] == 5
    assert body["filters"] == {
        "doc_type": "paper",
        "tag": "RAG",
    }
    assert len(body["results"]) == 1
    assert body["results"][0]["chunk_id"] == "doc_test_chunk_0000"
    assert body["results"][0]["score"] == 0.5