# TestClient 是 FastAPI 提供的测试客户端。不用真的启动 uvicorn，也能模拟 HTTP 请求。
from fastapi.testclient import TestClient

from src.app.main import app


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