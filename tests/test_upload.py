from fastapi.testclient import TestClient

from src.app import main


client = TestClient(main.app)

# tmp_path 是 pytest 提供的临时目录。
# 测试时不应该真的把文件写到：data/uploads/
# 否则测试会污染你的项目目录。所以我们用临时目录。
# monkeypatch 临时修改代码里的变量，比如把 UPLOAD_DIR 改成临时目录
def test_upload_markdown_success(tmp_path, monkeypatch):
    # 在这个测试运行期间，把 main.UPLOAD_DIR 临时替换成 tmp_path
    monkeypatch.setattr(main, "UPLOAD_DIR", tmp_path)

    response = client.post(
        "/documents/upload",
        # files这模拟用户上传文件。
        files={
            "file": (
                "note.md",
                b"# Title\n\nThis is a RAG note.",
                "text/markdown",
            )
        },
        data={
            "doc_type": "note",
            "tag": "RAG",
            "source": "upload",
        },
    )

    assert response.status_code == 200
    assert "X-Request-ID" in response.headers

    body = response.json()
    assert body["doc_id"].startswith("doc_")
    assert body["filename"] == "note.md"
    assert body["metadata"]["doc_type"] == "note"
    assert body["metadata"]["tag"] == "RAG"
    assert body["metadata"]["source"] == "upload"
    assert body["status"] == "uploaded"

    saved_files = list(tmp_path.iterdir())
    assert len(saved_files) == 1
    assert saved_files[0].name.endswith("_note.md")


def test_upload_rejects_invalid_file_type(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "UPLOAD_DIR", tmp_path)

    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "bad.exe",
                b"fake exe content",
                "application/octet-stream",
            )
        },
        data={
            "doc_type": "note",
            "tag": "RAG",
            "source": "upload",
        },
    )

    assert response.status_code == 400
    assert "X-Request-ID" in response.headers

    body = response.json()
    assert body["error"]["code"] == "INVALID_FILE_TYPE"
    assert body["error"]["retryable"] is False


def test_upload_rejects_empty_file(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "UPLOAD_DIR", tmp_path)

    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "empty.md",
                b"",
                "text/markdown",
            )
        },
        data={
            "doc_type": "note",
            "tag": "RAG",
            "source": "upload",
        },
    )

    assert response.status_code == 400
    assert "X-Request-ID" in response.headers

    body = response.json()
    assert body["error"]["code"] == "EMPTY_FILE"
    assert body["error"]["retryable"] is False


def test_upload_rejects_invalid_doc_type(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "UPLOAD_DIR", tmp_path)

    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "note.md",
                b"# Title",
                "text/markdown",
            )
        },
        data={
            "doc_type": "wrong_type",
            "tag": "RAG",
            "source": "upload",
        },
    )

    assert response.status_code == 422
    assert "X-Request-ID" in response.headers

    body = response.json()
    assert body["error"]["code"] == "INVALID_REQUEST"