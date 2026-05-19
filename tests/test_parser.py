import pytest

from src.app.errors import AppError
from src.app.services import parser
from src.app.services.parser import normalize_text, parse_document


def test_parse_txt_success(tmp_path):
    file_path = tmp_path / "note.txt"
    file_path.write_text(
        "  This   is a RAG note.\n\n\nIt supports FastAPI.  ",
        encoding="utf-8",
    )

    text = parse_document(str(file_path), filename="note.txt")

    assert "This is a RAG note." in text
    assert "It supports FastAPI." in text
    assert "\n\n\n" not in text


def test_parse_markdown_success(tmp_path):
    file_path = tmp_path / "note.md"
    file_path.write_text(
        "# Title\n\nThis is **important** for [RAG](https://example.com).\n\n- item one",
        encoding="utf-8",
    )

    text = parse_document(str(file_path), filename="note.md")

    assert "Title" in text
    assert "important" in text
    assert "RAG" in text
    assert "item one" in text
    assert "#" not in text
    assert "**" not in text
    assert "https://example.com" not in text


def test_parse_empty_txt_raises_app_error(tmp_path):
    file_path = tmp_path / "empty.txt"
    file_path.write_text("   \n\n   ", encoding="utf-8")

    with pytest.raises(AppError) as exc_info:
        parse_document(str(file_path), filename="empty.txt")

    assert exc_info.value.code == "EMPTY_DOCUMENT_TEXT"
    assert exc_info.value.retryable is False


def test_parse_unsupported_file_type(tmp_path):
    file_path = tmp_path / "data.csv"
    file_path.write_text("a,b,c", encoding="utf-8")

    with pytest.raises(AppError) as exc_info:
        parse_document(str(file_path), filename="data.csv")

    assert exc_info.value.code == "UNSUPPORTED_FILE_TYPE"


def test_parse_file_not_found():
    with pytest.raises(AppError) as exc_info:
        parse_document("not_exists.txt", filename="not_exists.txt")

    assert exc_info.value.code == "FILE_NOT_FOUND"


def test_normalize_text():
    raw = "  hello    world \r\n\r\n\r\n  RAG\tassistant  "
    text = normalize_text(raw)

    assert text == "hello world\n\nRAG assistant"


def test_parse_pdf_success_with_mock(tmp_path, monkeypatch):
    file_path = tmp_path / "paper.pdf"
    file_path.write_bytes(b"fake pdf bytes")

    class FakePage:
        def extract_text(self):
            return "This is a PDF paper about CLIP robustness."

    class FakeReader:
        pages = [FakePage()]

    monkeypatch.setattr(parser, "PdfReader", lambda path: FakeReader())

    text = parse_document(str(file_path), filename="paper.pdf")

    assert "Page 1" in text
    assert "CLIP robustness" in text