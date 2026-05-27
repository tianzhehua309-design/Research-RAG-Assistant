import json
from pathlib import Path
from typing import Any

from src.app.errors import AppError


class DocumentStore:
    # metadata_dir 就是 metadata 文件保存的目录路径
    def __init__(self, metadata_dir: str | Path = "data/metadata") -> None:
        self.metadata_dir = Path(metadata_dir)
        # 如果 data/metadata 不存在，就自动创建
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        self.documents_path = self.metadata_dir / "documents.json"
        self.chunks_path = self.metadata_dir / "chunks.json"

        self._ensure_json_file(self.documents_path, default={})
        self._ensure_json_file(self.chunks_path, default={})

    # 检查某个 JSON 文件是否存在，如果不存在，就创建它，并写入一个默认内容
    def _ensure_json_file(self, path: Path, default: Any) -> None:
        if not path.exists():
            self._write_json(path, default)

    # 安全地读取一个 JSON 文件
    def _read_json(self, path: Path, default: Any) -> Any:
        if not path.exists():
            return default

        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            raise AppError(
                code="METADATA_FILE_INVALID",
                message=f"{path.name} 不是合法 JSON 文件",
                retryable=False,
            )

    # 把 Python 数据写入到 JSON 文件里
    def _write_json(self, path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # 把一篇 document 的 metadata 保存到 documents.json 文件里
    def save_document(self, document: dict[str, Any]) -> dict[str, Any]:
        doc_id = document.get("doc_id")

        if not doc_id:
            raise AppError(
                code="MISSING_DOC_ID",
                message="document 缺少 doc_id",
                retryable=False,
            )

        documents = self._read_json(self.documents_path, default={})
        documents[doc_id] = document
        self._write_json(self.documents_path, documents)

        return document

    # 根据 doc_id 查找某一篇文档
    def get_document(self, doc_id: str) -> dict[str, Any]:
        documents = self._read_json(self.documents_path, default={})

        if doc_id not in documents:
            raise AppError(
                code="DOCUMENT_NOT_FOUND",
                message=f"文档不存在：{doc_id}",
                retryable=False,
            )

        return documents[doc_id]

    # 返回所有文档列表
    def list_documents(self) -> list[dict[str, Any]]:
        documents = self._read_json(self.documents_path, default={})
        return list(documents.values())

    # 保存某篇文档的 chunks
    def save_chunks(
        self,
        doc_id: str,
        chunks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not doc_id:
            raise AppError(
                code="MISSING_DOC_ID",
                message="保存 chunks 时缺少 doc_id",
                retryable=False,
            )

        chunk_store = self._read_json(self.chunks_path, default={})
        chunk_store[doc_id] = chunks
        self._write_json(self.chunks_path, chunk_store)

        return chunks

    # 根据 doc_id 获取 chunks
    def get_chunks_by_doc_id(self, doc_id: str) -> list[dict[str, Any]]:
        chunk_store = self._read_json(self.chunks_path, default={})
        return chunk_store.get(doc_id, [])

    # 删除文档，同时删除它对应的 chunks
    def delete_document(self, doc_id: str) -> None:
        documents = self._read_json(self.documents_path, default={})
        chunk_store = self._read_json(self.chunks_path, default={})

        if doc_id not in documents:
            raise AppError(
                code="DOCUMENT_NOT_FOUND",
                message=f"文档不存在：{doc_id}",
                retryable=False,
            )

        documents.pop(doc_id, None)
        chunk_store.pop(doc_id, None)

        self._write_json(self.documents_path, documents)
        self._write_json(self.chunks_path, chunk_store)

    # 重复索引时，先删除旧 chunks，再保存新 chunks
    def replace_chunks(
        self,
        doc_id: str,
        chunks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not doc_id:
            raise AppError(
                code="MISSING_DOC_ID",
                message="替换 chunks 时缺少 doc_id",
                retryable=False,
            )

        chunk_store = self._read_json(self.chunks_path, default={})
        chunk_store[doc_id] = chunks
        self._write_json(self.chunks_path, chunk_store)

        return chunks

    # 把 document 状态改成 indexed，并记录 chunk 数量
    def mark_indexed(self, doc_id: str, chunk_count: int) -> dict[str, Any]:
        documents = self._read_json(self.documents_path, default={})

        if doc_id not in documents:
            raise AppError(
                code="DOCUMENT_NOT_FOUND",
                message=f"文档不存在：{doc_id}",
                retryable=False,
            )

        document = documents[doc_id]
        document["status"] = "indexed"
        document["indexed"] = True
        document["chunk_count"] = chunk_count

        documents[doc_id] = document
        self._write_json(self.documents_path, documents)

        return document