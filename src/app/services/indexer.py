# 用户传入 doc_id
#     ↓
# DocumentStore 查 documents.json
#     ↓
# 根据 file_path 找到原始上传文件
#     ↓
# parse_document 解析文件内容
#     ↓
# chunk_text 切分成多个 chunk
#     ↓
# HashEmbeddingClient 生成向量
#     ↓
# ChromaVectorStore 写入向量库
#     ↓
# DocumentStore 保存 chunks metadata
#     ↓
# DocumentStore 更新 document 状态为 indexed
#     ↓
# 返回 doc_id、indexed、chunk_count


from pathlib import Path
from typing import Any

# 把文本变成向量
from src.app.clients.embedding_client import HashEmbeddingClient
# 创建 collection
# 写入 chunks
# 查询 top-k chunks
# 根据 doc_id 删除旧 chunks
from src.app.clients.vector_store import ChromaVectorStore
from src.app.errors import AppError
from src.app.services.chunker import chunk_text
from src.app.services.document_store import DocumentStore
from src.app.services.parser import parse_document


def index_document(
    doc_id: str,
    chunk_size: int = 800,
    overlap: int = 100,
) -> dict[str, Any]:
    store = DocumentStore()

    document = store.get_document(doc_id)

    if document is None:
        raise AppError(
            code="DOCUMENT_NOT_FOUND",
            message="文档不存在",
            retryable=False,
        )

    file_path = document.get("file_path")
    filename = document.get("filename", "")

    if not file_path or not Path(file_path).exists():
        raise AppError(
            code="DOCUMENT_FILE_NOT_FOUND",
            message="文档文件不存在",
            retryable=False,
        )

    text = parse_document(
        file_path=file_path,
        filename=filename,
    )

    if not text.strip():
        raise AppError(
            code="EMPTY_DOCUMENT_TEXT",
            message="文档解析后内容为空，无法建立索引",
            retryable=False,
        )
    # 从 document 这个字典里取出 metadata 字段；如果没有 metadata，就返回一个空字典 {}。
    document_metadata = document.get("metadata", {})

    metadata = {
        "doc_id": doc_id,
        "filename": filename,
        "doc_type": document_metadata.get("doc_type", ""),
        "tag": document_metadata.get("tag", ""),
        "source": document_metadata.get("source", ""),
    }

    chunks = chunk_text(
        text=text,
        doc_id=doc_id,
        metadata=metadata,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    if not chunks:
        raise AppError(
            code="EMPTY_CHUNKS",
            message="文档切分结果为空，无法建立索引",
            retryable=False,
        )

    texts = [chunk["text"] for chunk in chunks]

    embedding_client = HashEmbeddingClient()
    embeddings = embedding_client.embed_texts(texts)

    vector_store = ChromaVectorStore()

    # 重复索引时，先删除旧向量，再写入新向量
    vector_store.delete_by_doc_id(doc_id)
    vector_store.add_chunks(
        chunks=chunks,
        embeddings=embeddings,
    )

    store.replace_chunks(
        doc_id=doc_id,
        chunks=chunks,
    )

    store.mark_indexed(
        doc_id=doc_id,
        chunk_count=len(chunks),
    )

    return {
        "doc_id": doc_id,
        "indexed": True,
        "chunk_count": len(chunks),
    }