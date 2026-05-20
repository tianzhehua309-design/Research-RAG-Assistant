import re
from typing import Any

from src.app.errors import AppError

"""
        清洗 chunk 内部文本：
        1. 统一换行符
        2. 去掉每行首尾空白
        3. 压缩连续空行
"""
def normalize_chunk_text(text: str) -> str:
    text = text.replace("\r\n","\n").replace("\r","\n")

    lines: list[str] = []
    for line in text.split("\n"):
        line = re.sub(r"[ \u3000]+", " ", line)
        line = line.strip()
        lines.append(line)

    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}","\n\n",cleaned)
    return cleaned.strip()


"""
    按段落切分文本。
    空行通常表示一个段落结束。
"""
def split_into_paragraphs(text: str) -> list[str]:
    cleaned = normalize_chunk_text(text)

    if not cleaned:
        return []

    paragraphs = re.split(r"\n\s*\n", cleaned)

    return [p.strip() for p in paragraphs if p.strip()]


"""
    如果单个段落太长，就用滑动窗口切分。

    例如：
    chunk_size=800
    overlap=100

    第一个 chunk: 0 到 800
    第二个 chunk: 700 到 1500
    第三个 chunk: 1400 到 2200
"""
def split_long_text(
        text: str,
        chunk_size: int,
        overlap: int,
) ->list[str]:
    if chunk_size <=0:
        raise AppError(
            code="INVALID_CHUNK_SIZE",
            message="chunk_size 必须大于 0",
            retryable=False,
        )

    if overlap < 0:
        raise AppError(
            code="INVALID_OVERLAP",
            message="overlap 不能小于 0",
            retryable=False,
        )

    if overlap >= chunk_size:
        raise AppError(
            code="INVALID_OVERLAP",
            message="overlap 必须小于 chunk_size",
            retryable=False,
        )

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end -overlap

    return chunks


"""
    把完整文档文本切成多个 chunk。

    处理策略：
    1. 先按段落切分
    2. 尽量把多个短段落合并到一个 chunk
    3. 如果单段太长，就用滑动窗口继续切
    4. 每个 chunk 都带 doc_id、chunk_id、chunk_index、metadata
"""
def chunk_text(
        text: str,
        doc_id: str,
        metadata: dict[str, Any],
        chunk_size:int = 800,
        overlap:int = 100,
) -> list[dict[str, Any]]:
    cleaned_text = normalize_chunk_text(text)

    if not cleaned_text:
        raise AppError(
            code="EMPTY_TEXT",
            message="待切分文本不能为空",
            retryable=False,
        )

    if not doc_id.strip():
        raise AppError(
            code="EMPTY_DOC_ID",
            message="doc_id 不能为空",
            retryable=False,
        )

    if chunk_size <= 0:
        raise AppError(
            code="INVALID_CHUNK_SIZE",
            message="chunk_size 必须大于 0",
            retryable=False,
        )

    if overlap < 0 or overlap >= chunk_size:
        raise AppError(
            code="INVALID_OVERLAP",
            message="overlap 必须大于等于 0 且小于 chunk_size",
            retryable=False,
        )

    paragraphs = split_into_paragraphs(cleaned_text)

    raw_chunks: list[str] = []
    current_parts: list[str] = []
    current_length = 0

    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            if current_parts:
                raw_chunks.append("\n\n".join(current_parts).strip())
                current_parts = []
                current_length = 0

            raw_chunks.extend(
                split_long_text(
                    paragraph,
                    chunk_size = chunk_size,
                    overlap = overlap,
                )
            )
            continue

        # 段落之间用两个换行连接，所以额外加 2
        extra_length = 2 if current_parts else 0
        next_length = current_length + extra_length + len(paragraph)

        if next_length <= chunk_size:
            current_parts.append(paragraph)
        else:
            if current_parts:
                raw_chunks.append("\n\n".join(current_parts).strip())

            current_parts = [paragraph]
            current_length = len(paragraph)

    if  current_parts:
        raw_chunks.append("\n\n".join(current_parts).strip())

    chunks: list[dict[str, Any]] = []

    for index,chunk in enumerate(raw_chunks):
        chunk_id = f"{doc_id}_chunk_{index:04d}"

        chunks.append({
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "chunk_index": index,
            "text": chunk,
            "metadata": {
                **metadata,
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "chunk_index": index,
            },
        })

    return chunks





