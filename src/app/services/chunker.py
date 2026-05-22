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
    # 按行遍历文本
    for line in text.split("\n"):
        # 合并连续空格
        # r"[ \u3000]+" 匹配一个或多个空格（包括全角空格）
        line = re.sub(r"[ \u3000]+", " ", line)
        line = line.strip()
        lines.append(line)
    # 这句把列表里的每一行重新用 \n 拼起来。
    cleaned = "\n".join(lines)
    # 如果出现 3 个或更多连续换行，就压缩成 2 个换行
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
    # 按空行切分段落
    # r"\n\s*\n"
    # \n	一个换行
    # \s*	0 个或多个空白字符
    # \n	又一个换行
    # 匹配两个换行，中间可以夹着空格、tab 等空白字符
    paragraphs = re.split(r"\n\s*\n", cleaned)
    result = []

    for p in paragraphs:
        if p.strip():
            result.append(p.strip())

    return result


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

        start = end - overlap

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

    # 表示已经完成的 chunk 文本列表。
    raw_chunks: list[str] = []
    # 表示当前正在组装的 chunk 里面有哪些段落。
    current_parts: list[str] = []
    # 表示当前正在组装的 chunk 已经有多长。
    current_length = 0

    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            # 先把前面正在组装的 chunk 保存下来
            if current_parts:
                # 如果 current_parts 里面已经攒了一些短段落，先把它们拼成一个完整 chunk，放进 raw_chunks。
                raw_chunks.append("\n\n".join(current_parts).strip())
                current_parts = []
                current_length = 0

            # 长段落单独切成多个 chunk
            # append raw_chunks.append(["chunk1", "chunk2"])-》[["chunk1", "chunk2"]]
            # extend raw_chunks.extend(["chunk1", "chunk2"])-》["chunk1", "chunk2"]
            raw_chunks.extend(
                split_long_text(
                    paragraph,
                    chunk_size = chunk_size,
                    overlap = overlap,
                )
            )
            continue

        # 段落之间用两个换行连接，所以额外加 2
        # 如果 current_parts 里已经有段落了，那么再拼接新段落时，中间要加两个换行符 \n\n，所以额外长度是 2。
        extra_length = 2 if current_parts else 0
        # 计算加入当前段落后的总长度
        next_length = current_length + extra_length + len(paragraph)

        if next_length <= chunk_size:
            current_parts.append(paragraph)
            current_length = next_length
        # 如果超过 chunk_size，就保存旧 chunk，开启新 chunk
        else:
            if current_parts:
                raw_chunks.append("\n\n".join(current_parts).strip())

            current_parts = [paragraph]
            current_length = len(paragraph)
    
    # 循环结束后，还需要把最后一个 current_parts 保存进去。
    # 因为最后一个正在组装的 chunk，不一定在循环里触发了 else 保存。
    # 否则最后一组段落会丢掉。
    if  current_parts:
        raw_chunks.append("\n\n".join(current_parts).strip())


    # 把前面已经切好的 raw_chunks，
    # 包装成带 doc_id、chunk_id、chunk_index、metadata 的标准 chunk 结构。
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





