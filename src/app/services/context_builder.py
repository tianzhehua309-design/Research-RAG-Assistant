from typing import Any

from src.app.schemas import Citation, SearchResult


def build_context_and_citations(
    results: list[SearchResult],
    max_snippet_chars: int = 300,
) -> tuple[str, list[Citation]]:
    # 一段 context 文本，给 LLM 看
    context_blocks: list[str] = []
    # 一个 citation 对象，给用户看
    citations: list[Citation] = []

    for index, result in enumerate(results, start=1):
        text = result.text.strip()

        if not text:
            continue
        
        # 兼容 metadata 是 dict 或 Pydantic 对象
        source = get_metadata_value(result.metadata, "source", "")
        filename = result.filename or get_metadata_value(result.metadata, "filename", "")

        context_block = build_context_block(
            index=index,
            result=result,
            filename=filename,
            source=source,
            text=text,
        )

        context_blocks.append(context_block)

        citations.append(
            Citation(
                doc_id=result.doc_id,
                chunk_id=result.chunk_id,
                filename=filename,
                text_snippet=build_text_snippet(
                    text=text,
                    max_chars=max_snippet_chars,
                ),
                score=result.score,
            )
        )

    context = "\n\n".join(context_blocks)

    return context, citations


def build_context_block(
    index: int,
    result: SearchResult,
    filename: str,
    source: str,
    text: str,
) -> str:
    header = (
        f"[{index}] "
        f"doc_id={result.doc_id} | "
        f"chunk_id={result.chunk_id} | "
        f"filename={filename} | "
        f"source={source} | "
        f"score={result.score:.4f}"
    )

    return f"{header}\n{text}"

# citation 只是展示引用片段，不应该把整个 chunk 都塞进去。
# 比如一个 chunk 有 2000 字，citation 只保留前 300 字即可。
def build_text_snippet(text: str, max_chars: int) -> str:
    cleaned_text = text.strip()

    if len(cleaned_text) <= max_chars:
        return cleaned_text

    return cleaned_text[:max_chars].rstrip() + "..."


def get_metadata_value(
    metadata: Any,
    key: str,
    default: str = "",
) -> str:
    if isinstance(metadata, dict):
        value = metadata.get(key, default)
    else:
        value = getattr(metadata, key, default)

    if value is None:
        return default

    return str(value)