from typing import Any


def pack_context(
    results: list[dict[str, Any]],
    max_context_chars: int = 4000,
    max_chunks: int = 6,
    snippet_chars: int = 300,
) -> dict[str, Any]:
    if max_context_chars <= 0:
        raise ValueError("max_context_chars must be positive")

    if max_chunks <= 0:
        raise ValueError("max_chunks must be positive")
    
    # 按照 score 从大到小排序。
    sorted_results = sorted(
        results,
        key=lambda item: item.get("score", 0),
        reverse=True,
    )

    selected_parts: list[str] = []
    selected_citations: list[dict[str, Any]] = []
    # 记录已经放进 context 的 chunk。
    seen_keys: set[str] = set()
    current_length = 0

    for item in sorted_results:
        doc_id = str(item.get("doc_id", ""))
        chunk_id = str(item.get("chunk_id", ""))
        filename = str(item.get("filename", ""))
        text = str(item.get("text", "")).strip()
        score = float(item.get("score", 0))

        if not doc_id or not chunk_id or not text:
            continue

        unique_key = f"{doc_id}:{chunk_id}"

        if unique_key in seen_keys:
            continue

        if len(selected_citations) >= max_chunks:
            break

        remaining_chars = max_context_chars - current_length

        if remaining_chars <= 0:
            break

        header = (
            f"[chunk_id={chunk_id} | "
            f"doc_id={doc_id} | "
            f"filename={filename} | "
            f"score={score:.4f}]"
        )

        chunk_text = text

        available_text_chars = remaining_chars - len(header) - 2

        if available_text_chars <= 0:
            break

        if len(chunk_text) > available_text_chars:
            chunk_text = truncate_text(chunk_text, available_text_chars)

        context_part = f"{header}\n{chunk_text}"

        selected_parts.append(context_part)
        current_length += len(context_part) + 2
        seen_keys.add(unique_key)

        selected_citations.append(
            {
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "filename": filename,
                "text_snippet": truncate_text(text, snippet_chars),
                "score": score,
            }
        )

    return {
        "context": "\n\n".join(selected_parts),
        "citations": selected_citations,
    }


def truncate_text(text: str, max_chars: int) -> str:
    cleaned_text = text.strip()

    if max_chars <= 0:
        return ""

    if len(cleaned_text) <= max_chars:
        return cleaned_text

    if max_chars <= 3:
        return cleaned_text[:max_chars]

    return cleaned_text[: max_chars - 3].rstrip() + "..."