from typing import Any


def pack_context(
    results: list[dict[str, Any]],
    max_context_chars: int = 4000,
    max_chunks: int = 6,
    # citation 里保留的原文片段最多 300 个字符
    snippet_chars: int = 300,
) -> dict[str, Any]:
    if max_context_chars <= 0:
        raise ValueError("max_context_chars must be positive")

    if max_chunks <= 0:
        raise ValueError("max_chunks must be positive")
    
    # 按照 score 从大到小排序。
    sorted_results = sorted(
        results,
        # 排序时，每拿到一个 item，就用 item 里的 score 作为排序依据。
        # 这里的 item 就是 results 里的每一个字典。
        key=lambda item: item.get("score", 0),
        reverse=True,
    )
    
    # 用来保存最后要拼进 context 的每一段文本。
    selected_parts: list[str] = []
    # 用来保存最终返回的 citations。
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
        
        # 在当前剩余空间里，扣掉 header 占用的字符，再扣掉换行符占用的字符，剩下的才是真正能放 chunk 正文的字符数。
        available_text_chars = remaining_chars - len(header) - 2

        if available_text_chars <= 0:
            break
        
        # 如果当前 chunk 正文太长，超过了还能放进 context 的最大字符数，就把它截断到允许长度以内。
        if len(chunk_text) > available_text_chars:
            chunk_text = truncate_text(chunk_text, available_text_chars)
        
        # 把当前 chunk 整理成一段可以放进 prompt 的上下文。
        # header = "[chunk_id=chunk_1 | doc_id=doc_1 | filename=rag.md | score=0.9000]"
        # chunk_text = "RAG 的基本流程包括上传、解析、切分、embedding、检索和回答生成。"
        context_part = f"{header}\n{chunk_text}"

        selected_parts.append(context_part)
        current_length += len(context_part) + 2
        seen_keys.add(unique_key)
        
        # 保存 citation，也就是引用信息。
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