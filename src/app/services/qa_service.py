from typing import Any

from src.app.schemas import AskRequest, AskResponse, Citation
from src.app.services.context_packer import pack_context
from src.app.services.retriever import search_chunks as retrieve_chunks


def answer_question(
    payload: AskRequest,
    request_id: str | None = None,
) -> AskResponse:
    filters = None

    if payload.filters is not None:
        if hasattr(payload.filters, "model_dump"):
            filters = payload.filters.model_dump(exclude_none=True)
        else:
            filters = payload.filters

    search_results = retrieve_chunks(
        query=payload.question,
        top_k=payload.top_k,
        filters=filters,
    )

    packed = pack_context(
        results=search_results,
        max_context_chars=payload.max_context_chars,
    )

    context = packed["context"]
    selected_citations = packed["citations"]

    if not selected_citations:
        return AskResponse(
            question=payload.question,
            answer="没有在已索引文档中找到与该问题相关的依据。",
            citations=[],
        )

    answer = build_mock_answer(
        question=payload.question,
        context=context,
        citations=selected_citations,
    )

    return AskResponse(
        question=payload.question,
        answer=answer,
        citations=[
            Citation(**citation)
            for citation in selected_citations
        ],
    )


def build_mock_answer(
    question: str,
    context: str,
    citations: list[dict[str, Any]],
) -> str:
    citation_count = len(citations)

    top_sources = [
        f"{item['filename']} / {item['chunk_id']}"
        for item in citations[:3]
    ]

    source_text = "；".join(top_sources)

    return (
        f"根据检索到的 {citation_count} 个相关片段，系统找到了与问题相关的资料。\n\n"
        f"问题：{question}\n\n"
        f"主要依据来自：{source_text}。\n\n"
        "当前版本使用 mock answer，重点是验证 RAG 链路已经打通："
        "检索结果可以被组装成上下文，并且 citations 可以追溯到具体文档和 chunk。"
    )