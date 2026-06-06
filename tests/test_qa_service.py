from src.app.schemas import AskRequest
from src.app.services.qa_service import answer_question


def test_answer_question_with_citations(monkeypatch):
    def fake_retrieve_chunks(query, top_k, filters=None):
        return [
            {
                "doc_id": "doc_001",
                "chunk_id": "doc_001_chunk_0001",
                "filename": "rag_note.md",
                "text": "RAG 的基本流程包括文档上传、chunking、embedding、检索和回答生成。",
                "score": 0.9,
                "metadata": {
                    "doc_type": "note",
                    "source": "upload",
                    "tag": "RAG",
                },
            }
        ]

    def fake_pack_context(results, max_context_chars=4000):
        return {
            "context": "[chunk_id=doc_001_chunk_0001]\nRAG 的基本流程包括文档上传、chunking、embedding、检索和回答生成。",
            "citations": [
                {
                    "doc_id": "doc_001",
                    "chunk_id": "doc_001_chunk_0001",
                    "filename": "rag_note.md",
                    "text_snippet": "RAG 的基本流程包括文档上传、chunking、embedding、检索和回答生成。",
                    "score": 0.9,
                }
            ],
        }

    monkeypatch.setattr(
        "src.app.services.qa_service.retrieve_chunks",
        fake_retrieve_chunks,
    )
    monkeypatch.setattr(
        "src.app.services.qa_service.pack_context",
        fake_pack_context,
    )

    payload = AskRequest(
        question="RAG 的基本流程是什么？",
        top_k=5,
    )

    response = answer_question(payload)

    assert response.question == "RAG 的基本流程是什么？"
    assert "检索到的" in response.answer
    assert len(response.citations) == 1
    assert response.citations[0].doc_id == "doc_001"
    assert response.citations[0].chunk_id == "doc_001_chunk_0001"


def test_answer_question_without_results(monkeypatch):
    def fake_retrieve_chunks(query, top_k, filters=None):
        return []

    def fake_pack_context(results, max_context_chars=4000):
        return {
            "context": "",
            "citations": [],
        }

    monkeypatch.setattr(
        "src.app.services.qa_service.retrieve_chunks",
        fake_retrieve_chunks,
    )
    monkeypatch.setattr(
        "src.app.services.qa_service.pack_context",
        fake_pack_context,
    )

    payload = AskRequest(
        question="不存在的问题",
        top_k=5,
    )

    response = answer_question(payload)

    assert response.question == "不存在的问题"
    assert response.citations == []
    assert "没有在已索引文档中找到" in response.answer