from pathlib import Path
from typing import Any

import chromadb

from src.app.errors import AppError


class ChromaVectorStore:
    def __init__(
        self,
        # 这是 Chroma 数据保存的本地目录，里面会有 collection 数据和 embedding 向量数据。
        persist_directory: str = "data/chroma",
        # collection_name是 Chroma 里面的 collection 名字。
        # collection 是 Chroma 里一个逻辑概念，类似数据库里的表。我们把所有的文本块都放在同一个 collection 里，方便后续查询。
        collection_name: str = "research_chunks",
    ) -> None:
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        # 创建 Chroma 持久化客户端 也就是说，Chroma 的数据会真正保存到磁盘目录里，而不是只存在内存里。
        # 这里保存路径就是：path=self.persist_directory
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        # 拿到一个 Chroma collection。如果 collection 已经存在，就直接拿出来，如果 collection 不存在，就新建一个
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name
        )
    
    # 把一批 chunk 和对应的 embedding 向量写入 Chroma 向量库。
    def add_chunks(
        self,
        chunks: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> int:
        if not chunks:
            raise AppError(
                code="EMPTY_CHUNKS",
                message="chunks 不能为空",
                retryable=False,
            )

        if len(chunks) != len(embeddings):
            raise AppError(
                code="EMBEDDING_COUNT_MISMATCH",
                message="chunks 数量和 embeddings 数量不一致",
                retryable=False,
            )
        
        # 每条数据的唯一 ID
        ids: list[str] = []
        # 每条数据的文本内容
        documents: list[str] = []
        # 每条数据的元信息
        metadatas: list[dict[str, Any]] = []

        for chunk in chunks:
            chunk_id = chunk.get("chunk_id")
            text = chunk.get("text")
            metadata = chunk.get("metadata", {})

            if not chunk_id or not isinstance(chunk_id, str):
                raise AppError(
                    code="INVALID_CHUNK",
                    message="chunk_id 缺失或不合法",
                    retryable=False,
                )

            if not text or not isinstance(text, str):
                raise AppError(
                    code="INVALID_CHUNK",
                    message="chunk text 缺失或不合法",
                    retryable=False,
                )
            
            # 清洗 metadata
            clean_metadata = self._build_metadata(chunk, metadata)

            ids.append(chunk_id)
            documents.append(text)
            metadatas.append(clean_metadata)
        
        # 这一句是真正把数据写入 Chroma 向量库
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        return len(ids)
    

    # 用一个问题的 embedding 向量去 Chroma 里检索最相似的 chunks，并返回整理后的检索结果。
    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        # metadata 过滤条件，比如只查 paper 类型文档
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        where = filters or None
        
        if not query_embedding:
            raise AppError(
                code="EMPTY_QUERY_EMBEDDING",
                message="query embedding 不能为空",
                retryable=False,
            )

        if top_k <= 0:
            raise AppError(
                code="INVALID_TOP_K",
                message="top_k 必须大于 0",
                retryable=False,
            )
        
        # 去 Chroma collection 里查相似结果
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            # 告诉 Chroma：返回结果里需要带哪些内容。
            include=["documents", "metadatas", "distances"],
        )

        return self._format_query_results(results)

    def delete_by_doc_id(self, doc_id: str) -> None:
        if not doc_id.strip():
            raise AppError(
                code="EMPTY_DOC_ID",
                message="doc_id 不能为空",
                retryable=False,
            )

        self.collection.delete(
            where={"doc_id": doc_id}
        )
    

    # 把一个 chunk 的 metadata 整理成 Chroma 可以保存的格式。
    def _build_metadata(
        self,
        chunk: dict[str, Any],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        result = {
            **metadata,
            "doc_id": chunk.get("doc_id", metadata.get("doc_id", "")),
            "chunk_id": chunk.get("chunk_id", metadata.get("chunk_id", "")),
            "chunk_index": chunk.get("chunk_index", metadata.get("chunk_index", 0)),
        }

        clean_result = {}
        
        # 如果 value 是简单类型，直接保留
        for key, value in result.items():
            if isinstance(value, (str, int, float, bool)):
                clean_result[key] = value
            elif value is None:
                continue
            # 如果是复杂类型，转成字符串
            # 比如：value = ["CLIP", "VLM"]，value = {"author": "xxx"}-》"['CLIP', 'VLM']"，"{'author': 'xxx'}"
            else:
                clean_result[key] = str(value)

        return clean_result
    

    # 把 Chroma 返回的原始查询结果，整理成我们项目里更好用的结果格式。
    # Chroma 原始结果长什么样
    # {
#     "ids": [["doc_001_chunk_0000"]],
#     "documents": [["Python and FastAPI are useful for building APIs."]],
#     "metadatas": [[
#         {
#             "doc_id": "doc_001",
#             "chunk_id": "doc_001_chunk_0000",
#             "doc_type": "note",
#             "tag": "FastAPI"
#         }
#     ]],
#     "distances": [[0.0]]
#     }
    def _format_query_results(
        self,
        results: dict[str, Any],
    ) -> list[dict[str, Any]]:
        # 现在一次只查一个 query，所以我们只取第一个[0]
        # 取出 Chroma 返回的 chunk ID 列表。
        # "ids": [["doc_001_chunk_0000", "doc_001_chunk_0001"]]
        # ids = ["doc_001_chunk_0000", "doc_001_chunk_0001"]
        # get() 更安全。如果没有 "ids" 这个字段，它会用默认值：[[]]
        ids = results.get("ids", [[]])[0] 
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        # 这句取出 Chroma 返回的距离值。距离越小，说明越相似
        distances = results.get("distances", [[]])[0]

        formatted_results: list[dict[str, Any]] = []

        for chunk_id, text, metadata, distance in zip(
            ids,
            documents,
            metadatas,
            distances,
        ):
            #score 越大越相似
            score = 1 / (1 + distance)

            formatted_results.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": metadata.get("doc_id", ""),
                    "text": text,
                    "metadata": metadata,
                    "distance": distance,
                    "score": score,
                }
            )

        return formatted_results