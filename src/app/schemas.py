from typing import Literal

from pydantic import BaseModel,Field

# 基础响应
class HealthResponse(BaseModel):
    # 表示 /health 返回的 JSON 里必须有一个字符串字段 status。
    status: str

# 错误响应
class ErrorDetail(BaseModel):
    code: str = Field(...,examples=["EMPTY_FILE"])
    message: str = Field(...,examples=["上传文件不能为空"])
    retryable: bool = Field(...,examples=["False"])

class ErrorDetailResponse(BaseModel):
    error : ErrorDetail
# Literal 这表示 doc_type 只能是这几个值：paper,experiment,meeting,note,other
DocType = Literal["paper","experiment","metting","note","other"]
SourceType = Literal["upload","sample"]
DocumentStatus = Literal["uploaded","indexed","failed"]

# Document 和 Chunk
# DocumentMetadata表示文档元数据。
class DocumentMetadata(BaseModel):
    doc_type: DocType = Field(
        default="other",
        description="Document type, such as paper, experiment, meeting, note, or other.",
        examples=["paper"],
    )
    source: SourceType = Field(
        default="upload",
        description="Document source.",
        examples=["upload"],
    )
    tag: str = Field(
        default="general",
        description="User-defined tag, such as VLM, CLIP, RAG.",
        examples=["VLM"],
    )

# DocumentInfo表示一整篇文档。
class DocumentInfo(BaseModel):
    doc_id:str = Field(...,examples=["doc_001"])
    filename:str = Field(...,examples=["clip_robustness.md"])
    metadata: DocumentMetadata
    created_at: str = Field(..., examples=["2026-05-15T10:00:00Z"])
    status: DocumentStatus = Field(default="uploaded", examples=["uploaded"])
    chunk_count: int = Field(default=0, ge=0, examples=[0])

# ChunkInfo表示文档切出来的一小段。
class ChunkInfo(BaseModel):
    doc_id: str = Field(..., examples=["doc_001"])
    chunk_id: str = Field(..., examples=["doc_001_chunk_0001"])
    chunk_index: int = Field(..., ge=0, examples=[0])
    text: str = Field(..., examples=["CLIP shows vulnerability under adversarial perturbations..."])
    metadata: DocumentMetadata

class UploadResponse(BaseModel):
    doc_id: str = Field(..., examples=["doc_001"])
    filename: str = Field(..., examples=["clip_robustness.md"])
    metadata: DocumentMetadata
    status: DocumentStatus = Field(default="uploaded", examples=["uploaded"])

class IndexRequest(BaseModel):
    doc_id: str = Field(..., examples=["doc_001"])

class IndexResponse(BaseModel):
    doc_id: str = Field(..., examples=["doc_001"])
    indexed: bool = Field(..., examples=[True])
    chunk_count: int = Field(..., ge=0, examples=[12])


class SearchFilters(BaseModel):
    doc_type: DocType | None = Field(default=None, examples=["paper"])
    source: SourceType | None = Field(default=None, examples=["upload"])
    tag: str | None = Field(default=None, examples=["VLM"])

# 表示用户搜索时传什么。
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, examples=["CLIP 的对抗鲁棒性怎么样？"])
    # 默认 top_k = 5
    # 最小值是 1
    # 最大值是 20
    top_k: int = Field(default=5, ge=1, le=20, examples=[5])
    # filters 可以是 SearchFilters
    # 也可以不传
    filters: SearchFilters | None = None


class SearchResult(BaseModel):
    doc_id: str = Field(..., examples=["doc_001"])
    chunk_id: str = Field(..., examples=["doc_001_chunk_0003"])
    filename: str = Field(..., examples=["clip_robustness.md"])
    text: str = Field(..., examples=["CLIP shows vulnerability under adversarial perturbations..."])
    score: float = Field(..., ge=0, examples=[0.82])
    metadata: DocumentMetadata

# 表示搜索后返回什么。
class SearchResponse(BaseModel):
    query: str = Field(..., examples=["CLIP 的对抗鲁棒性怎么样？"])
    results: list[SearchResult]

# 表示回答引用了哪些来源。
class Citation(BaseModel):
    doc_id: str = Field(..., examples=["doc_001"])
    chunk_id: str = Field(..., examples=["doc_001_chunk_0003"])
    filename: str = Field(..., examples=["clip_robustness.md"])
    text_snippet: str = Field(..., examples=["CLIP shows vulnerability under adversarial perturbations..."])
    score: float = Field(..., ge=0, examples=[0.82])

# 表示用户问答请求。
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, examples=["这篇论文的方法核心是什么？"])
    top_k: int = Field(default=5, ge=1, le=20, examples=[5])
    filters: SearchFilters | None = None

# 表示系统回答。
class AskResponse(BaseModel):
    answer: str = Field(..., examples=["根据检索到的文档，这篇论文主要讨论了……"])
    citations: list[Citation]