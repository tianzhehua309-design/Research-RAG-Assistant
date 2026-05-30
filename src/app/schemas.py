from typing import Literal,Any

from pydantic import BaseModel,Field,field_validator

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

# 上传文档后的返回格式
class UploadResponse(BaseModel):
    doc_id: str = Field(..., examples=["doc_001"])
    filename: str = Field(..., examples=["clip_robustness.md"])
    metadata: DocumentMetadata
    status: DocumentStatus = Field(default="uploaded", examples=["uploaded"])

# 建立索引时的请求格式
class IndexRequest(BaseModel):
    doc_id: str = Field(..., examples=["doc_001"])

# 建立索引后的返回格式
class IndexResponse(BaseModel):
    doc_id: str = Field(..., examples=["doc_001"])
    indexed: bool = Field(..., examples=[True])
    # 这篇文档被切成了多少个 chunk
    chunk_count: int = Field(..., ge=0, examples=[12])

# 检索时的过滤条件
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
    # 检索时返回最相关的前 k 个 chunk
    top_k: int = Field(default=5, ge=1, le=20, examples=[5])
    # filters 可以是 SearchFilters
    # 也可以不传
    filters: SearchFilters | None = None

# 一次检索返回的单条结果格式。
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

class IndexRequest(BaseModel):
    doc_id: str = Field(..., examples=["doc_123"])
    chunk_size: int = Field(default=800, ge=100, le=3000)
    overlap: int = Field(default=100, ge=0, le=1000)

class IndexResponse(BaseModel):
    doc_id: str
    indexed: bool
    chunk_count: int

# 搜索时的过滤条件。
class SearchFilters(BaseModel):
    doc_type: str | None = Field(
        default=None,
        examples=["paper"],
        description="Filter by document type, such as paper, experiment, or meeting.",
    )
    tag: str | None = Field(
        default=None,
        examples=["VLM"],
        description="Filter by tag.",
    )
    source: str | None = Field(
        default=None,
        examples=["upload"],
        description="Filter by source.",
    )

# /search/chunks 的请求体。
class SearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        examples=["CLIP 的对抗鲁棒性怎么样？"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        examples=[5],
    )
    filters: SearchFilters | None = Field(
        default=None,
        description="Optional metadata filters.",
    )

# 每一条检索结果。
class SearchResult(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict[str, Any]
    distance: float | None = None
    score: float

class SearchResponse(BaseModel):
    query: str
    top_k: int
    filters: dict[str, Any]
    results: list[SearchResult]

# 表示答案的证据来源。每一个 citation 对应一个被检索出来的 chunk。
class Citation(BaseModel):
    doc_id: str = Field(
        ...,
        examples=["doc_001"],
        description="来源文档 ID",
    )
    chunk_id: str = Field(
        ...,
        examples=["doc_001_chunk_0003"],
        description="来源 chunk ID",
    )
    filename: str | None = Field(
        default=None,
        examples=["clip_robustness.md"],
        description="来源文件名",
    )
    source: str | None = Field(
        default=None,
        examples=["upload"],
        description="文档来源，例如 upload / sample",
    )
    doc_type: str | None = Field(
        default=None,
        examples=["paper"],
        description="文档类型，例如 paper / experiment / meeting",
    )
    tag: str | None = Field(
        default=None,
        examples=["RAG"],
        description="文档标签",
    )
    text_snippet: str = Field(
        ...,
        examples=["RAG 系统的基本流程包括文档上传、chunking、embedding 和检索。"],
        description="用于支撑答案的原文片段",
    )
    score: float = Field(
        ...,
        ge=0.0,
        examples=[0.82],
        description="检索相关性分数",
    )

# 用户请求
class AskRequest(BaseModel):
    question: str = Field(
        ...,
        examples=["RAG 的基本流程是什么？"],
        description="用户提出的问题",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        examples=[5],
        description="检索返回的 chunk 数量",
    )
    filters: dict[str, str] | None = Field(
        default=None,
        examples=[{"doc_type": "paper", "tag": "RAG"}],
        description="metadata 过滤条件，目前支持 doc_type、tag、source",
    )
    # validator 做了空字符串检查
    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("question 不能为空")

        return cleaned

    @field_validator("filters")
    @classmethod
    def validate_filters(
        cls,
        value: dict[str, str] | None,
    ) -> dict[str, str] | None:
        if value is None:
            return None

        allowed_keys = {"doc_type", "tag", "source"}

        for key, filter_value in value.items():
            if key not in allowed_keys:
                raise ValueError(
                    "filters 只支持 doc_type、tag、source"
                )

            if not isinstance(filter_value, str) or not filter_value.strip():
                raise ValueError(
                    "filters 的值必须是非空字符串"
                )

        return value


class AskResponse(BaseModel):
    answer: str = Field(
        ...,
        examples=["根据检索到的资料，RAG 的基本流程包括文档上传、解析、切分、向量化、检索和生成回答。"],
        description="基于检索上下文生成的回答",
    )
    citations: list[Citation] = Field(
        default_factory=list,
        description="答案引用的来源片段",
    )