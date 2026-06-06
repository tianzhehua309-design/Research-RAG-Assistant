from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


DocType = Literal["paper", "experiment", "meeting", "note", "other"]
SourceType = Literal["upload", "sample"]
DocumentStatus = Literal["uploaded", "indexed", "failed"]


class HealthResponse(BaseModel):
    status: str


class ErrorDetail(BaseModel):
    code: str = Field(..., examples=["EMPTY_FILE"])
    message: str = Field(..., examples=["上传文件不能为空"])
    retryable: bool = Field(..., examples=[False])


class ErrorResponse(BaseModel):
    error: ErrorDetail


class ErrorDetailResponse(BaseModel):
    error: ErrorDetail


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


class DocumentInfo(BaseModel):
    doc_id: str = Field(..., examples=["doc_001"])
    filename: str = Field(..., examples=["clip_robustness.md"])
    metadata: DocumentMetadata
    created_at: str = Field(..., examples=["2026-05-15T10:00:00Z"])
    status: DocumentStatus = Field(default="uploaded", examples=["uploaded"])
    chunk_count: int = Field(default=0, ge=0, examples=[0])


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
    doc_id: str = Field(..., examples=["doc_123"])
    chunk_size: int = Field(default=800, ge=100, le=3000)
    overlap: int = Field(default=100, ge=0, le=1000)


class IndexResponse(BaseModel):
    doc_id: str
    indexed: bool
    chunk_count: int


class SearchFilters(BaseModel):
    doc_type: DocType | None = Field(default=None, examples=["paper"])
    source: SourceType | None = Field(default=None, examples=["upload"])
    tag: str | None = Field(default=None, examples=["VLM"])


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


class SearchResult(BaseModel):
    chunk_id: str
    doc_id: str
    filename: str = ""
    text: str
    metadata: dict[str, Any]
    distance: float | None = None
    score: float


class SearchResponse(BaseModel):
    query: str
    top_k: int
    filters: dict[str, Any] = Field(default_factory=dict)
    results: list[SearchResult]


class Citation(BaseModel):
    doc_id: str = Field(..., examples=["doc_001"])
    chunk_id: str = Field(..., examples=["doc_001_chunk_0003"])
    filename: str | None = Field(default=None, examples=["clip_robustness.md"])
    source: str | None = Field(default=None, examples=["upload"])
    doc_type: str | None = Field(default=None, examples=["paper"])
    tag: str | None = Field(default=None, examples=["RAG"])
    text_snippet: str = Field(
        ...,
        examples=["RAG 系统的基本流程包括文档上传、chunking、embedding 和检索。"],
    )
    score: float = Field(..., ge=0.0, examples=[0.82])


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
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
    max_context_chars: int = Field(
        default=4000,
        ge=500,
        le=12000,
        description="context packing 的最大上下文字符数",
    )

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
                raise ValueError("filters 只支持 doc_type、tag、source")

            if not isinstance(filter_value, str) or not filter_value.strip():
                raise ValueError("filters 的值必须是非空字符串")

        return value


class AskResponse(BaseModel):
    question: str | None = Field(
        default=None,
        examples=["RAG 的基本流程是什么？"],
        description="用户提出的问题",
    )
    answer: str = Field(
        ...,
        examples=["根据检索到的资料，RAG 的基本流程包括文档上传、解析、切分、向量化、检索和生成回答。"],
        description="基于检索上下文生成的回答",
    )
    citations: list[Citation] = Field(
        default_factory=list,
        description="答案引用的来源片段",
    )