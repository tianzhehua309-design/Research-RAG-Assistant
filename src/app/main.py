import time
from pathlib import Path
from uuid import uuid4
# UploadFile 是 FastAPI 专门用来接收上传文件的类型。
# file.read() 是异步读取，所以接口函数要写成 async def。
# 它里面常用的东西有：
# file.filename	原始文件名
# await file.read()	读取文件内容，结果是 bytes
# file.content_type	文件 MIME 类型，例如 text/plain
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.app.errors import AppError
from src.app.logger import get_logger
from src.app.services.document_store import DocumentStore
from src.app.services.indexer import index_document
from src.app.schemas import SearchRequest, SearchResponse
from src.app.services.retriever import search_chunks
from src.app.schemas import (
    DocType,
    DocumentMetadata,
    HealthResponse,
    SourceType,
    UploadResponse,
    IndexRequest,
    IndexResponse,
)


logger = get_logger(__name__)
document_store = DocumentStore()

# 所有用户上传的原始文件，都先保存到 data/uploads 这个目录里
# 比如你上传：sample.md
# 系统生成：doc_id = "doc_b8424a216b3a"
# 最后保存成：data/uploads/doc_b8424a216b3a_sample.md
UPLOAD_DIR = Path("data/upload")
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf"}

app = FastAPI(
title="Research RAG Assistant",
    version="0.1.0",
    description="A RAG assistant for research papers, experiment logs, and meeting notes.",
)

def get_request_id(request: Request)->str:
    return getattr(request.state, "request_id", "unknown")

# middleware 会在每个 HTTP 请求的前后执行。
@app.middleware("http")
async def middleware(request: Request, call_next):
    request_id = str(uuid4())
    request.state.request_id = request_id

    start_time = time.perf_counter()

    logger.info(
        "request started",
        extra={
            "event": "request_started",
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
        },
    )

    # 继续执行真正的路由函数。
    response = await call_next(request)

    duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
    response.headers["X-Request-Id"] = request_id

    logger.info(
        "request completed",
        extra={
            "event": "request_completed",
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
        },
    )

    return response

#只要项目里任何地方 raise AppError(...)最终都会走这里
@app.exception_handler(AppError)
def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
    request_id = get_request_id(request)

    logger.error(
        "application error occurred",
        extra={
            "event": "application_error",
            "request_id": request_id,
            "error_code": exc.code,
            "retryable": exc.retryable,
            "method": request.method,
            "path": request.url.path,
        },
    )

    return JSONResponse(
        status_code=400,
        content=exc.to_dict(),
        headers={"X-Request-ID": request_id},
    )

#这个是处理 Pydantic 校验错误的。
#比如以后 /documents/upload 需要某些参数，如果用户少传、类型传错，就会触发这个异常处理器。
@app.exception_handler(RequestValidationError)
def handle__validation_error(request: Request, exc: RequestValidationError) ->JSONResponse:
    request_id = get_request_id(request)

    logger.error(
        "request validation failed",
        extra={
            "event": "validation_error",
            "request_id": request_id,
            "error_code": "INVALID_REQUEST",
            "retryable": False,
            "method": request.method,
            "path": request.url.path,
        },
    )

    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "INVALID_REQUEST",
                "message": "请求体校验失败",
                "retryable": False,
            }
        },
        headers={"X-Request-ID": request_id},
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")

#它的作用是帮你验证：
# AppError 是否能被统一捕获
# 错误响应格式是否正确
# X-Request-ID 是否也能返回
# 错误日志是否有 error_code
@app.get("/debug/error")
def debug_error() -> None:
    raise AppError(
        code="DEBUG_ERROR",
        message="这是一个用于测试统一错误返回的调试错误",
        retryable=False,
    )

# 步骤
# 1. 获取 request_id
# 2. 获取原始文件名
# 3. 清洗文件名，防止路径攻击
# 4. 检查文件名是否为空
# 5. 检查文件后缀是否允许
# 6. 读取上传文件内容
# 7. 检查文件是否为空
# 8. 检查文件大小是否超过 5MB
# 9. 生成唯一 doc_id
# 10. 拼出保存文件名
# 11. 创建上传目录
# 12. 保存文件到本地
# 13. 构造 metadata
# 14. 写一条上传成功日志
# 15. 返回 UploadResponse
@app.post("/documents/upload")
async def upload_document(
        request: Request,
        file: UploadFile = File(...),
        doc_type: DocType = Form("other"),
        tag: str = Form("general"),
        source: SourceType = Form("upload"),
)->UploadResponse:
    request_id = get_request_id(request)

    original_filename = file.filename or ""

    # 这是为了防止用户上传奇怪文件名，比如：
    # ../../secret.txt
    # Path(...).name 只保留最后的文件名：
    # secret.txt
    safe_filename = Path(original_filename).name

    if not safe_filename:
        raise AppError(
            code="EMPTY_FILE",
            message="文件名不能为空",
            retryable=False,
        )

    # 检查后缀
    # 因为项目二第一版只支持：
    # .txt
    # .md
    # .pdf
    suffix = Path(safe_filename).suffix.lower()

    if suffix not in ALLOWED_EXTENSIONS:
        raise AppError(
            code="INVALID_FILE_TYPE",
            message="只支持上传 .txt、.md、.pdf 文件",
            retryable=False,
        )

    # 这句会把上传文件内容读出来。
    # 注意：读出来的是：bytes 不是字符串。
    # 比如 Markdown 文件内容是：
    # Title
    # This is a note.
    # 读取后大概是：b"# Title\nThis is a note."
    # 为什么是 bytes？
    # 因为上传文件本质上是二进制数据，不管是 .txt、.md 还是 .pdf，从网络传过来都是 bytes。
    content = await file.read()

    if not content:
        raise AppError(
            code="EMPTY_FILE",
            message="上传文件不能为空",
            retryable=False,
        )

    if len(content) > MAX_UPLOAD_BYTES:
        raise AppError(
            code="FILE_TOO_LARGE",
            message="上传文件不能超过 5MB",
            retryable=False,
        )

    # 为什么要生成 doc_id？
    # 因为文件名可能重复。
    # 如果直接用文件名保存，就会覆盖。
    # 所以我们用唯一的 doc_id 标识文档：
    # doc_a1b2c3d4e5f6
    # 保存时变成：
    # doc_a1b2c3d4e5f6_notes.md
    # .hex 会去掉横杠，变成一串十六进制字符串：
    doc_id = f"doc_{uuid4().hex[:12]}"
    saved_filename = f"{doc_id}_{safe_filename}"

    # 创建上传目录
    # parents=True 表示如果父目录也不存在，就一起创建。
    # exist_ok=True 表示如果目录已经存在，不要报错。
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    # Path 对象可以用 / 拼路径。
    # UPLOAD_DIR = Path("data/uploads")
    # saved_filename = "doc_b8424a216b3a_sample.md"
    # 那么：
    # saved_path
    # 就是：
    # data/uploads/doc_b8424a216b3a_sample.md
    saved_path = UPLOAD_DIR / saved_filename
    # 它会把上传内容写入本地文件
    saved_path.write_bytes(content)

    # 构造 metadata、
    # 这个 metadata 后面非常重要。
    # 以后检索时可以做过滤：
    # 只查 doc_type=paper 的文档
    # 只查 tag=VLM 的文档
    # 只查 source=upload 的文档
    metadata = DocumentMetadata(
        doc_type=doc_type,
        source=source,
        tag=tag,
    )

    metadata_dict = {
        "doc_type": doc_type.value if hasattr(doc_type, "value") else str(doc_type),
        "source": source.value if hasattr(source, "value") else str(source),
        "tag": tag,
    }

    document = {
        "doc_id": doc_id,
        "filename": safe_filename,
        "file_path": str(saved_path),
        "metadata": metadata_dict,
        "status": "uploaded",
        "indexed": False,
        "chunk_count": 0,
    }

    # 关键：把 document metadata 真正写入 data/metadata/documents.json
    document_store.save_document(document)


    logger.info(
        "document uploaded",
        extra={
            "event": "document_uploaded",
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "doc_id": doc_id,
            "uploaded_filename": safe_filename,
        },
    )

    # 这个就是接口最终返回给用户的结果
    return UploadResponse(
        doc_id=doc_id,
        filename=safe_filename,
        metadata=metadata,
        status="uploaded",
    )

# 查看当前系统里保存了哪些文档
@app.get("/documents")
def list_documents():
    documents = document_store.list_documents()
    return {
        "documents": documents,
        "count": len(documents),
    }

@app.post("/documents/index", response_model=IndexResponse)
def index_document_endpoint(
    request: Request,
    # payload 表示用户传进来的 JSON 请求体。
    payload: IndexRequest,
) -> IndexResponse:
    request_id = get_request_id(request)

    logger.info(
        "document index called",
        extra={
            "event": "document_index_called",
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "doc_id": payload.doc_id,
        },
    )

    result = index_document(
        doc_id=payload.doc_id,
        chunk_size=payload.chunk_size,
        overlap=payload.overlap,
    )

    logger.info(
        "document index succeeded",
        extra={
            "event": "document_index_succeeded",
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "doc_id": payload.doc_id,
            "chunk_count": result["chunk_count"],
        },
    )

    return IndexResponse(**result)


@app.post("/search/chunks", response_model=SearchResponse)
def search_chunks_endpoint(
    request: Request,
    payload: SearchRequest,
) -> SearchResponse:
    request_id = get_request_id(request)

    filters = {}
    if payload.filters is not None:
        filters = payload.filters.model_dump(exclude_none=True)

    logger.info(
        "search chunks called",
        extra={
            "event": "search_chunks_called",
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "top_k": payload.top_k,
        },
    )

    results = search_chunks(
        query=payload.query,
        top_k=payload.top_k,
        filters=filters,
    )

    logger.info(
        "search chunks succeeded",
        extra={
            "event": "search_chunks_succeeded",
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "result_count": len(results),
        },
    )

    return SearchResponse(
        query=payload.query,
        top_k=payload.top_k,
        filters=filters,
        results=results,
    )
