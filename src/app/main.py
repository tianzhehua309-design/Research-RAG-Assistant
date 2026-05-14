import time
from uuid import uuid4

from fastapi import FastAPI,Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.app.errors import AppError
from src.app.schemas import HealthResponse
from src.app.logger import get_logger


logger = get_logger(__name__)
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

