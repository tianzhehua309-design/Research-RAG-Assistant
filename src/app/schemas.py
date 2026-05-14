from pydantic import BaseModel,Field

class HealthResponse(BaseModel):
    # 表示 /health 返回的 JSON 里必须有一个字符串字段 status。
    status: str

class ErrorDetail(BaseModel):
    code: str = Field(...,examples=["EMPTY_FILE"])
    message: str = Field(...,examples=["上传文件不能为空"])
    retryable: bool = Field(...,examples=["False"])

class ErrorDetailResponse(BaseModel):
    error : ErrorDetail