from pydantic import BaseModel

class HealthResponse(BaseModel):
    # 表示 /health 返回的 JSON 里必须有一个字符串字段 status。
    status: str