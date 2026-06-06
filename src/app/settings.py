import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    qa_mode: str
    llm_base_url: str | None
    llm_api_key: str | None
    llm_model: str
    llm_timeout: float
    llm_max_retries: int
    llm_backoff_seconds: float


def get_settings() -> Settings:
    return Settings(
        qa_mode=os.getenv("QA_MODE", "mock").strip().lower(),
        llm_base_url=os.getenv("LLM_BASE_URL", "").strip() or None,
        llm_api_key=os.getenv("LLM_API_KEY", "").strip() or None,
        llm_model=os.getenv("LLM_MODEL", "deepseek-chat").strip(),
        llm_timeout=float(os.getenv("LLM_TIMEOUT", "30.0")),
        llm_max_retries=int(os.getenv("LLM_MAX_RETRIES", "1")),
        llm_backoff_seconds=float(os.getenv("LLM_BACKOFF_SECONDS", "0.2")),
    )