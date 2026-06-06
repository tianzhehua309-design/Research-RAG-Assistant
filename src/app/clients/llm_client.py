import time
from typing import Any

import httpx
from src.app.settings import get_settings
from src.app.errors import AppError


class MockQAClient:
    def generate_answer(
        self,
        question: str,
        context: str,
        request_id: str | None = None,
    ) -> str:
        cleaned_question = question.strip()
        cleaned_context = context.strip()

        if not cleaned_question:
            raise AppError(
                code="EMPTY_QUESTION",
                message="question 不能为空",
                retryable=False,
            )

        if not cleaned_context:
            return "没有找到相关依据，无法基于当前知识库回答这个问题。"

        return (
            "根据检索到的资料，可以回答如下：\n"
            f"问题是：{cleaned_question}\n"
            "系统已经找到相关文档片段，具体依据请查看 citations。"
        )


class LLMClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 30.0,
        max_retries: int = 1,
        backoff_seconds: float = 0.2,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds

    def generate_answer(
        self,
        question: str,
        context: str,
        request_id: str | None = None,
    ) -> str:
        if not question.strip():
            raise AppError(
                code="EMPTY_QUESTION",
                message="question 不能为空",
                retryable=False,
            )

        headers: dict[str, str] = {}
        if request_id:
            headers["X-Request-ID"] = request_id

        max_attempts = self.max_retries + 1

        for attempt in range(1, max_attempts + 1):
            try:
                return self._generate_once(
                    question=question,
                    context=context,
                    headers=headers,
                )
            except AppError as exc:
                if not exc.retryable:
                    raise

                if attempt >= max_attempts:
                    raise

                time.sleep(self.backoff_seconds * attempt)

        raise AppError(
            code="UPSTREAM_REQUEST_ERROR",
            message="上游 LLM 请求失败",
            retryable=True,
        )

    def _generate_once(
        self,
        question: str,
        context: str,
        headers: dict[str, str],
    ) -> str:
        request_headers = {
            **headers,
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是一个研究资料 RAG 问答助手。"
                        "你只能基于用户提供的 context 回答。"
                        "如果 context 中没有依据，请明确说明没有找到相关依据。"
                        "不要编造 citation，citation 会由系统单独返回。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"问题：{question}\n\n"
                        f"可用 context：\n{context}\n\n"
                        "请基于 context 给出简洁、准确的中文回答。"
                    ),
                },
            ],
            "temperature": 0,
            "max_tokens": 512,
            "stream": False,
        }

        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=request_headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()

        except httpx.TimeoutException:
            raise AppError(
                code="UPSTREAM_TIMEOUT",
                message="上游 LLM 请求超时",
                retryable=True,
            )

        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code

            if status_code >= 500:
                raise AppError(
                    code="UPSTREAM_SERVER_ERROR",
                    message=f"上游 LLM 返回 {status_code}",
                    retryable=True,
                )

            raise AppError(
                code="UPSTREAM_BAD_REQUEST",
                message=f"上游 LLM 返回 {status_code}",
                retryable=False,
            )

        except httpx.RequestError:
            raise AppError(
                code="UPSTREAM_REQUEST_ERROR",
                message="上游 LLM 请求失败",
                retryable=True,
            )

        except ValueError:
            raise AppError(
                code="UPSTREAM_INVALID_JSON",
                message="上游 LLM 返回的不是合法 JSON",
                retryable=True,
            )

        return self._extract_answer(data)

    def _extract_answer(self, data: dict[str, Any]) -> str:
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise AppError(
                code="UPSTREAM_INVALID_RESPONSE",
                message="上游 LLM 返回结构不合法",
                retryable=True,
            )

        if not isinstance(content, str) or not content.strip():
            raise AppError(
                code="UPSTREAM_EMPTY_ANSWER",
                message="上游 LLM 返回内容为空",
                retryable=True,
            )

        return content.strip()

def build_qa_client():
    settings = get_settings()

    if settings.qa_mode == "mock":
        return MockQAClient()

    if settings.qa_mode == "llm":
        if settings.llm_base_url is None:
            raise AppError(
                code="LLM_NOT_CONFIGURED",
                message="LLM_BASE_URL 未配置",
                retryable=False,
            )

        if settings.llm_api_key is None:
            raise AppError(
                code="LLM_NOT_CONFIGURED",
                message="LLM_API_KEY 未配置",
                retryable=False,
            )

        return LLMClient(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            timeout=settings.llm_timeout,
            max_retries=settings.llm_max_retries,
            backoff_seconds=settings.llm_backoff_seconds,
        )

    raise AppError(
        code="INVALID_QA_MODE",
        message="QA_MODE 只支持 mock 或 llm",
        retryable=False,
    )