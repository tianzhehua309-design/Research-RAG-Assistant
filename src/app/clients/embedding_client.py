import hashlib
import math
import re

from src.app.errors import AppError

class MockEmbeddingClient:
    def __init__(self, dimension: int = 128):
        self.dimension = dimension

    def embed_query(self, query: str) -> list[float]:
        return self._embed_one(query)
    
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]
    
    def _embed_one(self, text: str) -> list[float]:
        cleaned_text = text.strip()

        if not cleaned_text:
            raise AppError(
                code="EMPTY_TEXT",
                message="embedding 输入文本不能为空",
                retryable=False,
            )
        
        vector = [0.0] * self.dimension
        vector[0] = 1.0

        if self.dimension > 1:
            vector[1] = min(len(cleaned_text) / 100, 1.0)

        return vector
    
class HashEmbeddingClient:
    def __init__(self, dimension: int = 64) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        
        self.dimension = dimension

    def embed_query(self, query: str) -> list[float]:
        return self._embed_one(query)
    
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]
    
    def _embed_one(self, text: str) -> list[float]:
        cleaned_text = text.strip()

        if not cleaned_text:
            raise AppError(
                code="EMPTY_TEXT",
                message="embedding 输入文本不能为空",
                retryable=False,
            )
        
        tokens = self._tokenize(cleaned_text)

        vector = [0.0] * self.dimension

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()

            index = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0

            vector[index] += sign

        return self._normalize(vector)
    
    def _tokenize(self, text: str) -> list[str]:
        tokens = re.findall(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]", text.lower())

        if not tokens:
            return [text.lower()]
        
        return tokens
    
    def _normalize(self, vector: list[float]) -> list[float]:
        norm = math.sqrt(sum(value * value for value in vector))

        if norm == 0:
            return vector

        return [value / norm for value in vector]


    
