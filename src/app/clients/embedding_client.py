import hashlib
import math
import re

from src.app.errors import AppError

class MockEmbeddingClient:
    def __init__(self, dimension: int = 128):
        self.dimension = dimension

    # 把用户的问题 query 转成一个 embedding 向量。
    def embed_query(self, query: str) -> list[float]:
        return self._embed_one(query)
    
    # 把多个文本一次性转成多个 embedding 向量。
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]
    
    # 把一段文本转换成一个固定长度的“假 embedding 向量”。
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
       
        # 如果 dimension=1，向量只有一个元素：这时候访问 vector[1] 会报错。
        # 只有维度大于 1 时，才设置第二维。
        if self.dimension > 1:
            # 把文本长度作为向量第二维的一个简单特征。
            # 最大值不超过 1.0
            vector[1] = min(len(cleaned_text) / 100, 1.0)

        return vector
    
class HashEmbeddingClient:
    def __init__(self, dimension: int = 64) -> None:
        # 向量维度必须是正数。
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        
        self.dimension = dimension

    def embed_query(self, query: str) -> list[float]:
        return self._embed_one(query)
    
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    # 上面那个是假 embedding，只根据文本长度生成简单向量。
    # 这个是hash embedding，会根据文本里的 token 内容生成更有区分度的向量。
    def _embed_one(self, text: str) -> list[float]:
        cleaned_text = text.strip()

        if not cleaned_text:
            raise AppError(
                code="EMPTY_TEXT",
                message="embedding 输入文本不能为空",
                retryable=False,
            )
        
        # 这一步会把文本切成一个个 token。
        # 后面不是直接对整段文本做 hash，而是对每个 token 做 hash。
        tokens = self._tokenize(cleaned_text)

        vector = [0.0] * self.dimension

        for token in tokens:
            # 把 token 转成一个稳定的 hash 结果。
            # 比如：token = "python"
            # 先执行：token.encode("utf-8") 把字符串转成 bytes。
            # hashlib.sha256(...).digest() 生成一个固定长度的二进制 hash 结果。
            # 为什么要 hash？ 因为我们想让同一个 token 每次都落到同一个向量位置。
            # 比如：python 每次都映射到第 12 维，fastapi 每次都映射到第 31 维
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            
            # 算 token 应该落到向量哪个位置
            # 第一步：取 hash 的前 4 个字节  digest[:4] 
            # 第二步：转成整数 int.from_bytes(digest[:4], "big") "big" 表示大端字节序
            # 第三步：取模 % self.dimension
            index = int.from_bytes(digest[:4], "big") % self.dimension

            # 根据 hash 的第 5 个字节，决定这个 token 是加 1 还是减 1。
            sign = 1.0 if digest[4] % 2 == 0 else -1.0

            vector[index] += sign

        return self._normalize(vector)
    
    # 把一整段文本拆成多个 token
    # "Python FastAPI RAG" -》 ["python", "fastapi", "rag"]
    # "CLIP 的对抗鲁棒性" -》 "clip", "的", "对", "抗", "鲁", "棒", "性"]
    def _tokenize(self, text: str) -> list[str]:
        # re.findall() 是 Python 的正则表达式
        # text.lower() 把英文全部转成小写。
        # 中间的 | 表示： 或者
        # [a-zA-Z0-9_]+英文字母，数字，下划线， + 表示连续出现一次或多次
        # [\u4e00-\u9fff] 是 Unicode 里汉字的范围，表示匹配任意一个汉字。
        tokens = re.findall(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]", text.lower())

        if not tokens:
            return [text.lower()]
        
        return tokens
    
    # 把一个向量归一化，也就是把向量长度缩放成 1。
    # 短文本：token 少，向量数值小
    # 长文本：token 多，向量数值大
    # 如果不处理，长文本可能天然更“强”，影响相似度比较。
    # 输入一个向量 vector，返回归一化后的向量。
    def _normalize(self, vector: list[float]) -> list[float]:
        # 向量长度 = sqrt(每一维平方之和)s
        norm = math.sqrt(sum(value * value for value in vector))

        if norm == 0:
            return vector

        return [value / norm for value in vector]


    
