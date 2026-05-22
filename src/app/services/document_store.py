import json
from pathlib import Path
from typing import Any

from src.app.errors import AppError

class DocumentStore:
    # metadata_dir 就是 metadata 文件保存的目录路径
    def __init__(self, metadata_dir:str | Path = "data/metadata")->None:
        self.metadata_dir = Path(metadata_dir)
        # 如果 data/metadata 不存在，就自动创建
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        self.documents_path = self.metadata_dir / "documents.json"
        self.chunks_path = self.metadata_dir / "chunks.json"

        self._ensure_json_file(self.documents_path,default={})
        self._ensure_json_file(self.chunks_path,default={})

    # 检查某个 JSON 文件是否存在，如果不存在，就创建它，并写入一个默认内容
    def _ensure_json_file(self,path:Path,default:Any)->None:
        if not path.exists():
            self._write_json(path,default)
    
    # 安全地读取一个 JSON 文件；如果文件不存在，就返回默认值；如果文件内容不是合法 JSON，就抛出统一业务错误。
    def _read_json(self,path:Path,default:Any)->Any:
        if not path.exists():
            return default
        
        try:
            # 用只读模式打开这个文件，并且用 utf-8 编码读取
            # with 语句会在读取完成后自动关闭文件，避免资源泄露
            # "r" 表示只读模式
            # encoding="utf-8" 表示用 utf-8 编码读取，这样就能正确处理中文等非 ASCII 字符了
            with path.open("r",encoding="utf-8") as f:
                return json.load(f)
        # 如果 json.load() 解析失败，说明这个文件不是合法的 JSON 文件，我们就抛出一个统一的业务错误，告诉用户这个问题
        except json.JSONDecodeError:
            raise AppError(
                code="METADATA_FILE_INVALID",
                message=f"{path.name} 不是合法 JSON 文件",
                retryable=False,
            )
    
    # 把 Python 数据写入到 JSON 文件里。
    # data: Any 表示要写进去的数据。
    def _write_json(self,path:Path,data:Any)->None:
        # 以写入模式打开这个文件。
        # "w" 表示写入模式，如果文件不存在就创建，如果文件存在就覆盖。
        # encoding="utf-8" 表示用 utf-8 编码写入，这样就能正确处理中文等非 ASCII 字符了。
        with path.open("w",encoding="utf-8") as f:
            # json.dump() 的作用是：把 Python 对象写入文件，保存成 JSON 格式。
            # json.dump(data, f)  写入文件 把 data 这个 Python 对象写入到 f 这个文件里，保存成 JSON 格式。
            # json.dumps(data) 转成 JSON 字符串
            # ensure_ascii=False 保存中文时，不要转义成 Unicode 编码。
            # indent=2 表示保存的 JSON 文件格式化，缩进 2 个空格，这样人类更容易阅读。
            json.dump(data,f,ensure_ascii=False,indent=2)
    
    # 把一篇 document 的 metadata 保存到 documents.json 文件里。
    def save_document(self,document: dict[str, Any])->dict[str, Any]:
        doc_id = document.get("doc_id")

        if not doc_id:
            raise AppError(
                code="MISSING_DOC_ID",
                message="document 缺少 doc_id",
                retryable=False,
            )
        
        documents = self._read_json(self.documents_path,default={})
        documents[doc_id] = document
        self._write_json(self.documents_path,documents)

        return document
    
    # 根据 doc_id 从 documents.json 里查找某一篇文档的 metadata。找到了就返回，找不到就抛出统一错误。
    def get_document(self,doc_id:str)->dict[str,Any]|None:
        documents = self._read_json(self.documents_path,default={})

        if doc_id not in documents:
            raise AppError(
                code="DOCUMENT_NOT_FOUND",
                message=f"文档不存在：{doc_id}",
                retryable=False,
            )

        return documents[doc_id]
    
    # 从 documents.json 里面读取所有文档信息，然后把它们转换成列表返回。
    def list_documents(self) -> list[dict[str, Any]]:
        documents = self._read_json(self.documents_path, default={})
        # documents.values() 返回的是一个 dict_values 对象，不是普通列表。所以我们用 list() 函数把它转换成普通列表，这样调用者就能直接使用了。
        return list(documents.values())
    
    def save_chunks(self, doc_id: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not doc_id:
            raise AppError(
                code="MISSING_DOC_ID",
                message="保存 chunks 时缺少 doc_id",
                retryable=False,
            )

        chunk_store = self._read_json(self.chunks_path, default={})
        chunk_store[doc_id] = chunks
        self._write_json(self.chunks_path, chunk_store)

        return chunks
    
    def get_chunks_by_doc_id(self, doc_id: str) -> list[dict[str, Any]]:
        chunk_store = self._read_json(self.chunks_path, default={})
        return chunk_store.get(doc_id, [])
    
    def delete_document(self, doc_id: str) -> None:
        documents = self._read_json(self.documents_path, default={})
        chunk_store = self._read_json(self.chunks_path, default={})

        if doc_id not in documents:
            raise AppError(
                code="DOCUMENT_NOT_FOUND",
                message=f"文档不存在：{doc_id}",
                retryable=False,
            )

        documents.pop(doc_id, None)
        chunk_store.pop(doc_id, None)

        self._write_json(self.documents_path, documents)
        self._write_json(self.chunks_path, chunk_store)


