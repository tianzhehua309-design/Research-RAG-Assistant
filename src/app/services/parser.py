import re
from pathlib import Path

from click.termui import raw_terminal
from pypdf import PdfReader

from  src.app.errors import AppError

SUPPORTED_EXTENSIONS = {".txt",".md",".markdown",".pdf"}

"""
    根据文件路径和文件名解析文档，统一返回 clean_text。

    支持：
    - .txt
    - .md / .markdown
    - .pdf
"""
def parse_document(file_path:str,filename:str|None = None) -> str:
    path = Path(file_path)

    if not path.exists():
        raise AppError(
            code="FILE_NOT_FOUND",
            message="文件不存在",
            retryable=False,
        )

    display_name = filename or path.name
    suffix = Path(display_name).suffix.lower() or path.suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        raise AppError(
            code="UNSUPPORTED_FILE_TYPE",
            message="只支持 TXT、Markdown、PDF 文件",
            retryable=False,
        )

    if suffix == ".txt":
        raw_text = _read_text_file(path)
    elif suffix in {".md",".markdown"}:
        raw_text = _read_text_file(path)
        raw_text = _clean_markdown(raw_text)
    elif suffix == ".pdf":
        raw_text = _read_pdf_file(path)
    else:
        raise AppError(
            code="UNSUPPORTED_FILE_TYPE",
            message="只支持 TXT、Markdown、PDF 文件",
            retryable=False,
        )

    clean_text = normalize_text(raw_text)

    if not clean_text:
        raise AppError(
            code="EMPTY_DOCUMENT_TEXT",
            message="文档解析后没有有效文本",
            retryable=False,
        )

    return clean_text

"""
    读取 TXT / Markdown 文本文件。

    为什么尝试多个编码？
    - 有些文件是 utf-8
    - 有些中文 Windows 文件可能是 gbk
"""
def _read_text_file(path:Path) -> str:
    encodings = ["utf-8","utf-8-sig","gbk"]

    for encoding in encodings:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    # 为什么这里要抛出一个错误
    raise AppError(
        code="FILE_DECODE_ERROR",
        message="文本文件解码失败，请确认文件编码为 UTF-8 或 GBK",
        retryable=False,
    )


"""
    清洗 Markdown 语法，但尽量保留正文内容。

    目标：
    - 去掉 #、**、`、[]() 等符号
    - 保留标题文字
    - 保留段落结构
"""

def _clean_markdown(text:str) -> str:
    # 去掉代码块，只保留里面的文本内容会比较复杂；
    # 第一版先去掉代码块标记本身。
    text = re.sub(r"```[a-zA-Z0-9_-]*", "", text)
    text = text.replace("```", "")

    # 去掉行内代码反引号：`FastAPI` -> FastAPI
    text = re.sub(r"`([^`]*)`", r"\1", text)

    # 图片：![alt](url) -> alt
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)

    # 链接：[text](url) -> text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    # 标题：### Title -> Title
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)

    # 引用：> content -> content
    text = re.sub(r"^\s*>\s?", "", text, flags=re.MULTILINE)

    # 无序列表：- item / * item / + item -> item
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)

    # 有序列表：1. item -> item
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)

    # 粗体 / 斜体：**text** / *text* / __text__ / _text_
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    text = re.sub(r"_([^_]+)_", r"\1", text)

    return text

"""
    读取文本型 PDF。

    注意：
    - 只支持有文本层的 PDF
    - 扫描版 PDF 第一版不支持 OCR
"""
def _read_pdf_file(path:Path) -> str:
    try:
        reader = PdfReader(str(path))
    except Exception:
        raise AppError(
            code="INVALID_PDF",
            message="无法读取 PDF 文件，请确认上传的是合法 PDF",
            retryable=False,
        )

    page_texts: list[str] = []
    for page_index,page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()

        if text:
            page_texts.append(f"--- Page {page_index + 1} ---\n{text}")

    return "\n\n".join(page_texts)


"""
    统一清洗文本格式。

    做这些事：
    - 统一换行符
    - 去掉每行首尾空白
    - 合并连续空格
    - 合并过多空行
    - 保留段落边界
"""
def normalize_text(text:str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\t", " ")

    cleaned_lines: list[str] = []

    for line in text.split("\n"):
        # 合并英文空格和中文全角空格
        line = re.sub(r"[ \u3000]+", " ", line)
        line = line.strip()
        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines)

    # 三个以上换行，压缩成两个换行，保留段落感
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()

