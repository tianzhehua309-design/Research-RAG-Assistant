# Research RAG Assistant

研究资料与实验记录 RAG 助手。

## 项目目标

本项目面向论文、实验日志、组会记录和方法笔记，构建一个可检索、可问答、可返回引用来源的 RAG 知识库系统

最终目标包括：

- 文档上次
- 文档解析
- 文本清洗
- chunking 切分
- embedding 向量化
- Chroma 向量索引
- top-k 检索
- metadata filtering
- context packing
- answer + citations 返回

## Day 1:

- 初始化 FastAPI 项目骨架
- 实现 GET /health
- 建立基础目录结构

## 启动方式

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
uvicorn src.app.main:app --reload
```

## Day 2: 工程化基础

本阶段增加：

- 统一业务错误 `AppError`
- 统一错误返回格式：`error -> code / message / retryable`
- `request_id` middleware
- 响应头 `X-Request-ID`
- JSON 结构化日志
- 请求耗时 `duration_ms`
- 基础 API 测试

## 错误返回格式

```json
{
  "error": {
    "code": "DEBUG_ERROR",
    "message": "这是一个用于测试统一错误返回的调试错误",
    "retryable": false
  }
}
```

## Day 3: Schema 设计

本阶段完成项目二核心数据模型设计：

- `DocumentMetadata`
- `DocumentInfo`
- `ChunkInfo`
- `UploadResponse`
- `IndexRequest`
- `IndexResponse`
- `SearchRequest`
- `SearchResponse`
- `AskRequest`
- `AskResponse`
- `Citation`

核心对象关系：

```text
Document -> Chunks -> Search Results -> Citations
```

## Day 4: 文档上传接口

本阶段增加：

- `POST /documents/upload` 文档上传接口
- 支持上传 `.txt`、`.md`、`.pdf`
- 使用 `UploadFile` 接收用户上传文件
- 使用 `Form` 接收文档元数据
- 支持 `doc_type`、`tag`、`source` 字段
- 自动生成唯一 `doc_id`
- 将上传文件保存到本地 `data/uploads`
- 返回上传后的文档基础信息
- 非法文件类型返回统一错误
- 空文件返回统一错误
- 上传接口响应头包含 `X-Request-ID`

## 上传接口

```text
POST /documents/upload
```

## 请求参数

```text
file: 上传的文档文件
doc_type: 文档类型，例如 paper / experiment / meeting / note
tag: 文档标签，例如 RAG / CLIP / VLM
source: 文档来源，例如 upload
```

## 返回格式

```json
{
  "doc_id": "doc_xxx",
  "filename": "note.md",
  "doc_type": "note",
  "tag": "RAG",
  "source": "upload"
}
```

## 上传流程

```text
用户上传文件
↓
FastAPI 接收 UploadFile
↓
读取文件内容
↓
检查文件是否为空
↓
检查文件类型是否合法
↓
生成 doc_id
↓
保存文件到 data/uploads
↓
返回 UploadResponse
```

## 支持的文件类型

```text
.txt
.md
.pdf
```

## 错误返回示例

```json
{
  "error": {
    "code": "UNSUPPORTED_FILE_TYPE",
    "message": "只支持 TXT、Markdown、PDF 文件",
    "retryable": false
  }
}
```

## 测试覆盖

本阶段增加测试：

- Markdown 文件上传成功
- TXT 文件上传成功
- 非法文件类型返回错误
- 空文件返回错误
- 上传接口响应头包含 `X-Request-ID`

---

## Day 5: 文档解析与文本清洗

本阶段增加：

- `src/app/services/parser.py`
- 统一文档解析函数 `parse_document`
- TXT 文本读取
- Markdown 文本读取与语法清洗
- PDF 文本抽取
- 文本统一清洗 `normalize_text`
- 文件不存在错误处理
- 非法文件类型错误处理
- 文本解码失败错误处理
- PDF 解析失败错误处理
- 解析后空文本错误处理
- parser 单元测试

## 核心函数

``` python
parse_document(file_path: str, filename: str | None = None) -> str
```

作用：

```text
输入文件路径和文件名
↓
判断文件类型
↓
读取原始文本
↓
清洗 Markdown 或 PDF 内容
↓
统一处理空格和换行
↓
返回 clean_text
```

## 支持的解析类型

| 文件类型 | 后缀 | 处理方式 |
|---|---|---|
| TXT | `.txt` | 直接读取文本 |
| Markdown | `.md` / `.markdown` | 读取文本后清洗 Markdown 语法 |
| PDF | `.pdf` | 使用 `pypdf` 抽取文本 |

## 文本清洗内容

```text
去掉首尾空白
合并连续空格
统一换行符
删除多余空行
保留段落结构
```

## Markdown 清洗内容

```text
去掉标题符号 #
去掉粗体和斜体符号
去掉行内代码反引号
将 Markdown 链接转换为普通文本
清理列表符号
清理引用符号 >
```

## Parser 流程

```text
file_path + filename
↓
获取文件后缀 suffix
↓
判断是否属于支持类型
↓
.txt 直接读取
.md / .markdown 读取后清洗 Markdown
.pdf 使用 pypdf 抽取文本
↓
normalize_text 统一清洗文本
↓
如果 clean_text 为空，抛出 EMPTY_DOCUMENT_TEXT
↓
返回 clean_text
```

## 错误返回示例

```json
{
  "error": {
    "code": "EMPTY_DOCUMENT_TEXT",
    "message": "文档解析后没有有效文本",
    "retryable": false
  }
}
```

## 测试覆盖

本阶段增加测试：

- TXT 解析成功
- Markdown 解析成功
- 空 TXT 文件返回 `EMPTY_DOCUMENT_TEXT`
- 非法文件类型返回 `UNSUPPORTED_FILE_TYPE`
- 文件不存在返回 `FILE_NOT_FOUND`
- `normalize_text` 文本清洗正确
- PDF 解析流程通过 mock 测试

## 当前限制

```text
当前 PDF 解析只支持文本型 PDF。
扫描版 PDF 暂不支持，因为还没有加入 OCR。
```