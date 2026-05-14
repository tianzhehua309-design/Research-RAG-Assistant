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