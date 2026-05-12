from fastapi import FastAPI
from src.app.schemas import HealthResponse
app = FastAPI(
title="Research RAG Assistant",
    version="0.1.0",
    description="A RAG assistant for research papers, experiment logs, and meeting notes.",
)

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


