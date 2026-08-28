"""DASBase Arşiv AI — isteğe bağlı FastAPI katmanı."""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from services.indexer import build_index
from services.rag import answer_question

app = FastAPI(title="DASBase Arşiv AI", version="0.2.0")


class QueryRequest(BaseModel):
    query: str
    role: str = Field(default="Genel Müdür")
    search_type: str = Field(default="Hybrid Search")


@app.on_event("startup")
def _startup() -> None:
    build_index(force=False)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/query")
def query(body: QueryRequest) -> dict:
    result = answer_question(body.query, role_label=body.role)
    return result.model_dump()
