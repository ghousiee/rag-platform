from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from rag_platform.models import IngestResult, QueryResult

__all__ = [
    "HealthResponse",
    "IngestRequest",
    "IngestResult",
    "QueryRequest",
    "QueryResult",
]


class IngestRequest(BaseModel):
    path: str


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None
    filters: dict[str, Any] | None = None


class HealthResponse(BaseModel):
    status: str = "ok"
    index_count: int