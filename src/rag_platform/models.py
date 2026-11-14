from typing import Any

from pydantic import BaseModel, Field


class Document(BaseModel):
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Citation(BaseModel):
    source: str
    score: float
    text: str


class QueryResult(BaseModel):
    answer: str
    citations: list[Citation]


class IngestResult(BaseModel):
    files: int
    chunks: int
    errors: list[str]


class EmptyIndexError(Exception):
    """Raised when the vector index has no documents indexed yet."""