from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class SearchHit:
    id: str
    text: str
    score: float
    metadata: dict


@runtime_checkable
class VectorStore(Protocol):
    def upsert(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> None:
        ...

    def similarity_search(
        self,
        vector: list[float],
        k: int,
        filters: dict | None = None,
    ) -> list[SearchHit]:
        ...

    def delete_by_source(self, source: str) -> None:
        ...

    def count(self) -> int:
        ...