from __future__ import annotations

import logging
from dataclasses import dataclass

import pytest

from rag_platform.config import RetrievalConfig
from rag_platform.retrieval import Retriever, build_retriever
from rag_platform.stores.base import SearchHit


class _FakeSettings:
    def __init__(self, retrieval: RetrievalConfig) -> None:
        self.retrieval = retrieval


@dataclass
class FakeEmbed:
    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 0.1, 0.2]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t)), 0.1, 0.2] for t in texts]


class FakeStore:
    def __init__(self, hits: list[SearchHit] | None = None) -> None:
        self._hits = hits or []
        self.last_k: int | None = None
        self.last_filters: dict | None = None

    def similarity_search(
        self,
        vector: list[float],
        k: int,
        filters: dict | None = None,
    ) -> list[SearchHit]:
        self.last_k = k
        self.last_filters = filters
        return self._hits[:k]

    def upsert(self, ids, texts, embeddings, metadatas) -> None:
        pass

    def delete_by_source(self, source: str) -> None:
        pass

    def count(self) -> int:
        return 0


def _retrieval_cfg(*, hybrid: bool = True, rerank: bool = False) -> RetrievalConfig:
    return RetrievalConfig(top_k=4, hybrid=hybrid, bm25_weight=0.4, rerank=rerank)


def _make_retriever(
    cfg: RetrievalConfig, store: FakeStore, embedder: FakeEmbed | None = None
) -> Retriever:
    return build_retriever(_FakeSettings(cfg), store, embedder or FakeEmbed())


def test_retrieve_dense_only_when_hybrid_disabled():
    store = FakeStore([SearchHit("d1", "hello world", 0.95, {"source": "a"})])
    retriever = _make_retriever(_retrieval_cfg(hybrid=False), store)
    hits = retriever.retrieve("hello")
    assert len(hits) == 1
    assert hits[0].id == "d1"
    assert store.last_k == 4


def test_retrieve_hybrid_after_rebuild_bm25():
    store = FakeStore(
        [
            SearchHit("1", "python tutorial", 0.5, {}),
            SearchHit("2", "unrelated noise", 0.9, {}),
        ]
    )
    retriever = _make_retriever(_retrieval_cfg(hybrid=True), store)
    retriever.rebuild_bm25(
        texts=["python tutorial", "unrelated noise", "python advanced"],
        metadatas=[{}, {}, {}],
        ids=["1", "2", "3"],
    )
    hits = retriever.retrieve("python", k=2)
    assert len(hits) == 2
    ids = {h.id for h in hits}
    assert "3" in ids or "1" in ids


def test_retrieve_respects_custom_k():
    store = FakeStore([SearchHit("a", "x", 1.0, {}), SearchHit("b", "y", 0.5, {})])
    retriever = _make_retriever(_retrieval_cfg(hybrid=False), store)
    hits = retriever.retrieve("q", k=1)
    assert len(hits) == 1
    assert store.last_k == 1


def test_rerank_missing_flashrank_warns(caplog: pytest.LogCaptureFixture):
    store = FakeStore([SearchHit("1", "doc one", 0.9, {})])
    retriever = _make_retriever(_retrieval_cfg(hybrid=False, rerank=True), store)
    with caplog.at_level(logging.WARNING):
        hits = retriever.retrieve("query")
    assert len(hits) == 1
    assert any("flashrank" in r.message.lower() for r in caplog.records)