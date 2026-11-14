from __future__ import annotations

from rank_bm25 import BM25Okapi

from rag_platform.config import RetrievalConfig, Settings
from rag_platform.providers.base import EmbeddingProvider
from rag_platform.retrieval.hybrid import fuse_dense_and_bm25
from rag_platform.retrieval.rerank import rerank_hits
from rag_platform.stores.base import SearchHit, VectorStore


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class Retriever:
    def __init__(
        self,
        retrieval: RetrievalConfig,
        store: VectorStore,
        embedder: EmbeddingProvider,
    ) -> None:
        self._config = retrieval
        self._store = store
        self._embedder = embedder
        self._bm25: BM25Okapi | None = None
        self._corpus_ids: list[str] = []
        self._corpus_texts: list[str] = []
        self._corpus_metas: list[dict] = []

    def rebuild_bm25(
        self,
        texts: list[str],
        metadatas: list[dict],
        ids: list[str],
    ) -> None:
        self._corpus_ids = list(ids)
        self._corpus_texts = list(texts)
        self._corpus_metas = list(metadatas)
        if not texts:
            self._bm25 = None
            return
        tokenized = [_tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(tokenized)

    def _bm25_search(self, query: str, k: int) -> list[SearchHit]:
        if self._bm25 is None or not self._corpus_texts:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)[:k]
        hits: list[SearchHit] = []
        for idx, score in ranked:
            if score <= 0:
                continue
            hits.append(
                SearchHit(
                    id=self._corpus_ids[idx],
                    text=self._corpus_texts[idx],
                    score=float(score),
                    metadata=dict(self._corpus_metas[idx]),
                )
            )
        return hits

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        filters: dict | None = None,
    ) -> list[SearchHit]:
        limit = k if k is not None else self._config.top_k
        vector = self._embedder.embed_query(query)
        dense_hits = self._store.similarity_search(vector, k=limit, filters=filters)

        if self._config.hybrid and self._bm25 is not None:
            bm25_hits = self._bm25_search(query, limit)
            hits = fuse_dense_and_bm25(
                dense_hits, bm25_hits, self._config.bm25_weight, limit
            )
        else:
            hits = dense_hits

        if self._config.rerank:
            hits = rerank_hits(query, hits, limit)
        return hits[:limit]


def build_retriever(
    settings: Settings,
    store: VectorStore,
    embedder: EmbeddingProvider,
) -> Retriever:
    return Retriever(settings.retrieval, store, embedder)