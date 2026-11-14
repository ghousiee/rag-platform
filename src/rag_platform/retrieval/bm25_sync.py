from __future__ import annotations

from rag_platform.retrieval.retriever import Retriever
from rag_platform.stores.base import VectorStore

# Hybrid BM25 requires store.list_chunks (Chroma has it); other backends are
# dense-only until they add it.


def sync_bm25(
    retriever: Retriever,
    store: VectorStore,
    *,
    fallback_ids: list[str] | None = None,
    fallback_texts: list[str] | None = None,
    fallback_metas: list[dict] | None = None,
) -> None:
    """Rebuild BM25 from the full store corpus when list_chunks is available."""
    list_chunks = getattr(store, "list_chunks", None)
    if list_chunks is not None:
        ids, texts, metas = list_chunks()
        if texts:
            retriever.rebuild_bm25(texts, metas, ids)
        return
    if fallback_texts:
        retriever.rebuild_bm25(
            fallback_texts,
            fallback_metas or [],
            fallback_ids or [],
        )