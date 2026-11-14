from __future__ import annotations

from rag_platform.config import Settings
from rag_platform.models import Citation, QueryResult
from rag_platform.providers import build_embedder, build_llm
from rag_platform.providers.base import EmbeddingProvider, LLMProvider
from rag_platform.retrieval import Retriever, build_retriever
from rag_platform.retrieval.bm25_sync import sync_bm25
from rag_platform.stores import build_store
from rag_platform.stores.base import SearchHit, VectorStore

_EMPTY_INDEX_MSG = "Index is empty. Run ingest first."


def _format_context(hits: list[SearchHit]) -> str:
    parts: list[str] = []
    for index, hit in enumerate(hits, start=1):
        source = hit.metadata.get("source", "unknown")
        parts.append(f"[{index}] (source: {source})\n{hit.text}")
    return "\n\n".join(parts)


def query(
    question: str,
    settings: Settings,
    *,
    top_k: int | None = None,
    filters: dict | None = None,
    embedder: EmbeddingProvider | None = None,
    store: VectorStore | None = None,
    llm: LLMProvider | None = None,
    retriever: Retriever | None = None,
) -> QueryResult:
    embedder = embedder or build_embedder(settings)
    store = store or build_store(settings)
    llm = llm or build_llm(settings)

    if store.count() == 0:
        raise RuntimeError(_EMPTY_INDEX_MSG)

    retriever = retriever or build_retriever(settings, store, embedder)
    if settings.retrieval.hybrid:
        sync_bm25(retriever, store)

    hits = retriever.retrieve(question, k=top_k, filters=filters)
    context = _format_context(hits)
    user_content = f"Context:\n{context}\n\nQuestion: {question}"

    messages = [
        {"role": "system", "content": settings.prompts.system},
        {"role": "user", "content": user_content},
    ]
    answer = llm.generate(messages, temperature=settings.llm.temperature)

    citations = [
        Citation(
            source=str(hit.metadata.get("source", "")),
            score=hit.score,
            text=hit.text,
        )
        for hit in hits
    ]
    return QueryResult(answer=answer, citations=citations)