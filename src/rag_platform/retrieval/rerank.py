from __future__ import annotations

import logging

from rag_platform.stores.base import SearchHit

logger = logging.getLogger(__name__)


def rerank_hits(query: str, hits: list[SearchHit], k: int) -> list[SearchHit]:
    """Rerank with FlashRank when installed; otherwise warn and return top *k* as-is."""
    if not hits:
        return []

    try:
        from flashrank import Ranker, RerankRequest
    except ImportError:
        logger.warning(
            "retrieval.rerank is enabled but FlashRank is not installed; "
            "skipping rerank. Install with: pip install 'rag-platform[rerank]'"
        )
        return hits[:k]

    passages = [{"id": h.id, "text": h.text} for h in hits]
    ranker = Ranker()
    request = RerankRequest(query=query, passages=passages)
    results = ranker.rerank(request)

    by_id = {h.id: h for h in hits}
    reranked: list[SearchHit] = []
    for item in results[:k]:
        doc_id = str(item["id"])
        base = by_id.get(doc_id)
        if base is None:
            continue
        score = float(item.get("score", base.score))
        reranked.append(
            SearchHit(id=base.id, text=base.text, score=score, metadata=base.metadata)
        )
    return reranked