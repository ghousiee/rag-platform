from __future__ import annotations

from rag_platform.stores.base import SearchHit


def _normalize_scores(hits: list[SearchHit]) -> dict[str, float]:
    if not hits:
        return {}
    scores = [h.score for h in hits]
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return {h.id: 1.0 for h in hits}
    span = hi - lo
    return {h.id: (h.score - lo) / span for h in hits}


def fuse_dense_and_bm25(
    dense_hits: list[SearchHit],
    bm25_hits: list[SearchHit],
    bm25_weight: float,
    k: int,
) -> list[SearchHit]:
    """Merge dense vector and BM25 rankings with weighted normalized scores."""
    dense_w = 1.0 - bm25_weight
    dense_norm = _normalize_scores(dense_hits)
    bm25_norm = _normalize_scores(bm25_hits)

    by_id: dict[str, SearchHit] = {}
    for hit in dense_hits:
        by_id[hit.id] = hit
    for hit in bm25_hits:
        by_id.setdefault(hit.id, hit)

    combined: list[tuple[float, SearchHit]] = []
    for doc_id, hit in by_id.items():
        score = dense_w * dense_norm.get(doc_id, 0.0) + bm25_weight * bm25_norm.get(
            doc_id, 0.0
        )
        combined.append(
            (score, SearchHit(id=hit.id, text=hit.text, score=score, metadata=hit.metadata))
        )

    combined.sort(key=lambda pair: pair[0], reverse=True)
    return [hit for _, hit in combined[:k]]