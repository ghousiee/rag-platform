"""SearchHit.score conventions: higher = more similar (comparable across backends)."""


def cosine_distance_to_similarity(distance: float) -> float:
    """Chroma/pgvector cosine distance -> similarity in roughly [0, 1]."""
    return 1.0 - float(distance)


def qdrant_cosine_similarity(score: float) -> float:
    """Qdrant COSINE search returns similarity; use as-is (not 1/(1+d))."""
    return float(score)