from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rag_platform.config import VectorStoreConfig
from rag_platform.stores.base import SearchHit
from rag_platform.stores.scoring import qdrant_cosine_similarity

if TYPE_CHECKING:
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct


class QdrantVectorStore:
    def __init__(self, config: VectorStoreConfig, client: QdrantClient) -> None:
        self._config = config
        self._client = client
        self._collection_ready = False

    def _ensure_collection(self, vector_dim: int) -> None:
        if self._collection_ready:
            return
        from qdrant_client.models import Distance, VectorParams

        name = self._config.collection
        collections = {c.name for c in self._client.get_collections().collections}
        if name not in collections:
            # Prefer configured size; otherwise infer from first upsert batch.
            dim = self._config.vector_size or vector_dim
            self._client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
        self._collection_ready = True

    def upsert(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> None:
        from qdrant_client.models import PointStruct

        if embeddings:
            self._ensure_collection(len(embeddings[0]))

        points: list[PointStruct] = []
        for point_id, text, vector, meta in zip(ids, texts, embeddings, metadatas):
            payload = {**meta, "text": text}
            points.append(
                PointStruct(id=point_id, vector=vector, payload=payload)
            )
        self._client.upsert(collection_name=self._config.collection, points=points)

    def similarity_search(
        self,
        vector: list[float],
        k: int,
        filters: dict | None = None,
    ) -> list[SearchHit]:
        q_filter = _filters_to_qdrant(filters) if filters else None
        results = self._client.search(
            collection_name=self._config.collection,
            query_vector=vector,
            limit=k,
            query_filter=q_filter,
        )
        hits: list[SearchHit] = []
        for point in results:
            payload = point.payload or {}
            text = str(payload.pop("text", ""))
            hits.append(
                SearchHit(
                    id=str(point.id),
                    text=text,
                    score=qdrant_cosine_similarity(float(point.score)),
                    metadata=dict(payload),
                )
            )
        return hits

    def delete_by_source(self, source: str) -> None:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        self._client.delete(
            collection_name=self._config.collection,
            points_selector=Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=source))]
            ),
        )

    def count(self) -> int:
        info = self._client.get_collection(self._config.collection)
        return int(info.points_count or 0)


def _filters_to_qdrant(filters: dict) -> Any:
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    conditions = [
        FieldCondition(key=key, match=MatchValue(value=value))
        for key, value in filters.items()
    ]
    return Filter(must=conditions)