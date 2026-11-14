from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from rag_platform.config import VectorStoreConfig
from rag_platform.stores.base import SearchHit
from rag_platform.stores.identifiers import normalize_filters, qdrant_point_id
from rag_platform.stores.scoring import qdrant_cosine_similarity

if TYPE_CHECKING:
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct

logger = logging.getLogger(__name__)

_CHUNK_ID_PAYLOAD_KEY = "chunk_id"


class QdrantVectorStore:
    def __init__(self, config: VectorStoreConfig, client: QdrantClient) -> None:
        self._config = config
        self._client = client
        self._collection_ready = False

    def _collection_exists(self) -> bool:
        name = self._config.collection
        collections = {c.name for c in self._client.get_collections().collections}
        return name in collections

    def _ensure_collection(self, vector_dim: int) -> None:
        if self._collection_ready:
            return
        from qdrant_client.models import Distance, VectorParams

        name = self._config.collection
        if not self._collection_exists():
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
        for chunk_id, text, vector, meta in zip(ids, texts, embeddings, metadatas):
            payload = {**meta, "text": text, _CHUNK_ID_PAYLOAD_KEY: chunk_id}
            points.append(
                PointStruct(id=qdrant_point_id(chunk_id), vector=vector, payload=payload)
            )
        self._client.upsert(collection_name=self._config.collection, points=points)

    def similarity_search(
        self,
        vector: list[float],
        k: int,
        filters: dict | None = None,
    ) -> list[SearchHit]:
        q_filter = _filters_to_qdrant(normalize_filters(filters))
        response = self._client.query_points(
            collection_name=self._config.collection,
            query=vector,
            limit=k,
            query_filter=q_filter,
        )
        hits: list[SearchHit] = []
        for point in response.points:
            payload = dict(point.payload or {})
            text = str(payload.pop("text", ""))
            chunk_id = payload.pop(_CHUNK_ID_PAYLOAD_KEY, None)
            hit_id = str(chunk_id) if chunk_id is not None else str(point.id)
            hits.append(
                SearchHit(
                    id=hit_id,
                    text=text,
                    score=qdrant_cosine_similarity(float(point.score)),
                    metadata=payload,
                )
            )
        return hits

    def delete_by_source(self, source: str) -> None:
        if not self._collection_exists():
            return
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        self._client.delete(
            collection_name=self._config.collection,
            points_selector=Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=source))]
            ),
        )

    def count(self) -> int:
        if not self._collection_exists():
            return 0
        info = self._client.get_collection(self._config.collection)
        return int(info.points_count or 0)


def _filters_to_qdrant(filters: dict | None) -> Any:
    if not filters:
        return None
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    conditions = [
        FieldCondition(key=key, match=MatchValue(value=value))
        for key, value in filters.items()
    ]
    return Filter(must=conditions)