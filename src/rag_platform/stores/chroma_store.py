from __future__ import annotations

import chromadb

from rag_platform.config import Settings
from rag_platform.stores.base import SearchHit
from rag_platform.stores.scoring import cosine_distance_to_similarity


class ChromaVectorStore:
    def __init__(self, settings: Settings) -> None:
        self._client = chromadb.PersistentClient(path=str(settings.paths.index))
        self._collection = self._client.get_or_create_collection(
            name=settings.vector_store.collection,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> None:
        self._collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def similarity_search(
        self,
        vector: list[float],
        k: int,
        filters: dict | None = None,
    ) -> list[SearchHit]:
        result = self._collection.query(
            query_embeddings=[vector],
            n_results=k,
            where=filters,
            include=["documents", "metadatas", "distances"],
        )
        ids = result.get("ids") or [[]]
        documents = result.get("documents") or [[]]
        metadatas = result.get("metadatas") or [[]]
        distances = result.get("distances") or [[]]
        hits: list[SearchHit] = []
        for row_id, text, meta, dist in zip(
            ids[0], documents[0], metadatas[0], distances[0]
        ):
            score = cosine_distance_to_similarity(float(dist))
            hits.append(
                SearchHit(
                    id=row_id,
                    text=text or "",
                    score=score,
                    metadata=meta or {},
                )
            )
        return hits

    def delete_by_source(self, source: str) -> None:
        self._collection.delete(where={"source": source})

    def count(self) -> int:
        return self._collection.count()