from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from rag_platform.config import Settings
from rag_platform.stores.base import SearchHit, VectorStore
from rag_platform.stores.chroma_store import ChromaVectorStore

if TYPE_CHECKING:
    pass

_BACKENDS = ("chroma", "qdrant", "pgvector")


def build_store(settings: Settings) -> VectorStore:
    backend = settings.vector_store.backend
    if backend == "chroma":
        return ChromaVectorStore(settings)
    if backend == "qdrant":
        if not settings.vector_store.url:
            raise ValueError(
                "vector_store.url is required when vector_store.backend is 'qdrant'."
            )
        _require_qdrant_extra()
        from rag_platform.stores.qdrant_store import QdrantVectorStore

        from qdrant_client import QdrantClient

        client = QdrantClient(
            url=settings.vector_store.url,
            api_key=settings.vector_store.api_key,
        )
        return QdrantVectorStore(settings.vector_store, client)
    if backend == "pgvector":
        if not settings.vector_store.dsn:
            raise ValueError(
                "vector_store.dsn is required when vector_store.backend is 'pgvector'."
            )
        _require_pgvector_extra()
        from rag_platform.stores.pgvector_store import PgVectorStore

        return PgVectorStore(settings.vector_store, settings.vector_store.dsn)
    supported = ", ".join(_BACKENDS)
    raise ValueError(
        f"Unknown vector store backend {backend!r}. Supported backends: {supported}."
    )


def _require_qdrant_extra() -> None:
    try:
        importlib.import_module("qdrant_client")
    except ImportError as exc:
        raise ImportError(
            "Qdrant vector store requires the optional 'qdrant' extra. "
            "Install with: pip install 'rag-platform[qdrant]'"
        ) from exc


def _require_pgvector_extra() -> None:
    try:
        importlib.import_module("psycopg")
        importlib.import_module("pgvector")
    except ImportError as exc:
        raise ImportError(
            "pgvector store requires the optional 'pgvector' extra. "
            "Install with: pip install 'rag-platform[pgvector]'"
        ) from exc


__all__ = ["SearchHit", "VectorStore", "build_store"]