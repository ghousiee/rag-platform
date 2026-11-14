from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from rag_platform.config import VectorStoreConfig
from rag_platform.stores.identifiers import (
    qdrant_point_id,
    sql_index_slug,
    sql_table_identifier,
)
from rag_platform.stores.qdrant_store import QdrantVectorStore
from rag_platform.stores.scoring import (
    cosine_distance_to_similarity,
    qdrant_cosine_similarity,
)


def test_sql_table_identifier_safe_and_quoted():
    assert sql_table_identifier("rag_docs") == "rag_docs"
    assert sql_table_identifier("rag-docs") == '"rag-docs"'
    assert sql_table_identifier('say "hi"') == '"say ""hi"""'


def test_sql_index_slug():
    assert sql_index_slug("rag-docs") == "rag_docs"
    assert sql_index_slug("9bad") == "t_9bad"


def test_cosine_distance_to_similarity():
    assert cosine_distance_to_similarity(0.2) == 0.8


def test_qdrant_search_uses_cosine_similarity_not_inverse():
    config = VectorStoreConfig(
        backend="qdrant",
        collection="rag_docs",
        url="http://localhost:6333",
    )
    client = MagicMock()
    client.query_points.return_value = SimpleNamespace(
        points=[
            SimpleNamespace(
                id=qdrant_point_id("src::0"),
                score=0.92,
                payload={"text": "hello", "source": "a", "chunk_id": "src::0"},
            ),
        ]
    )
    store = QdrantVectorStore(config, client)
    hits = store.similarity_search([0.1, 0.2], k=1)
    assert len(hits) == 1
    assert hits[0].score == 0.92
    assert hits[0].text == "hello"
    assert hits[0].id == "src::0"


def test_qdrant_lazy_collection_uses_embedding_dim():
    config = VectorStoreConfig(
        backend="qdrant",
        collection="rag_docs",
        url="http://localhost:6333",
    )
    client = MagicMock()
    client.get_collections.return_value = SimpleNamespace(collections=[])
    store = QdrantVectorStore(config, client)
    store.upsert(
        ids=["1"],
        texts=["x"],
        embeddings=[[0.0] * 128],
        metadatas=[{}],
    )
    client.create_collection.assert_called_once()
    call_kwargs = client.create_collection.call_args.kwargs
    assert call_kwargs["vectors_config"].size == 128


def test_qdrant_point_id_stable_uuid():
    a = qdrant_point_id("doc.txt::0")
    b = qdrant_point_id("doc.txt::0")
    c = qdrant_point_id("doc.txt::1")
    assert a == b
    assert a != c


@pytest.mark.parametrize("method", ["delete_by_source", "count"])
def test_qdrant_missing_collection_no_error(method):
    config = VectorStoreConfig(
        backend="qdrant",
        collection="rag_docs",
        url="http://localhost:6333",
    )
    client = MagicMock()
    client.get_collections.return_value = SimpleNamespace(collections=[])
    store = QdrantVectorStore(config, client)
    if method == "delete_by_source":
        store.delete_by_source("missing.txt")
        client.delete.assert_not_called()
    else:
        assert store.count() == 0
        client.get_collection.assert_not_called()


def test_qdrant_memory_upsert_search_delete_count():
    qdrant_client = pytest.importorskip("qdrant_client")
    from qdrant_client import QdrantClient

    from rag_platform.stores.qdrant_store import QdrantVectorStore

    config = VectorStoreConfig(
        backend="qdrant",
        collection="test_rag",
        url=":memory:",
    )
    store = QdrantVectorStore(config, QdrantClient(":memory:"))
    vec = [1.0, 0.0, 0.0]
    store.upsert(
        ids=["s::0"],
        texts=["alpha"],
        embeddings=[vec],
        metadatas=[{"source": "s"}],
    )
    assert store.count() == 1
    hits = store.similarity_search(vec, k=1)
    assert hits[0].id == "s::0"
    assert hits[0].text == "alpha"
    store.delete_by_source("s")
    assert store.count() == 0
    assert qdrant_client.QdrantClient is not None