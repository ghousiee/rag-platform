from types import SimpleNamespace
from unittest.mock import MagicMock

from rag_platform.config import VectorStoreConfig
from rag_platform.stores.identifiers import sql_index_slug, sql_table_identifier
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
    client.search.return_value = [
        SimpleNamespace(id="1", score=0.92, payload={"text": "hello", "source": "a"}),
    ]
    store = QdrantVectorStore(config, client)
    hits = store.similarity_search([0.1, 0.2], k=1)
    assert len(hits) == 1
    assert hits[0].score == 0.92
    assert hits[0].text == "hello"


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