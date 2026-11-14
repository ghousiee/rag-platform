from pathlib import Path

import pytest

from rag_platform.config import Settings, load_settings
from rag_platform.stores import build_store


def _minimal_config_yaml(
    *,
    index: str = "data/indexes",
    backend: str = "chroma",
    vector_store_extra: str = "",
) -> str:
    return f"""
paths:
  raw: data/raw
  index: {index}
chunking:
  size: 800
  overlap: 120
embeddings:
  provider: ollama
  model: nomic-embed-text
  base_url: http://localhost:11434
llm:
  provider: ollama
  model: llama3.1
  temperature: 0.1
  base_url: http://localhost:11434
vector_store:
  backend: {backend}
  collection: rag_docs
{vector_store_extra}
retrieval:
  top_k: 8
  hybrid: true
  bm25_weight: 0.3
  rerank: false
prompts:
  system: "Answer using only the provided context."
eval:
  golden_path: evals/golden.jsonl
"""


def make_settings(tmp_path: Path, *, index: Path | None = None) -> Settings:
    idx = index or (tmp_path / "idx")
    cfg = tmp_path / "config.yaml"
    cfg.write_text(_minimal_config_yaml(index=str(idx)))
    return load_settings(cfg)


def test_chroma_empty_filters_dict_treated_as_none(tmp_path: Path):
    settings = make_settings(tmp_path, index=tmp_path / "idx")
    store = build_store(settings)
    store.upsert(
        ids=["1"],
        texts=["hello world"],
        embeddings=[[0.1, 0.2, 0.3]],
        metadatas=[{"source": "a.txt"}],
    )
    hits = store.similarity_search([0.1, 0.2, 0.3], k=1, filters={})
    assert len(hits) == 1


def test_chroma_upsert_and_search(tmp_path: Path):
    settings = make_settings(tmp_path, index=tmp_path / "idx")
    store = build_store(settings)
    store.upsert(
        ids=["1"],
        texts=["hello world"],
        embeddings=[[0.1, 0.2, 0.3]],
        metadatas=[{"source": "a.txt"}],
    )
    hits = store.similarity_search([0.1, 0.2, 0.3], k=1)
    assert hits[0].text == "hello world"
    store.delete_by_source("a.txt")
    assert store.count() == 0


def test_build_store_unknown_backend(tmp_path: Path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(_minimal_config_yaml(backend="weaviate"))
    settings = load_settings(cfg)
    with pytest.raises(ValueError, match="vector store"):
        build_store(settings)


def test_build_store_qdrant_missing_url(tmp_path: Path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(_minimal_config_yaml(backend="qdrant"))
    settings = load_settings(cfg)
    with pytest.raises(ValueError, match="url"):
        build_store(settings)


def test_build_store_qdrant_missing_extra(monkeypatch, tmp_path: Path):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "qdrant_client" or name.startswith("qdrant_client."):
            raise ImportError("No module named 'qdrant_client'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        _minimal_config_yaml(
            backend="qdrant",
            vector_store_extra="  url: http://localhost:6333\n",
        )
    )
    settings = load_settings(cfg)
    with pytest.raises(ImportError, match="qdrant"):
        build_store(settings)


def test_build_store_pgvector_missing_dsn(tmp_path: Path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(_minimal_config_yaml(backend="pgvector"))
    settings = load_settings(cfg)
    with pytest.raises(ValueError, match="dsn"):
        build_store(settings)


def test_build_store_pgvector_missing_extra(monkeypatch, tmp_path: Path):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in ("psycopg", "pgvector") or name.startswith(("psycopg.", "pgvector.")):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        _minimal_config_yaml(
            backend="pgvector",
            vector_store_extra="  dsn: postgresql://localhost/rag\n",
        )
    )
    settings = load_settings(cfg)
    with pytest.raises(ImportError, match="pgvector"):
        build_store(settings)