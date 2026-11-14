from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from rag_platform.config import Settings, load_settings
from rag_platform.generation import query
from rag_platform.ingest import ingest_path


class FakeEmbed:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t)), 0.1, 0.2] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 0.1, 0.2]


class FakeLLM:
    def generate(self, messages, temperature=None):
        return "ANSWER"


def _minimal_config_yaml(*, index: str = "data/indexes") -> str:
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
  backend: chroma
  collection: rag_docs
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


def make_chroma_settings(tmp_path: Path) -> Settings:
    idx = tmp_path / "idx"
    cfg = tmp_path / "config.yaml"
    cfg.write_text(_minimal_config_yaml(index=str(idx)))
    return load_settings(cfg)


def test_ingest_then_query(tmp_path: Path, fixtures_dir: Path):
    settings = make_chroma_settings(tmp_path)
    raw = tmp_path / "raw"
    raw.mkdir()
    for name in ("sample.md", "sample.txt"):
        shutil.copy(fixtures_dir / name, raw / name)

    result = ingest_path(raw, settings, embedder=FakeEmbed())
    assert result.chunks > 0
    assert result.errors == []

    qr = query("What is RAG?", settings, embedder=FakeEmbed(), llm=FakeLLM())
    assert qr.answer == "ANSWER"
    assert isinstance(qr.citations, list)
    assert len(qr.citations) > 0


def test_query_empty_index(tmp_path: Path):
    settings = make_chroma_settings(tmp_path)
    with pytest.raises(RuntimeError, match="Index is empty"):
        query("What is RAG?", settings, embedder=FakeEmbed(), llm=FakeLLM())


def test_ingest_empty_directory(tmp_path: Path):
    settings = make_chroma_settings(tmp_path)
    empty = tmp_path / "raw"
    empty.mkdir()
    result = ingest_path(empty, settings, embedder=FakeEmbed())
    assert result.files == 0
    assert result.chunks == 0
    assert len(result.errors) == 1
    assert "loadable" in result.errors[0].lower() or "empty" in result.errors[0].lower()