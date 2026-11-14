from pathlib import Path

import pytest

from rag_platform.config import load_settings
from rag_platform.providers import build_embedder, build_llm
from rag_platform.providers.base import EmbeddingProvider


def _minimal_config_yaml(
    *,
    embeddings_provider: str = "ollama",
    llm_provider: str = "ollama",
) -> str:
    return f"""
paths:
  raw: data/raw
  index: data/indexes
chunking:
  size: 800
  overlap: 120
embeddings:
  provider: {embeddings_provider}
  model: nomic-embed-text
  base_url: http://localhost:11434
llm:
  provider: {llm_provider}
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


def _write_settings(
    tmp_path: Path,
    *,
    embeddings_provider: str = "ollama",
    llm_provider: str = "ollama",
) -> Path:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        _minimal_config_yaml(
            embeddings_provider=embeddings_provider,
            llm_provider=llm_provider,
        )
    )
    return cfg


def test_build_embedder_ollama(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    settings = load_settings(_write_settings(tmp_path))
    embedder = build_embedder(settings)
    assert embedder.__class__.__name__ == "OllamaEmbeddingsProvider"


def test_build_embedder_openai_compatible(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    settings = load_settings(
        _write_settings(tmp_path, embeddings_provider="openai_compatible")
    )
    embedder = build_embedder(settings)
    assert embedder.__class__.__name__ == "OpenAICompatibleEmbeddingsProvider"


def test_build_llm_ollama(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    settings = load_settings(_write_settings(tmp_path))
    llm = build_llm(settings)
    assert llm.__class__.__name__ == "OllamaLLMProvider"


def test_build_llm_openai_compatible(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    settings = load_settings(
        _write_settings(tmp_path, llm_provider="openai_compatible")
    )
    llm = build_llm(settings)
    assert llm.__class__.__name__ == "OpenAICompatibleLLMProvider"


def test_build_embedder_unknown_provider(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    settings = load_settings(
        _write_settings(tmp_path, embeddings_provider="unknown_vendor")
    )
    with pytest.raises(ValueError, match="embeddings"):
        build_embedder(settings)


def test_build_llm_unknown_provider(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    settings = load_settings(_write_settings(tmp_path, llm_provider="unknown_vendor"))
    with pytest.raises(ValueError, match="llm"):
        build_llm(settings)


class FakeEmbeddingProvider:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(i), 1.0] for i, _ in enumerate(texts)]

    def embed_query(self, text: str) -> list[float]:
        return [0.5, 1.0]


def test_fake_embedding_provider_satisfies_protocol():
    fake = FakeEmbeddingProvider()
    assert isinstance(fake, EmbeddingProvider)