from pathlib import Path

from rag_platform.config import load_settings


def test_load_default_settings(tmp_path: Path):
    cfg = tmp_path / "default.yaml"
    cfg.write_text(
        """
paths:
  raw: data/raw
  index: data/indexes
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
    )
    settings = load_settings(cfg)
    assert settings.chunking.size == 800
    assert settings.embeddings.provider == "ollama"
    assert settings.vector_store.backend == "chroma"
    assert settings.retrieval.hybrid is True


def _minimal_config_yaml() -> str:
    return """
paths:
  raw: data/raw
  index: data/indexes
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


def test_env_overrides_ollama_base_url(tmp_path: Path, monkeypatch):
    cfg = tmp_path / "default.yaml"
    cfg.write_text(_minimal_config_yaml())
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.test:11434")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    settings = load_settings(cfg)

    assert settings.embeddings.base_url == "http://ollama.test:11434"
    assert settings.llm.base_url == "http://ollama.test:11434"


def test_env_overrides_openai(tmp_path: Path, monkeypatch):
    cfg = tmp_path / "openai.yaml"
    cfg.write_text(
        _minimal_config_yaml().replace("provider: ollama", "provider: openai_compatible")
    )
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.test/v1")

    settings = load_settings(cfg)

    assert settings.embeddings.base_url == "https://api.openai.test/v1"
    assert settings.llm.base_url == "https://api.openai.test/v1"
    assert settings.embeddings.api_key == "sk-test-key"
    assert settings.llm.api_key == "sk-test-key"


def test_load_dotenv_invoked(monkeypatch):
    called = {"ok": False}

    def fake_load_dotenv(*args, **kwargs):
        called["ok"] = True

    monkeypatch.setattr("rag_platform.config.load_dotenv", fake_load_dotenv)
    load_settings()
    assert called["ok"] is True


def test_env_overrides_vector_store(tmp_path: Path, monkeypatch):
    cfg = tmp_path / "default.yaml"
    cfg.write_text(
        """
paths:
  raw: data/raw
  index: data/indexes
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
  backend: qdrant
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
    )
    monkeypatch.setenv("QDRANT_URL", "http://qdrant.test:6333")
    monkeypatch.setenv("QDRANT_API_KEY", "secret-key")
    monkeypatch.setenv("PGVECTOR_DSN", "postgresql://db/rag")

    settings = load_settings(cfg)

    assert settings.vector_store.url == "http://qdrant.test:6333"
    assert settings.vector_store.api_key == "secret-key"
    assert settings.vector_store.dsn == "postgresql://db/rag"