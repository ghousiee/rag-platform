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