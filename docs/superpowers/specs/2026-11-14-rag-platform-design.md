# RAG Platform Template — Design Spec

**Date:** 2026-11-14  
**Status:** Approved for implementation planning  
**Repo:** `rag-platform`

## 1. Goal

Build a production-grade, copyable RAG template: drop documents, run ingest, query via CLI or HTTP, tune accuracy through config, and fork the repo for each new project with minimal code changes.

## 2. Non-goals (v1)

- Chat UI / frontend
- Multi-tenant SaaS auth and billing
- Agent orchestration beyond single-turn RAG (no multi-agent LangGraph flows in v1)
- Real-time document sync / watchers
- Fine-tuning or training pipelines

## 3. Approach

**Monolith config + LangChain RAG:** one Python package with a shared pipeline core; CLI and FastAPI are thin entrypoints. Project-specific behavior lives in `config.yaml` and `data/`, not scattered code. Vector stores and LLM/embedding providers are adapters behind interfaces.

## 4. Architecture

```
config.yaml ──► CLI (ingest|query|eval|serve)
            └─► FastAPI (POST /ingest, POST /query, GET /health)
                      │
                      ▼
               Pipeline core
        ┌─────────────┼─────────────┐
     Loaders      Providers      Stores
   PDF/MD/TXT    Ollama (def)   Chroma (def)
   + registry    OpenAI-compat  Qdrant / pgvector
        │             │              │
   Ingest: load→chunk→embed→upsert   │
   Query:  retrieve (±hybrid ±rerank)→LLM→answer+citations
   Eval:   golden Q&A → same query path → scores/report
```

### Package layout

```
src/rag_platform/
  config.py           # typed settings from YAML + env
  loaders/            # registry + pdf/md/txt
  chunking.py
  providers/          # embeddings + llm adapters
  stores/             # chroma, qdrant, pgvector
  retrieval/          # dense, hybrid, rerank
  generation.py       # RAG chain
  ingest.py
  eval.py
  api/                # FastAPI app
  cli.py
configs/default.yaml
data/raw/
data/indexes/
evals/golden.jsonl
tests/
docker-compose.yml    # optional Qdrant / Postgres
```

## 5. Components & interfaces

### Config
- Load `configs/default.yaml` (or path override) into typed settings (Pydantic).
- Env overrides for secrets and URLs (`OLLAMA_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_BASE_URL`, DB URLs).
- Owns paths, chunking, embeddings, LLM, vector store, retrieval, prompts, eval paths.

### Loaders (registry)
- Interface: `load(path: Path) -> list[Document]` with `text` + `metadata` (source, title, …).
- Built-in: PDF, Markdown, plain text.
- Extensibility: register by extension/MIME for DOCX/HTML/URL loaders later.

### Chunker
- Configurable `size` and `overlap`.
- Default strategy: recursive character splitter (LangChain).
- Propagate source metadata to every chunk.

### Embedding provider
- Interface: `embed_documents(texts) -> list[vector]`, `embed_query(text) -> vector`.
- Adapters: Ollama, OpenAI-compatible (`base_url` + API key).

### LLM provider
- Interface: `generate(messages, **kwargs) -> str`.
- Adapters: Ollama, OpenAI-compatible.
- Prompt template in config (system + context + question).

### Vector store
- Interface: `upsert(...)`, `similarity_search(vector, k, filter?)`, optional `delete_by_source`, `stats`.
- Adapters: Chroma (default, local), Qdrant, Postgres + pgvector.

### Retriever
1. Dense vector search  
2. Optional BM25 hybrid merge via `rank_bm25` (configurable weight)  
3. Optional reranker (off by default): FlashRank when available; if the dependency is missing, log a warning and skip rerank rather than failing the query  
4. Optional metadata filters from the request  

Returns ranked chunks with scores for citations.

### Generation
- LangChain RAG chain: retrieve → format context → LLM → answer + citation metadata.
- Shared by CLI and API.

### Ingestion
- Input: `data/raw` or explicit paths.
- Flow: load → chunk → embed → upsert.
- Idempotency: re-ingest of a source replaces that source’s chunks when the store supports delete-by-source; otherwise document the limitation per backend.

### Eval harness
- Input: `evals/golden.jsonl` records with `question` and either `expected_keywords` or `reference_answer`.
- Runs the live query pipeline; scores keyword coverage and optional simple overlap vs reference.
- CLI writes a JSON/Markdown report for comparing config variants.

### Entry points
- CLI: `ingest`, `query`, `eval`, `serve`
- API: `POST /ingest`, `POST /query`, `GET /health` (eval remains CLI-first in v1)

## 6. Config shape (canonical)

```yaml
paths:
  raw: data/raw
  index: data/indexes

chunking:
  size: 800
  overlap: 120

embeddings:
  provider: ollama          # ollama | openai_compatible
  model: nomic-embed-text
  base_url: http://localhost:11434

llm:
  provider: ollama          # ollama | openai_compatible
  model: llama3.1
  temperature: 0.1
  base_url: http://localhost:11434
  # api_key via env when openai_compatible

vector_store:
  backend: chroma           # chroma | qdrant | pgvector
  collection: rag_docs

retrieval:
  top_k: 8
  hybrid: true
  bm25_weight: 0.3
  rerank: false

prompts:
  system: |
    Answer using only the provided context. Cite sources.
    If the context is insufficient, say you do not know.

eval:
  golden_path: evals/golden.jsonl
```

## 7. API contracts

### `POST /ingest`
```json
{ "path": "data/raw" }
```
Response: `{ "files": N, "chunks": M, "errors": [...] }`

### `POST /query`
```json
{
  "question": "...",
  "top_k": null,
  "filters": {}
}
```
Response: `{ "answer": "...", "citations": [{ "source": "...", "score": 0.0, "text": "..." }] }`

### `GET /health`
Reports process up plus reachability checks for configured embedding/LLM endpoints when feasible.

## 8. Error handling

- Empty raw dir / no loadable files → clear error before indexing.
- Query against empty index → clear error advising ingest.
- Provider or store unavailable → fail with actionable message (URL, backend name).
- Ingest: per-file errors, continue remaining files, summarize at end.
- API: structured JSON errors; non-2xx for hard failures.

## 9. Defaults vs swaps

| Concern | Default | Swap via config |
|--------|---------|-----------------|
| Vector DB | Chroma | Qdrant, pgvector |
| LLM | Ollama | OpenAI-compatible `base_url` |
| Embeddings | Ollama | OpenAI-compatible |
| Docs | PDF, MD, TXT | Loader registry |

## 10. Testing

- Unit: config load, chunker, loader registry, hybrid merge scoring.
- Integration: ingest fixture docs → retrieve (mock embeddings/LLM where external deps are unavailable).
- Smoke: CLI ingest/query/eval against fixtures with mocks or local Ollama when present.

## 11. Template usage workflow

1. Copy repo for a new project.
2. Place docs in `data/raw/`.
3. Adjust `configs/default.yaml` (models, store, retrieval knobs).
4. Run ingest → query via CLI or API.
5. Edit config → run eval → keep changes that improve scores.

## 12. Tech stack

- Python 3.11+
- LangChain (RAG plumbing)
- FastAPI + Uvicorn
- Typer CLI
- Chroma (default dependency); Qdrant client and psycopg/pgvector as optional extras
- `rank_bm25` for hybrid; FlashRank optional for rerank
- Pydantic Settings + PyYAML
- pytest
- Packaging: `pyproject.toml` with `rag-platform` console script entrypoint

## 13. Success criteria

- Fresh clone: configure env, drop sample docs, ingest, query successfully with defaults (Chroma + Ollama).
- Switching `vector_store.backend` or `llm.provider` requires config (and optional Docker), not core rewrites.
- Eval CLI produces a comparable score report for two config variants.
- README documents copy → ingest → query → tweak → eval loop.
