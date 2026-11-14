# rag-platform

Production-grade, copyable RAG template: ingest local documents, hybrid retrieval, grounded generation with citations, eval harness, Typer CLI, and FastAPI API.

## Quick start

1. **Copy** this repo for a new project.
2. **Install** (Python 3.11+):

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -e ".[dev]"
   ```

3. **Pull Ollama models** (default config uses local Ollama):

   ```bash
   ollama pull nomic-embed-text
   ollama pull llama3.1
   ```

   Ensure Ollama is running at `http://localhost:11434` (or set `OLLAMA_BASE_URL` in `.env`).

4. **Add documents** under `data/raw/` (PDF, Markdown, and plain text). A starter file is included: `data/raw/sample.md`.

5. **Ingest** into the index (Chroma under `data/indexes` by default):

   ```bash
   rag ingest
   ```

6. **Query** from the CLI:

   ```bash
   rag query "What does the sample markdown file mention about retrieval?"
   ```

7. **Evaluate** against the golden set:

   ```bash
   rag eval
   ```

   Writes `evals/report.json` with per-case keyword scores.

8. **Tweak config**, re-ingest if needed, and re-run eval to compare variants (see table below).

9. **Serve** the HTTP API:

   ```bash
   rag serve --host 127.0.0.1 --port 8000
   ```

   - `GET /health` — liveness and index chunk count
   - `POST /ingest` — JSON body `{"path": "data/raw"}`
   - `POST /query` — JSON body `{"question": "..."}`

## Configuration

Edit `configs/default.yaml` or pass `--config path/to.yaml` on CLI commands.

| Concern | Default | Change in config |
|--------|---------|------------------|
| Raw / index paths | `data/raw`, `data/indexes` | `paths.raw`, `paths.index` |
| Chunking | 800 tokens, 120 overlap | `chunking.size`, `chunking.overlap` |
| Embeddings | Ollama `nomic-embed-text` | `embeddings.provider`, `model`, `base_url` |
| LLM | Ollama `llama3.1`, temp 0.1 | `llm.provider`, `model`, `temperature`, `base_url` |
| Vector store | Chroma (`chroma`) | `vector_store.backend`: `chroma` \| `qdrant` \| `pgvector` |
| Retrieval | hybrid, top 8, no rerank | `retrieval.top_k`, `hybrid`, `bm25_weight`, `rerank` |
| Eval golden set | `evals/golden.jsonl` | `eval.golden_path` |
| System prompt | cite-only-from-context | `prompts.system` |

Environment (see `.env.example`):

- `OLLAMA_BASE_URL` — Ollama API base URL
- `OPENAI_API_KEY`, `OPENAI_BASE_URL` — when using `openai_compatible` providers

### Optional extras

```bash
pip install -e ".[qdrant]"      # Qdrant backend
pip install -e ".[pgvector]"    # Postgres pgvector backend
pip install -e ".[rerank]"      # FlashRank reranking (warn-and-skip if missing)
```

## Optional Docker services

The default path uses **embedded Chroma** — no database container required.

For **Qdrant** or **pgvector**, start backing services:

```bash
docker compose up -d qdrant    # REST/gRPC on localhost:6333
docker compose up -d postgres  # postgresql://rag:rag@localhost:5432/rag
```

Then set in `configs/default.yaml`:

**Qdrant**

```yaml
vector_store:
  backend: qdrant
  collection: rag_docs
  url: http://localhost:6333
  # vector_size: 768   # optional; inferred on first upsert if omitted
```

**pgvector**

```yaml
vector_store:
  backend: pgvector
  collection: rag_docs
  dsn: postgresql://rag:rag@localhost:5432/rag
```

Re-run `rag ingest` after switching backends.

## CLI reference

| Command | Purpose |
|---------|---------|
| `rag ingest [--path DIR] [--config FILE]` | Load, chunk, embed, and index documents |
| `rag query QUESTION [--config FILE] [--top-k N]` | Retrieve context and generate an answer |
| `rag eval [--config FILE] [--out FILE]` | Score golden questions; default output `evals/report.json` |
| `rag serve [--host HOST] [--port PORT] [--config FILE]` | Run FastAPI with ingest, query, and health routes |

## Development

```bash
pytest -v
```

Integration tests mock embeddings and LLMs where external services are unavailable.

## Template workflow

Copy → install → pull models → drop docs in `data/raw/` → `rag ingest` → `rag query` → adjust `configs/default.yaml` → `rag eval` → keep changes that improve scores → optionally `rag serve` or swap vector store via config and optional `docker compose`.