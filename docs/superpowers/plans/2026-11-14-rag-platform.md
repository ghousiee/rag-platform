# RAG Platform Template Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a production-grade, copyable RAG template: config-driven ingest → query via CLI/FastAPI, with pluggable stores/providers, hybrid retrieval, and an eval harness.

**Architecture:** One Python package (`src/rag_platform`) owns the pipeline; Typer CLI and FastAPI are thin entrypoints sharing the same services. Adapters behind interfaces cover Ollama / OpenAI-compatible providers and Chroma / Qdrant / pgvector stores. All project knobs live in `configs/default.yaml`.

**Tech Stack:** Python 3.11+, LangChain, FastAPI, Uvicorn, Typer, Pydantic Settings, PyYAML, Chroma, rank_bm25, pytest; optional Qdrant, psycopg/pgvector, FlashRank.

## Global Constraints

- Python 3.11+
- Defaults: Chroma + Ollama (chat + embeddings)
- Providers: `ollama` | `openai_compatible` only in v1
- Stores: `chroma` | `qdrant` | `pgvector`
- Loaders v1: PDF, Markdown, TXT + extension registry
- Eval is CLI-first (no `/eval` API in v1)
- No chat UI; no multi-agent LangGraph flows in v1
- Spec: `docs/superpowers/specs/2026-11-14-rag-platform-design.md`

## File Structure

```
pyproject.toml
README.md
.env.example
docker-compose.yml
configs/default.yaml
data/raw/.gitkeep
data/indexes/.gitkeep
evals/golden.jsonl
src/rag_platform/
  __init__.py
  models.py              # Document, Chunk, Citation, QueryResult
  config.py
  chunking.py
  ingest.py
  generation.py
  eval.py
  cli.py
  loaders/
    __init__.py          # registry + load_path/load_directory
    base.py
    text.py
    markdown.py
    pdf.py
  providers/
    __init__.py          # factories
    base.py
    ollama.py
    openai_compatible.py
  stores/
    __init__.py          # factory
    base.py
    chroma_store.py
    qdrant_store.py
    pgvector_store.py
  retrieval/
    __init__.py
    retriever.py
    hybrid.py
    rerank.py
  api/
    __init__.py
    app.py
    schemas.py
tests/
  conftest.py
  test_config.py
  test_loaders.py
  test_chunking.py
  test_hybrid.py
  test_ingest_query.py
  fixtures/
    sample.md
    sample.txt
```

---

### Task 1: Project scaffolding + config

**Files:**
- Create: `pyproject.toml`, `src/rag_platform/__init__.py`, `src/rag_platform/config.py`, `src/rag_platform/models.py`, `configs/default.yaml`, `.env.example`, `tests/conftest.py`, `tests/test_config.py`
- Modify: `README.md` (minimal placeholder until Task 10)

**Interfaces:**
- Produces: `load_settings(config_path: Path | None = None) -> Settings`
- Produces: Pydantic models `PathsConfig`, `ChunkingConfig`, `EmbeddingsConfig`, `LLMConfig`, `VectorStoreConfig`, `RetrievalConfig`, `PromptsConfig`, `EvalConfig`, `Settings`
- Produces: `Document(text: str, metadata: dict[str, Any])`

- [ ] **Step 1: Write failing config test**

```python
# tests/test_config.py
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
```

- [ ] **Step 2: Run test — expect fail**

Run: `pytest tests/test_config.py -v`  
Expected: FAIL (module not found / `load_settings` missing)

- [ ] **Step 3: Add `pyproject.toml` and implement config**

```toml
# pyproject.toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "rag-platform"
version = "0.1.0"
description = "Production-grade, copyable RAG template"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
  "pydantic>=2.6",
  "pydantic-settings>=2.2",
  "pyyaml>=6.0",
  "langchain>=0.3",
  "langchain-community>=0.3",
  "langchain-text-splitters>=0.3",
  "langchain-ollama>=0.2",
  "langchain-openai>=0.2",
  "chromadb>=0.5",
  "rank-bm25>=0.2",
  "pypdf>=4.0",
  "fastapi>=0.115",
  "uvicorn[standard]>=0.30",
  "typer>=0.12",
  "httpx>=0.27",
  "python-dotenv>=1.0",
]

[project.optional-dependencies]
qdrant = ["qdrant-client>=1.11"]
pgvector = ["psycopg[binary]>=3.2", "pgvector>=0.3"]
rerank = ["FlashRank>=0.2"]
dev = ["pytest>=8.0", "pytest-asyncio>=0.24"]

[project.scripts]
rag = "rag_platform.cli:app"

[tool.hatch.build.targets.wheel]
packages = ["src/rag_platform"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

Implement `src/rag_platform/models.py` with `Document`, and `src/rag_platform/config.py` with nested Pydantic models + `load_settings` reading YAML (resolve relative paths against repo root / cwd). Ship `configs/default.yaml` matching the spec §6. Ship `.env.example` with `OLLAMA_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_BASE_URL`.

- [ ] **Step 4: Install editable + run test**

Run: `pip install -e ".[dev]" && pytest tests/test_config.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml configs/default.yaml .env.example src/rag_platform/__init__.py src/rag_platform/models.py src/rag_platform/config.py tests/test_config.py tests/conftest.py
git commit -m "feat: scaffold project and typed YAML config"
```

---

### Task 2: Loaders + chunking

**Files:**
- Create: `src/rag_platform/loaders/{__init__,base,text,markdown,pdf}.py`, `src/rag_platform/chunking.py`, `tests/test_loaders.py`, `tests/test_chunking.py`, `tests/fixtures/sample.md`, `tests/fixtures/sample.txt`

**Interfaces:**
- Consumes: `Document`, `Settings.chunking`
- Produces: `register_loader(ext: str, loader: Loader)`, `load_path(path: Path) -> list[Document]`, `load_directory(path: Path) -> list[Document]`
- Produces: `chunk_documents(docs: list[Document], size: int, overlap: int) -> list[Document]` (chunks keep `metadata` + `metadata["chunk_index"]`)

- [ ] **Step 1: Write failing loader + chunker tests**

```python
# tests/test_loaders.py
from pathlib import Path
from rag_platform.loaders import load_path, load_directory

def test_load_markdown(fixtures_dir: Path):
    docs = load_path(fixtures_dir / "sample.md")
    assert len(docs) == 1
    assert "RAG" in docs[0].text
    assert docs[0].metadata["source"].endswith("sample.md")

def test_load_directory_skips_unknown(fixtures_dir: Path):
    docs = load_directory(fixtures_dir)
    assert len(docs) >= 2
```

```python
# tests/test_chunking.py
from rag_platform.models import Document
from rag_platform.chunking import chunk_documents

def test_chunk_preserves_metadata_and_overlap():
    doc = Document(text="a" * 1000, metadata={"source": "x.txt"})
    chunks = chunk_documents([doc], size=200, overlap=50)
    assert len(chunks) > 1
    assert all(c.metadata["source"] == "x.txt" for c in chunks)
    assert chunks[0].metadata["chunk_index"] == 0
```

Add `fixtures_dir` fixture in `tests/conftest.py`. Create sample fixture files.

- [ ] **Step 2: Run tests — expect fail**

Run: `pytest tests/test_loaders.py tests/test_chunking.py -v`  
Expected: FAIL

- [ ] **Step 3: Implement loaders + chunker**

- `Loader` protocol / ABC with `load(path) -> list[Document]`
- Text/Markdown: read UTF-8; PDF: `pypdf`
- Registry maps `.txt`, `.md`, `.markdown`, `.pdf`
- Unknown extensions skipped in `load_directory` (logged)
- Chunker: `langchain_text_splitters.RecursiveCharacterTextSplitter`

- [ ] **Step 4: Run tests — expect pass**

Run: `pytest tests/test_loaders.py tests/test_chunking.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/rag_platform/loaders src/rag_platform/chunking.py tests/test_loaders.py tests/test_chunking.py tests/fixtures tests/conftest.py
git commit -m "feat: add document loaders registry and chunker"
```

---

### Task 3: Providers (embeddings + LLM)

**Files:**
- Create: `src/rag_platform/providers/{__init__,base,ollama,openai_compatible}.py`, `tests/test_providers.py`

**Interfaces:**
- Consumes: `Settings.embeddings`, `Settings.llm`
- Produces:
  - `EmbeddingProvider`: `embed_documents(texts: list[str]) -> list[list[float]]`, `embed_query(text: str) -> list[float]`
  - `LLMProvider`: `generate(messages: list[dict[str, str]], temperature: float | None = None) -> str`
  - `build_embedder(settings) -> EmbeddingProvider`
  - `build_llm(settings) -> LLMProvider`

- [ ] **Step 1: Write failing factory tests with fakes**

```python
# tests/test_providers.py
from rag_platform.config import load_settings
from rag_platform.providers import build_embedder, build_llm

def test_build_embedder_ollama(tmp_path, monkeypatch):
    # write minimal yaml with embeddings.provider=ollama
    settings = load_settings(...)  # use helper or inline yaml from Task 1
    embedder = build_embedder(settings)
    assert embedder.__class__.__name__ == "OllamaEmbeddingsProvider"

def test_build_llm_openai_compatible(tmp_path):
    settings = ...  # provider openai_compatible
    llm = build_llm(settings)
    assert llm.__class__.__name__ == "OpenAICompatibleLLMProvider"
```

Also add a unit test that a hand-written `FakeEmbeddingProvider` satisfies the interface used by later tasks (duck typing / Protocol).

- [ ] **Step 2: Run — expect fail**

- [ ] **Step 3: Implement adapters**

- Ollama: wrap `langchain_ollama.OllamaEmbeddings` / `ChatOllama`
- OpenAI-compatible: wrap `langchain_openai.OpenAIEmbeddings` / `ChatOpenAI` with `base_url` + `api_key` from settings/env
- Factories raise `ValueError` on unknown provider names with actionable message

- [ ] **Step 4: Run — expect pass**

- [ ] **Step 5: Commit**

```bash
git commit -m "feat: add Ollama and OpenAI-compatible providers"
```

---

### Task 4: Vector stores (Chroma default + Qdrant + pgvector)

**Files:**
- Create: `src/rag_platform/stores/{__init__,base,chroma_store,qdrant_store,pgvector_store}.py`, `tests/test_chroma_store.py`

**Interfaces:**
- Consumes: `Settings.vector_store`, `Settings.paths`, embedder dimension implicitly via vectors
- Produces: `VectorStore` protocol:
  - `upsert(ids: list[str], texts: list[str], embeddings: list[list[float]], metadatas: list[dict]) -> None`
  - `similarity_search(vector: list[float], k: int, filters: dict | None = None) -> list[SearchHit]`
  - `delete_by_source(source: str) -> None`
  - `count() -> int`
- `SearchHit(id: str, text: str, score: float, metadata: dict)`
- `build_store(settings) -> VectorStore`

- [ ] **Step 1: Write Chroma integration test**

```python
def test_chroma_upsert_and_search(tmp_path):
    settings = make_settings(index=tmp_path / "idx")
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
```

- [ ] **Step 2: Run — expect fail**

- [ ] **Step 3: Implement Chroma fully; stub Qdrant/pgvector behind factory**

- Chroma persists under `settings.paths.index`
- Qdrant/pgvector modules implement the same protocol; factory imports optional deps and raises clear `ImportError` / config error if extra not installed or connection settings missing
- Add config fields as needed: `vector_store.url`, `vector_store.api_key`, `vector_store.dsn` (optional, default None)

- [ ] **Step 4: Run Chroma test — pass**  
  Qdrant/pgvector: unit-test factory error messages without live servers.

- [ ] **Step 5: Commit**

```bash
git commit -m "feat: add pluggable vector stores with Chroma default"
```

---

### Task 5: Retrieval (dense + hybrid + optional rerank)

**Files:**
- Create: `src/rag_platform/retrieval/{__init__,retriever,hybrid,rerank}.py`, `tests/test_hybrid.py`, `tests/test_retriever.py`

**Interfaces:**
- Consumes: `VectorStore`, `EmbeddingProvider`, `Settings.retrieval`
- Produces:
  - `fuse_dense_and_bm25(dense_hits, bm25_hits, bm25_weight: float, k: int) -> list[SearchHit]`
  - `Retriever.retrieve(query: str, k: int | None = None, filters: dict | None = None) -> list[SearchHit]`
  - BM25 corpus: built from store texts on init or via `retriever.rebuild_bm25(texts, metadatas, ids)` called after ingest

- [ ] **Step 1: Write hybrid fuse unit test**

```python
from rag_platform.stores.base import SearchHit
from rag_platform.retrieval.hybrid import fuse_dense_and_bm25

def test_fuse_prefers_agreement():
    dense = [SearchHit("1", "a", 0.9, {}), SearchHit("2", "b", 0.5, {})]
    bm25 = [SearchHit("2", "b", 0.8, {}), SearchHit("3", "c", 0.7, {})]
    fused = fuse_dense_and_bm25(dense, bm25, bm25_weight=0.4, k=2)
    assert len(fused) == 2
    assert {h.id for h in fused} <= {"1", "2", "3"}
```

- [ ] **Step 2: Run — fail**

- [ ] **Step 3: Implement RRF-style or weighted score fusion + Retriever**

- If `hybrid=False`: dense only
- If `rerank=True`: try FlashRank; on import failure log warning and return fused hits
- Expose `build_retriever(settings, store, embedder) -> Retriever`

- [ ] **Step 4: Run — pass**

- [ ] **Step 5: Commit**

```bash
git commit -m "feat: add hybrid retrieval and optional rerank"
```

---

### Task 6: Ingest + generation (RAG query)

**Files:**
- Create: `src/rag_platform/ingest.py`, `src/rag_platform/generation.py`, `tests/test_ingest_query.py`

**Interfaces:**
- Produces:
  - `IngestResult(files: int, chunks: int, errors: list[str])`
  - `ingest_path(path: Path, settings: Settings, *, embedder=None, store=None) -> IngestResult`
  - `QueryResult(answer: str, citations: list[Citation])`
  - `Citation(source: str, score: float, text: str)`
  - `query(question: str, settings: Settings, *, top_k: int | None = None, filters: dict | None = None, embedder=None, store=None, llm=None, retriever=None) -> QueryResult`

- [ ] **Step 1: Write integration test with fakes**

```python
class FakeEmbed:
    def embed_documents(self, texts):
        return [[float(len(t)), 0.1, 0.2] for t in texts]
    def embed_query(self, text):
        return [float(len(text)), 0.1, 0.2]

class FakeLLM:
    def generate(self, messages, temperature=None):
        return "ANSWER"

def test_ingest_then_query(tmp_path, fixtures_dir):
    settings = make_chroma_settings(tmp_path)
    # copy fixtures into tmp raw
    result = ingest_path(tmp_path / "raw", settings, embedder=FakeEmbed())
    assert result.chunks > 0
    assert result.errors == []
    qr = query("What is RAG?", settings, embedder=FakeEmbed(), llm=FakeLLM())
    assert qr.answer == "ANSWER"
    assert isinstance(qr.citations, list)
```

Empty index query must raise a clear error.

- [ ] **Step 2: Run — fail**

- [ ] **Step 3: Implement ingest + generation**

- Ingest: load_directory → chunk → delete_by_source per file → embed → upsert; catch per-file errors
- Generation: retrieve → format context with source labels → messages `[system, user]` → llm.generate → map hits to citations
- Empty store: raise `RuntimeError("Index is empty. Run ingest first.")`

- [ ] **Step 4: Run — pass**

- [ ] **Step 5: Commit**

```bash
git commit -m "feat: implement ingest and RAG query pipeline"
```

---

### Task 7: Eval harness

**Files:**
- Create: `src/rag_platform/eval.py`, `evals/golden.jsonl`, `tests/test_eval.py`

**Interfaces:**
- Produces:
  - `GoldenItem(question: str, expected_keywords: list[str] | None = None, reference_answer: str | None = None)`
  - `EvalCaseResult(question: str, answer: str, keyword_score: float, overlap_score: float | None, passed: bool)`
  - `EvalReport(results: list[EvalCaseResult], mean_keyword_score: float)`
  - `run_eval(settings, *, query_fn=query) -> EvalReport`
  - `write_report(report, path: Path) -> None` (JSON)

Scoring:
- `keyword_score` = fraction of `expected_keywords` found case-insensitively in answer (1.0 if no keywords)
- `overlap_score` = token Jaccard vs `reference_answer` when present
- `passed` = `keyword_score >= 0.5` (configurable constant in module is fine for v1)

- [ ] **Step 1: Write test with stub query_fn**

```python
def test_run_eval_keyword_scoring(tmp_path):
    golden = tmp_path / "g.jsonl"
    golden.write_text('{"question":"q","expected_keywords":["alpha","beta"]}\n')
    settings = make_settings(eval_path=golden)
    def fake_query(question, settings, **kwargs):
        return QueryResult(answer="alpha and gamma", citations=[])
    report = run_eval(settings, query_fn=fake_query)
    assert report.results[0].keyword_score == 0.5
```

- [ ] **Step 2–4: Implement, pass, commit**

```bash
git commit -m "feat: add golden-set eval harness"
```

Ship `evals/golden.jsonl` with 2–3 sample questions aligned to fixture docs.

---

### Task 8: CLI (Typer)

**Files:**
- Create: `src/rag_platform/cli.py`, `tests/test_cli.py`

**Interfaces:**
- Console script `rag` with commands:
  - `rag ingest [--path PATH] [--config PATH]`
  - `rag query QUESTION [--config PATH] [--top-k N]`
  - `rag eval [--config PATH] [--out PATH]`
  - `rag serve [--host HOST] [--port PORT] [--config PATH]`

- [ ] **Step 1: Test Typer invoke for ingest/query with monkeypatched pipeline functions**

- [ ] **Step 2–4: Implement CLI wiring `load_settings` → services; commit**

```bash
git commit -m "feat: add Typer CLI for ingest, query, eval, serve"
```

---

### Task 9: FastAPI

**Files:**
- Create: `src/rag_platform/api/__init__.py`, `src/rag_platform/api/app.py`, `src/rag_platform/api/schemas.py`, `tests/test_api.py`

**Interfaces:**
- `create_app(settings: Settings | None = None) -> FastAPI`
- `POST /ingest` body `{ "path": "data/raw" }` → `IngestResult` JSON
- `POST /query` body `{ "question": str, "top_k": int | null, "filters": object }` → `QueryResult` JSON
- `GET /health` → `{ "status": "ok", "index_count": N }` (best-effort; provider ping optional via httpx)

- [ ] **Step 1: TestClient tests with fake embed/llm injected via app dependency overrides or constructor hooks**

Expose `app.state.embedder` / settings injection so tests do not need Ollama.

- [ ] **Step 2–4: Implement, pass, commit**

```bash
git commit -m "feat: add FastAPI ingest, query, and health endpoints"
```

Structured errors: HTTP 400 for empty ingest path / bad input; HTTP 409 or 400 for empty index on query; HTTP 500 with message for provider failures.

---

### Task 10: Docker Compose, sample data, README

**Files:**
- Create: `docker-compose.yml`, `data/raw/sample.md`, `data/raw/.gitkeep`, `data/indexes/.gitkeep`
- Modify: `README.md`

**Contents:**
- `docker-compose.yml`: optional `qdrant` and `postgres` (pgvector image) services; not required for default Chroma path
- README: copy → install → pull Ollama models → drop docs → `rag ingest` → `rag query` → `rag eval` → config tweak table → `rag serve` → extras for Qdrant/pgvector

- [ ] **Step 1: Write README sections matching success criteria in the spec**
- [ ] **Step 2: Add sample doc + compose file**
- [ ] **Step 3: Run full unit suite**

Run: `pytest -v`  
Expected: all PASS

- [ ] **Step 4: Commit + push**

```bash
git add README.md docker-compose.yml data evals
git commit -m "docs: add template README, sample data, and optional compose services"
git push -u origin cursor/rag-platform-template-4334
```

---

## Spec coverage checklist

| Spec section | Task(s) |
|--------------|---------|
| Config YAML + env | 1 |
| Loaders PDF/MD/TXT + registry | 2 |
| Chunker | 2 |
| Ollama + OpenAI-compatible providers | 3 |
| Chroma / Qdrant / pgvector | 4 |
| Hybrid + optional rerank | 5 |
| Ingest + generation + citations | 6 |
| Eval harness | 7 |
| CLI | 8 |
| FastAPI contracts | 9 |
| Error handling | 6, 8, 9 |
| Docker + README workflow | 10 |
| Testing | all tasks |

## Self-review notes

- No TBD placeholders; Typer locked; FlashRank optional with warn-and-skip.
- `SearchHit` / `QueryResult` / `IngestResult` names consistent across tasks.
- Qdrant/pgvector verified via factory + protocol; live server tests not required for v1 CI.
