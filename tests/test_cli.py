from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from rag_platform.cli import app
from rag_platform.eval import EvalReport
from rag_platform.models import EmptyIndexError, IngestResult, QueryResult


def _minimal_config_yaml(*, raw: str = "data/raw", index: str = "data/indexes") -> str:
    return f"""
paths:
  raw: {raw}
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


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    cfg = tmp_path / "config.yaml"
    raw = tmp_path / "raw"
    raw.mkdir()
    cfg.write_text(_minimal_config_yaml(raw=str(raw)), encoding="utf-8")
    return cfg


def test_ingest_invokes_pipeline(runner: CliRunner, config_file: Path, monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[Path, object]] = []

    def fake_ingest(path: Path, settings, **kwargs):
        calls.append((path, settings))
        return IngestResult(files=2, chunks=5, errors=[])

    monkeypatch.setattr("rag_platform.cli.ingest_path", fake_ingest)
    monkeypatch.setattr("rag_platform.cli.load_settings", lambda p=None: __import__(
        "rag_platform.config", fromlist=["load_settings"]
    ).load_settings(config_file))

    result = runner.invoke(app, ["ingest", "--config", str(config_file)])
    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert calls[0][0] == config_file.parent / "raw"
    assert "files=2" in result.output or "2" in result.output


def test_ingest_custom_path(runner: CliRunner, config_file: Path, monkeypatch: pytest.MonkeyPatch):
    custom = config_file.parent / "docs"
    custom.mkdir()
    seen: list[Path] = []

    def fake_ingest(path: Path, settings, **kwargs):
        seen.append(path)
        return IngestResult(files=1, chunks=1, errors=[])

    monkeypatch.setattr("rag_platform.cli.ingest_path", fake_ingest)
    monkeypatch.setattr("rag_platform.cli.load_settings", lambda p=None: __import__(
        "rag_platform.config", fromlist=["load_settings"]
    ).load_settings(config_file))

    result = runner.invoke(
        app,
        ["ingest", "--path", str(custom), "--config", str(config_file)],
    )
    assert result.exit_code == 0, result.output
    assert seen == [custom]


def test_query_prints_citations(runner: CliRunner, config_file: Path, monkeypatch: pytest.MonkeyPatch):
    from rag_platform.models import Citation

    def fake_query(question: str, settings, *, top_k=None, **kwargs):
        return QueryResult(
            answer="done",
            citations=[
                Citation(source="doc.txt", score=0.9, text="snippet text"),
            ],
        )

    monkeypatch.setattr("rag_platform.cli.run_query", fake_query)
    monkeypatch.setattr(
        "rag_platform.cli.load_settings",
        lambda p=None: __import__(
            "rag_platform.config", fromlist=["load_settings"]
        ).load_settings(config_file),
    )

    result = runner.invoke(app, ["query", "hi?", "--config", str(config_file)])
    assert result.exit_code == 0, result.output
    assert "done" in result.output
    assert "doc.txt" in result.output


def test_query_empty_index_exit_1(runner: CliRunner, config_file: Path, monkeypatch: pytest.MonkeyPatch):
    def fake_query(question: str, settings, *, top_k=None, **kwargs):
        raise EmptyIndexError("Index is empty. Run ingest first.")

    monkeypatch.setattr("rag_platform.cli.run_query", fake_query)
    monkeypatch.setattr(
        "rag_platform.cli.load_settings",
        lambda p=None: __import__(
            "rag_platform.config", fromlist=["load_settings"]
        ).load_settings(config_file),
    )

    result = runner.invoke(app, ["query", "hi?", "--config", str(config_file)])
    assert result.exit_code == 1
    assert "empty" in result.output.lower()


def test_query_invokes_pipeline(runner: CliRunner, config_file: Path, monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    def fake_query(question: str, settings, *, top_k=None, **kwargs):
        captured["question"] = question
        captured["top_k"] = top_k
        return QueryResult(answer="42", citations=[])

    monkeypatch.setattr("rag_platform.cli.run_query", fake_query)
    monkeypatch.setattr("rag_platform.cli.load_settings", lambda p=None: __import__(
        "rag_platform.config", fromlist=["load_settings"]
    ).load_settings(config_file))

    result = runner.invoke(
        app,
        ["query", "What is the answer?", "--config", str(config_file), "--top-k", "3"],
    )
    assert result.exit_code == 0, result.output
    assert captured["question"] == "What is the answer?"
    assert captured["top_k"] == 3
    assert "42" in result.output


def test_eval_writes_report(runner: CliRunner, config_file: Path, monkeypatch: pytest.MonkeyPatch):
    report = EvalReport(results=[], mean_keyword_score=0.9)
    out_path = config_file.parent / "out.json"
    written: list[tuple[EvalReport, Path]] = []

    monkeypatch.setattr("rag_platform.cli.run_eval", lambda settings: report)
    monkeypatch.setattr(
        "rag_platform.cli.write_report",
        lambda r, p: written.append((r, p)),
    )
    monkeypatch.setattr("rag_platform.cli.load_settings", lambda p=None: __import__(
        "rag_platform.config", fromlist=["load_settings"]
    ).load_settings(config_file))

    result = runner.invoke(
        app,
        ["eval", "--config", str(config_file), "--out", str(out_path)],
    )
    assert result.exit_code == 0, result.output
    assert written == [(report, out_path)]


def test_serve_starts_uvicorn(runner: CliRunner, config_file: Path, monkeypatch: pytest.MonkeyPatch):
    uvicorn_calls: list[dict] = []
    fastapi_app = MagicMock(name="fastapi_app")

    def fake_create_app(settings=None):
        assert settings is not None
        return fastapi_app

    def fake_uvicorn_run(app_obj, *, host, port):
        uvicorn_calls.append({"app": app_obj, "host": host, "port": port})

    monkeypatch.setattr("rag_platform.cli.create_app", fake_create_app)
    monkeypatch.setattr("rag_platform.cli.uvicorn.run", fake_uvicorn_run)
    monkeypatch.setattr("rag_platform.cli.load_settings", lambda p=None: __import__(
        "rag_platform.config", fromlist=["load_settings"]
    ).load_settings(config_file))

    result = runner.invoke(
        app,
        ["serve", "--config", str(config_file), "--host", "0.0.0.0", "--port", "9000"],
    )
    assert result.exit_code == 0, result.output
    assert len(uvicorn_calls) == 1
    assert uvicorn_calls[0]["app"] is fastapi_app
    assert uvicorn_calls[0]["host"] == "0.0.0.0"
    assert uvicorn_calls[0]["port"] == 9000