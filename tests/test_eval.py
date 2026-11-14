import json
from pathlib import Path

import pytest

from rag_platform.config import Settings, load_settings
from rag_platform.eval import (
    EvalReport,
    run_eval,
    token_jaccard,
    write_report,
)
from rag_platform.models import QueryResult


def _minimal_config_yaml(*, index: str = "data/indexes", golden_path: str = "evals/golden.jsonl") -> str:
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
  golden_path: {golden_path}
"""


def make_settings(tmp_path: Path, *, eval_path: Path | None = None, index: Path | None = None) -> Settings:
    idx = index or (tmp_path / "idx")
    golden = eval_path or (tmp_path / "golden.jsonl")
    cfg = tmp_path / "config.yaml"
    cfg.write_text(_minimal_config_yaml(index=str(idx), golden_path=str(golden)))
    return load_settings(cfg)


def test_run_eval_keyword_scoring(tmp_path: Path):
    golden = tmp_path / "g.jsonl"
    golden.write_text('{"question":"q","expected_keywords":["alpha","beta"]}\n')
    settings = make_settings(tmp_path, eval_path=golden)

    def fake_query(question, settings, **kwargs):
        return QueryResult(answer="alpha and gamma", citations=[])

    report = run_eval(settings, query_fn=fake_query)
    assert len(report.results) == 1
    assert report.results[0].keyword_score == 0.5
    assert report.results[0].passed is True
    assert report.mean_keyword_score == 0.5


def test_keyword_score_one_when_no_keywords(tmp_path: Path):
    golden = tmp_path / "g.jsonl"
    golden.write_text('{"question":"q"}\n')
    settings = make_settings(tmp_path, eval_path=golden)

    def fake_query(question, settings, **kwargs):
        return QueryResult(answer="anything", citations=[])

    report = run_eval(settings, query_fn=fake_query)
    assert report.results[0].keyword_score == 1.0
    assert report.results[0].passed is True


def test_overlap_score_with_reference(tmp_path: Path):
    golden = tmp_path / "g.jsonl"
    golden.write_text(
        '{"question":"q","reference_answer":"the cat sat on the mat"}\n'
    )
    settings = make_settings(tmp_path, eval_path=golden)

    def fake_query(question, settings, **kwargs):
        return QueryResult(answer="the cat sat", citations=[])

    report = run_eval(settings, query_fn=fake_query)
    assert report.results[0].overlap_score is not None
    assert report.results[0].overlap_score == pytest.approx(token_jaccard("the cat sat", "the cat sat on the mat"))


def test_keyword_scoring_case_insensitive(tmp_path: Path):
    golden = tmp_path / "g.jsonl"
    golden.write_text('{"question":"q","expected_keywords":["RAG"]}\n')
    settings = make_settings(tmp_path, eval_path=golden)

    def fake_query(question, settings, **kwargs):
        return QueryResult(answer="mentions rag retrieval", citations=[])

    report = run_eval(settings, query_fn=fake_query)
    assert report.results[0].keyword_score == 1.0


def test_write_report_roundtrip(tmp_path: Path):
    report = EvalReport(
        results=[],
        mean_keyword_score=1.0,
    )
    out = tmp_path / "report.json"
    write_report(report, out)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["mean_keyword_score"] == 1.0
    assert data["results"] == []


def test_golden_jsonl_exists_in_repo():
    golden = Path(__file__).resolve().parents[1] / "evals" / "golden.jsonl"
    assert golden.is_file()
    lines = [ln for ln in golden.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert 2 <= len(lines) <= 3
    for line in lines:
        obj = json.loads(line)
        assert "question" in obj