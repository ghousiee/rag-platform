from __future__ import annotations

import shutil
from pathlib import Path

from fastapi.testclient import TestClient

from rag_platform.api.app import create_app
from tests.test_ingest_query import FakeEmbed, FakeLLM, make_chroma_settings


def test_health_empty_index(tmp_path: Path):
    settings = make_chroma_settings(tmp_path)
    app = create_app(settings, embedder=FakeEmbed(), llm=FakeLLM())
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["index_count"] == 0


def test_ingest_empty_path_returns_400(tmp_path: Path):
    settings = make_chroma_settings(tmp_path)
    app = create_app(settings, embedder=FakeEmbed())
    client = TestClient(app)
    resp = client.post("/ingest", json={"path": ""})
    assert resp.status_code == 400
    assert "detail" in resp.json()


def test_ingest_and_query_round_trip(tmp_path: Path, fixtures_dir: Path):
    settings = make_chroma_settings(tmp_path)
    raw = tmp_path / "raw"
    raw.mkdir()
    for name in ("sample.md", "sample.txt"):
        shutil.copy(fixtures_dir / name, raw / name)

    app = create_app(settings, embedder=FakeEmbed(), llm=FakeLLM())
    client = TestClient(app)

    ingest_resp = client.post("/ingest", json={"path": str(raw)})
    assert ingest_resp.status_code == 200
    body = ingest_resp.json()
    assert body["chunks"] > 0
    assert body["errors"] == []

    health = client.get("/health")
    assert health.json()["index_count"] > 0

    query_resp = client.post(
        "/query",
        json={"question": "What is RAG?", "top_k": 4, "filters": {}},
    )
    assert query_resp.status_code == 200
    qr = query_resp.json()
    assert qr["answer"] == "ANSWER"
    assert len(qr["citations"]) > 0


def test_query_empty_index_returns_409(tmp_path: Path):
    settings = make_chroma_settings(tmp_path)
    app = create_app(settings, embedder=FakeEmbed(), llm=FakeLLM())
    client = TestClient(app)
    resp = client.post("/query", json={"question": "anything"})
    assert resp.status_code == 409
    assert "empty" in resp.json()["detail"].lower()


def test_state_override_without_constructor(tmp_path: Path, fixtures_dir: Path):
    settings = make_chroma_settings(tmp_path)
    raw = tmp_path / "raw"
    raw.mkdir()
    shutil.copy(fixtures_dir / "sample.txt", raw / "sample.txt")

    app = create_app(settings)
    app.state.embedder = FakeEmbed()
    app.state.llm = FakeLLM()
    client = TestClient(app)

    assert client.post("/ingest", json={"path": str(raw)}).status_code == 200
    assert client.post("/query", json={"question": "hi"}).status_code == 200


class BrokenQueryEmbed(FakeEmbed):
    def embed_query(self, text: str) -> list[float]:
        raise RuntimeError("embedding service down")


def test_provider_failure_returns_500(tmp_path: Path, fixtures_dir: Path):
    settings = make_chroma_settings(tmp_path)
    raw = tmp_path / "raw"
    raw.mkdir()
    shutil.copy(fixtures_dir / "sample.txt", raw / "sample.txt")

    app = create_app(settings, embedder=FakeEmbed(), llm=FakeLLM())
    client = TestClient(app)
    assert client.post("/ingest", json={"path": str(raw)}).status_code == 200

    app.state.embedder = BrokenQueryEmbed()
    resp = client.post("/query", json={"question": "What?"})
    assert resp.status_code == 500
    assert "embedding" in resp.json()["detail"].lower()