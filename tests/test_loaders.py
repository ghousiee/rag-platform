from pathlib import Path

from rag_platform.loaders import load_directory, load_path


def test_load_markdown(fixtures_dir: Path):
    docs = load_path(fixtures_dir / "sample.md")
    assert len(docs) == 1
    assert "RAG" in docs[0].text
    assert docs[0].metadata["source"].endswith("sample.md")


def test_load_directory_skips_unknown(fixtures_dir: Path):
    docs = load_directory(fixtures_dir)
    assert len(docs) >= 2