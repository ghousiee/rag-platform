from pathlib import Path

from pypdf import PdfWriter

from rag_platform.loaders import load_directory, load_path


def test_load_markdown(fixtures_dir: Path):
    docs = load_path(fixtures_dir / "sample.md")
    assert len(docs) == 1
    assert "RAG" in docs[0].text
    assert docs[0].metadata["source"].endswith("sample.md")


def test_load_directory_skips_unknown(fixtures_dir: Path):
    docs = load_directory(fixtures_dir)
    assert len(docs) >= 2


def test_load_pdf_generated(tmp_path: Path):
    pdf_path = tmp_path / "sample.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with pdf_path.open("wb") as handle:
        writer.write(handle)

    docs = load_path(pdf_path)
    assert len(docs) == 1
    assert docs[0].metadata["source"] == str(pdf_path.resolve())
    assert isinstance(docs[0].text, str)