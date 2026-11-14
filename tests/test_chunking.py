from rag_platform.chunking import chunk_documents
from rag_platform.models import Document


def test_chunk_preserves_metadata_and_overlap():
    doc = Document(text="a" * 1000, metadata={"source": "x.txt"})
    chunks = chunk_documents([doc], size=200, overlap=50)
    assert len(chunks) > 1
    assert all(c.metadata["source"] == "x.txt" for c in chunks)
    assert chunks[0].metadata["chunk_index"] == 0