from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_platform.models import Document


def chunk_documents(
    docs: list[Document],
    size: int,
    overlap: int,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
    )
    chunks: list[Document] = []
    for doc in docs:
        texts = splitter.split_text(doc.text)
        for index, text in enumerate(texts):
            metadata = {**doc.metadata, "chunk_index": index}
            chunks.append(Document(text=text, metadata=metadata))
    return chunks