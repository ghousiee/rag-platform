from __future__ import annotations

import logging
from pathlib import Path

from rag_platform.chunking import chunk_documents
from rag_platform.config import Settings
from rag_platform.loaders import load_path
from rag_platform.models import IngestResult
from rag_platform.providers import build_embedder
from rag_platform.providers.base import EmbeddingProvider
from rag_platform.retrieval import Retriever, build_retriever
from rag_platform.retrieval.bm25_sync import sync_bm25
from rag_platform.stores import build_store
from rag_platform.stores.base import VectorStore

logger = logging.getLogger(__name__)

_EMPTY_DIR_ERROR = "No loadable files found in the ingest path."


def _chunk_id(source: str, chunk_index: int) -> str:
    return f"{source}::{chunk_index}"


def ingest_path(
    path: Path,
    settings: Settings,
    *,
    embedder: EmbeddingProvider | None = None,
    store: VectorStore | None = None,
    retriever: Retriever | None = None,
) -> IngestResult:
    path = Path(path)
    if not path.exists():
        return IngestResult(files=0, chunks=0, errors=[f"Path does not exist: {path}"])
    if not path.is_dir():
        return IngestResult(files=0, chunks=0, errors=[f"Not a directory: {path}"])

    embedder = embedder or build_embedder(settings)
    store = store or build_store(settings)

    files = 0
    chunks_total = 0
    errors: list[str] = []
    corpus_ids: list[str] = []
    corpus_texts: list[str] = []
    corpus_metas: list[dict] = []

    for file_path in sorted(path.rglob("*")):
        if not file_path.is_file():
            continue
        try:
            docs = load_path(file_path)
        except ValueError as exc:
            if "No loader registered" in str(exc):
                logger.info("Skipping file with unknown extension: %s", file_path)
                continue
            logger.exception("Failed to ingest %s", file_path)
            errors.append(f"{file_path}: {exc}")
            continue
        try:
            chunks = chunk_documents(
                docs,
                settings.chunking.size,
                settings.chunking.overlap,
            )
            if not chunks:
                continue
            source = str(chunks[0].metadata.get("source", file_path.resolve()))
            ids = [
                _chunk_id(
                    source,
                    int(chunk.metadata.get("chunk_index", index)),
                )
                for index, chunk in enumerate(chunks)
            ]
            texts = [chunk.text for chunk in chunks]
            metas = [chunk.metadata for chunk in chunks]
            vectors = embedder.embed_documents(texts)
            store.delete_by_source(source)
            store.upsert(ids, texts, vectors, metas)
            files += 1
            chunks_total += len(chunks)
            corpus_ids.extend(ids)
            corpus_texts.extend(texts)
            corpus_metas.extend(metas)
        except Exception as exc:
            logger.exception("Failed to ingest %s", file_path)
            errors.append(f"{file_path}: {exc}")

    if files == 0 and not errors:
        errors.append(_EMPTY_DIR_ERROR)

    if corpus_texts:
        r = retriever or build_retriever(settings, store, embedder)
        sync_bm25(
            r,
            store,
            fallback_ids=corpus_ids,
            fallback_texts=corpus_texts,
            fallback_metas=corpus_metas,
        )

    return IngestResult(files=files, chunks=chunks_total, errors=errors)