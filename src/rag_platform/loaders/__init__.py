from __future__ import annotations

import logging
from pathlib import Path

from rag_platform.loaders.base import Loader
from rag_platform.loaders.markdown import MarkdownLoader
from rag_platform.loaders.pdf import PdfLoader
from rag_platform.loaders.text import TextLoader
from rag_platform.models import Document

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, Loader] = {}


def register_loader(ext: str, loader: Loader) -> None:
    normalized = ext if ext.startswith(".") else f".{ext}"
    normalized = normalized.lower()
    _REGISTRY[normalized] = loader


def _loader_for_path(path: Path) -> Loader | None:
    return _REGISTRY.get(path.suffix.lower())


def load_path(path: Path) -> list[Document]:
    path = Path(path)
    if not path.is_file():
        msg = f"Not a file: {path}"
        raise FileNotFoundError(msg)
    loader = _loader_for_path(path)
    if loader is None:
        msg = f"No loader registered for extension {path.suffix!r}: {path}"
        raise ValueError(msg)
    return loader.load(path)


def load_directory(path: Path) -> list[Document]:
    path = Path(path)
    if not path.is_dir():
        msg = f"Not a directory: {path}"
        raise NotADirectoryError(msg)

    documents: list[Document] = []
    for file_path in sorted(path.rglob("*")):
        if not file_path.is_file():
            continue
        loader = _loader_for_path(file_path)
        if loader is None:
            logger.info("Skipping file with unknown extension: %s", file_path)
            continue
        documents.extend(loader.load(file_path))
    return documents


register_loader(".txt", TextLoader())
register_loader(".md", MarkdownLoader())
register_loader(".markdown", MarkdownLoader())
register_loader(".pdf", PdfLoader())

__all__ = [
    "Loader",
    "load_directory",
    "load_path",
    "register_loader",
]