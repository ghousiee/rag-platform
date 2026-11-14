from pathlib import Path

from rag_platform.loaders.base import Loader
from rag_platform.models import Document


class TextLoader(Loader):
    def load(self, path: Path) -> list[Document]:
        text = path.read_text(encoding="utf-8")
        return [
            Document(
                text=text,
                metadata={"source": str(path.resolve())},
            )
        ]