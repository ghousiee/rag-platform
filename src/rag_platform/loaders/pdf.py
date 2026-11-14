from pathlib import Path

from pypdf import PdfReader

from rag_platform.loaders.base import Loader
from rag_platform.models import Document


class PdfLoader(Loader):
    def load(self, path: Path) -> list[Document]:
        reader = PdfReader(str(path))
        parts: list[str] = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                parts.append(page_text)
        text = "\n\n".join(parts)
        return [
            Document(
                text=text,
                metadata={"source": str(path.resolve())},
            )
        ]