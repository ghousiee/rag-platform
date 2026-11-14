from abc import ABC, abstractmethod
from pathlib import Path

from rag_platform.models import Document


class Loader(ABC):
    @abstractmethod
    def load(self, path: Path) -> list[Document]:
        """Load a file into one or more documents."""
        ...