from typing import Any

from pydantic import BaseModel, Field


class Document(BaseModel):
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)