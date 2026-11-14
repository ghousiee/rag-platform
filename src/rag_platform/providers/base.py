from __future__ import annotations

from typing import Protocol, runtime_checkable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

_ROLE_TO_MESSAGE = {
    "system": SystemMessage,
    "human": HumanMessage,
    "user": HumanMessage,
    "assistant": AIMessage,
    "ai": AIMessage,
}


def to_langchain_messages(messages: list[dict[str, str]]):
    out = []
    for message in messages:
        role = message.get("role", "user").lower()
        content = message.get("content", "")
        cls = _ROLE_TO_MESSAGE.get(role, HumanMessage)
        out.append(cls(content=content))
    return out


@runtime_checkable
class EmbeddingProvider(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


@runtime_checkable
class LLMProvider(Protocol):
    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
    ) -> str:
        ...