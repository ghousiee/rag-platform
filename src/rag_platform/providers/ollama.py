from __future__ import annotations

from langchain_ollama import ChatOllama, OllamaEmbeddings

from rag_platform.config import EmbeddingsConfig, LLMConfig
from rag_platform.providers.base import to_langchain_messages


class OllamaEmbeddingsProvider:
    def __init__(self, config: EmbeddingsConfig) -> None:
        self._client = OllamaEmbeddings(
            model=config.model,
            base_url=config.base_url,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._client.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._client.embed_query(text)


class OllamaLLMProvider:
    def __init__(self, config: LLMConfig) -> None:
        self._default_temperature = config.temperature
        self._client = ChatOllama(
            model=config.model,
            base_url=config.base_url,
            temperature=config.temperature,
        )

    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
    ) -> str:
        lc_messages = to_langchain_messages(messages)
        temp = self._default_temperature if temperature is None else temperature
        response = self._client.invoke(lc_messages, temperature=temp)
        content = response.content
        if isinstance(content, str):
            return content
        return str(content)