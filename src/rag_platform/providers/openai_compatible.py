from __future__ import annotations

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from rag_platform.config import EmbeddingsConfig, LLMConfig
from rag_platform.providers.base import to_langchain_messages


class OpenAICompatibleEmbeddingsProvider:
    def __init__(self, config: EmbeddingsConfig) -> None:
        kwargs: dict = {
            "model": config.model,
            "base_url": config.base_url,
        }
        if config.api_key is not None:
            kwargs["api_key"] = config.api_key
        else:
            # Local OpenAI-compatible gateways often accept a placeholder key.
            kwargs["api_key"] = "not-needed"
        self._client = OpenAIEmbeddings(**kwargs)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._client.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._client.embed_query(text)


class OpenAICompatibleLLMProvider:
    def __init__(self, config: LLMConfig) -> None:
        self._default_temperature = config.temperature
        kwargs: dict = {
            "model": config.model,
            "base_url": config.base_url,
            "temperature": config.temperature,
        }
        if config.api_key is not None:
            kwargs["api_key"] = config.api_key
        else:
            kwargs["api_key"] = "not-needed"
        self._client = ChatOpenAI(**kwargs)

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