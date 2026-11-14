from __future__ import annotations

from rag_platform.config import Settings
from rag_platform.providers.base import EmbeddingProvider, LLMProvider
from rag_platform.providers.ollama import OllamaEmbeddingsProvider, OllamaLLMProvider
from rag_platform.providers.openai_compatible import (
    OpenAICompatibleEmbeddingsProvider,
    OpenAICompatibleLLMProvider,
)

_EMBEDDINGS_PROVIDERS = {
    "ollama": OllamaEmbeddingsProvider,
    "openai_compatible": OpenAICompatibleEmbeddingsProvider,
}

_LLM_PROVIDERS = {
    "ollama": OllamaLLMProvider,
    "openai_compatible": OpenAICompatibleLLMProvider,
}


def build_embedder(settings: Settings) -> EmbeddingProvider:
    name = settings.embeddings.provider
    factory = _EMBEDDINGS_PROVIDERS.get(name)
    if factory is None:
        supported = ", ".join(sorted(_EMBEDDINGS_PROVIDERS))
        raise ValueError(
            f"Unknown embeddings provider {name!r}. Supported providers: {supported}."
        )
    return factory(settings.embeddings)


def build_llm(settings: Settings) -> LLMProvider:
    name = settings.llm.provider
    factory = _LLM_PROVIDERS.get(name)
    if factory is None:
        supported = ", ".join(sorted(_LLM_PROVIDERS))
        raise ValueError(
            f"Unknown llm provider {name!r}. Supported providers: {supported}."
        )
    return factory(settings.llm)


__all__ = [
    "EmbeddingProvider",
    "LLMProvider",
    "build_embedder",
    "build_llm",
]