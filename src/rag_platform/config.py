from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


class PathsConfig(BaseModel):
    raw: Path
    index: Path


class ChunkingConfig(BaseModel):
    size: int
    overlap: int


class EmbeddingsConfig(BaseModel):
    provider: str
    model: str
    base_url: str
    api_key: str | None = None


class LLMConfig(BaseModel):
    provider: str
    model: str
    temperature: float
    base_url: str
    api_key: str | None = None


class VectorStoreConfig(BaseModel):
    backend: str
    collection: str
    url: str | None = None
    api_key: str | None = None
    dsn: str | None = None


class RetrievalConfig(BaseModel):
    top_k: int
    hybrid: bool
    bm25_weight: float
    rerank: bool


class PromptsConfig(BaseModel):
    system: str


class EvalConfig(BaseModel):
    golden_path: Path


class Settings(BaseModel):
    paths: PathsConfig
    chunking: ChunkingConfig
    embeddings: EmbeddingsConfig
    llm: LLMConfig
    vector_store: VectorStoreConfig
    retrieval: RetrievalConfig
    prompts: PromptsConfig
    eval: EvalConfig


def _find_repo_root(start: Path) -> Path:
    current = start.resolve()
    for directory in (current, *current.parents):
        if (directory / "pyproject.toml").is_file():
            return directory
    return Path.cwd().resolve()


def _resolve_path(value: Path | str, base: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def _resolve_settings_paths(data: dict[str, Any], base: Path) -> dict[str, Any]:
    paths = data.get("paths", {})
    if paths:
        data["paths"] = {
            "raw": _resolve_path(paths["raw"], base),
            "index": _resolve_path(paths["index"], base),
        }
    eval_cfg = data.get("eval", {})
    if eval_cfg and "golden_path" in eval_cfg:
        data["eval"] = {
            **eval_cfg,
            "golden_path": _resolve_path(eval_cfg["golden_path"], base),
        }
    return data


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    ollama_base = os.environ.get("OLLAMA_BASE_URL")
    if ollama_base:
        for key in ("embeddings", "llm"):
            section = data.get(key)
            if section and section.get("provider") == "ollama":
                section["base_url"] = ollama_base

    openai_base = os.environ.get("OPENAI_BASE_URL")
    if openai_base:
        for key in ("embeddings", "llm"):
            section = data.get(key)
            if section and section.get("provider") == "openai_compatible":
                section["base_url"] = openai_base

    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        for key in ("embeddings", "llm"):
            section = data.get(key)
            if section and section.get("provider") == "openai_compatible":
                section["api_key"] = openai_key

    return data


def load_settings(config_path: Path | None = None) -> Settings:
    if config_path is None:
        repo_root = _find_repo_root(Path.cwd())
        config_path = repo_root / "configs" / "default.yaml"
    else:
        config_path = Path(config_path)

    config_path = config_path.resolve()
    base = _find_repo_root(config_path.parent)

    with config_path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    resolved = _resolve_settings_paths(raw, base)
    resolved = _apply_env_overrides(resolved)
    return Settings.model_validate(resolved)