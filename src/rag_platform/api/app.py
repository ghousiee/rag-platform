from __future__ import annotations

from fastapi import FastAPI

from rag_platform.config import Settings, load_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    """Minimal FastAPI shell; Task 9 adds ingest/query/health routes."""
    if settings is None:
        settings = load_settings()
    application = FastAPI(title="rag-platform")
    application.state.settings = settings
    return application