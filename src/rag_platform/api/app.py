from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request

from rag_platform.api.schemas import HealthResponse, IngestRequest, QueryRequest
from rag_platform.config import Settings, load_settings
from rag_platform.generation import query as run_query
from rag_platform.ingest import ingest_path
from rag_platform.models import IngestResult, QueryResult
from rag_platform.providers.base import EmbeddingProvider, LLMProvider
from rag_platform.stores import build_store
from rag_platform.stores.base import VectorStore

logger = logging.getLogger(__name__)

_EMPTY_INDEX_MSG = "Index is empty. Run ingest first."


def _get_embedder(request: Request) -> EmbeddingProvider | None:
    return getattr(request.app.state, "embedder", None)


def _get_llm(request: Request) -> LLMProvider | None:
    return getattr(request.app.state, "llm", None)


def _get_store(request: Request, settings: Settings) -> VectorStore:
    store = getattr(request.app.state, "store", None)
    if store is None:
        store = build_store(settings)
        request.app.state.store = store
    return store


def create_app(
    settings: Settings | None = None,
    *,
    embedder: EmbeddingProvider | None = None,
    llm: LLMProvider | None = None,
    store: VectorStore | None = None,
) -> FastAPI:
    if settings is None:
        settings = load_settings()
    application = FastAPI(title="rag-platform")
    application.state.settings = settings
    application.state.embedder = embedder
    application.state.llm = llm
    application.state.store = store

    @application.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        try:
            st = _get_store(request, settings)
            count = st.count()
        except Exception:
            logger.exception("health index_count failed")
            count = 0
        return HealthResponse(status="ok", index_count=count)

    @application.post("/ingest", response_model=IngestResult)
    def ingest(body: IngestRequest, request: Request) -> IngestResult:
        path_str = (body.path or "").strip()
        if not path_str:
            raise HTTPException(status_code=400, detail="Ingest path must not be empty.")
        embedder = _get_embedder(request)
        store = _get_store(request, settings)
        try:
            return ingest_path(
                Path(path_str),
                settings,
                embedder=embedder,
                store=store,
            )
        except Exception as exc:
            logger.exception("ingest failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @application.post("/query", response_model=QueryResult)
    def query_endpoint(body: QueryRequest, request: Request) -> QueryResult:
        question = (body.question or "").strip()
        if not question:
            raise HTTPException(status_code=400, detail="Question must not be empty.")
        embedder = _get_embedder(request)
        llm = _get_llm(request)
        store = _get_store(request, settings)
        try:
            return run_query(
                question,
                settings,
                top_k=body.top_k,
                filters=body.filters,
                embedder=embedder,
                store=store,
                llm=llm,
            )
        except RuntimeError as exc:
            if _EMPTY_INDEX_MSG in str(exc) or "empty" in str(exc).lower():
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("query failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return application