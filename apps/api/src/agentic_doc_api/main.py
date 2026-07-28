"""FastAPI application factory and ASGI entrypoint."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agentic_doc_api.api.router import api_router
from agentic_doc_api.config import get_api_settings
from agentic_doc_api.logging import configure_logging
from agentic_doc_api.middleware.request_id import RequestIdMiddleware

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Process lifespan: log start/stop. No external connections in Phase 0."""
    settings = get_api_settings()
    logger.info(
        "API starting name=%s version=%s environment=%s",
        settings.app_name,
        settings.app_version,
        settings.environment.value,
    )
    yield
    logger.info("API shutting down")


def create_app() -> FastAPI:
    """Build a configured FastAPI application (used by tests and uvicorn)."""
    settings = get_api_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Production API for Agentic Doc (M3). "
            "RAG and agent workflows are provided by library packages; "
            "this service is the HTTP boundary."
        ),
        docs_url="/docs" if settings.docs_enabled else None,
        redoc_url="/redoc" if settings.docs_enabled else None,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )
    app.add_middleware(RequestIdMiddleware)
    app.include_router(api_router)
    return app


app = create_app()
