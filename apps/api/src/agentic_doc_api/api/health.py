"""Liveness and readiness probes."""

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Liveness probe payload."""

    status: str = Field(description="Always 'ok' when the process is up")


class ReadyResponse(BaseModel):
    """Readiness probe payload.

    Phase 0 reports ready once the app is loaded. Later phases may check DB
    connectivity and index presence without changing the path.
    """

    status: str = Field(description="Always 'ready' in the Phase 0 skeleton")


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness: process is running."""
    return HealthResponse(status="ok")


@router.get("/ready", response_model=ReadyResponse)
def ready() -> ReadyResponse:
    """Readiness: process can accept traffic (skeleton: always ready)."""
    return ReadyResponse(status="ready")
