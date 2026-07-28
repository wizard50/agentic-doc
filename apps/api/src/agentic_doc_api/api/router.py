"""Aggregate API routers."""

from fastapi import APIRouter

from agentic_doc_api.api import health

api_router = APIRouter()
api_router.include_router(health.router)
