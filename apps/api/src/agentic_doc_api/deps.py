"""FastAPI dependencies shared across routers."""

from agentic_doc_api.config import ApiSettings, get_api_settings


def get_settings() -> ApiSettings:
    """Inject process settings into route handlers."""
    return get_api_settings()
