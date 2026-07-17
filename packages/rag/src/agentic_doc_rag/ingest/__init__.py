from typing import Any

from agentic_doc_rag.ingest.errors import IngestEmptyCorpusError, IngestSourceNotFoundError
from agentic_doc_rag.ingest.models import IngestResult, IngestSettings
from agentic_doc_rag.ingest.settings import (
    ingest_settings_from_rag,
    parse_skip_files,
    resolve_ingest_settings,
)

__all__ = [
    "IngestEmptyCorpusError",
    "IngestResult",
    "IngestSettings",
    "IngestSourceNotFoundError",
    "ingest_settings_from_rag",
    "parse_skip_files",
    "resolve_ingest_settings",
    "run_ingestion",
]


def __getattr__(name: str) -> Any:
    if name == "run_ingestion":
        from agentic_doc_rag.ingest.runner import run_ingestion

        return run_ingestion
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
