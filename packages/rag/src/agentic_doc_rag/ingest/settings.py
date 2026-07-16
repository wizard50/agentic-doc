from agentic_doc_rag.config import RagSettings
from agentic_doc_rag.ingest.models import IngestSettings


def parse_skip_files(value: str) -> frozenset[str]:
    if not value.strip():
        return frozenset()
    return frozenset(part.strip() for part in value.split(",") if part.strip())


def ingest_settings_from_rag(settings: RagSettings) -> IngestSettings:
    return IngestSettings(
        source_dir=settings.ingest_source_dir,
        skip_files=parse_skip_files(settings.ingest_skip_files),
    )