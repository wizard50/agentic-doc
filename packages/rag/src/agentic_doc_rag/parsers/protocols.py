from pathlib import Path
from typing import Protocol

from agentic_doc_rag.ingest.models import IngestSettings
from agentic_doc_rag.models import DocumentChunk


class DocumentParser(Protocol):
    """Parse a single source file into RAG chunks."""

    @property
    def extensions(self) -> frozenset[str]:
        """Lowercase file suffixes this parser handles (e.g. {'.md'})."""
        ...

    def can_parse(self, path: Path) -> bool: ...

    def parse(self, path: Path, settings: IngestSettings) -> list[DocumentChunk]: ...
