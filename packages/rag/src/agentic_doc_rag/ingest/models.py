from pathlib import Path

from pydantic import BaseModel, Field

from agentic_doc_rag.chunk.chunker import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_HEADER_LEVELS,
)


class IngestSettings(BaseModel):
    source_dir: Path
    skip_files: frozenset[str] = Field(default_factory=frozenset)
    chunk_size: int = Field(default=DEFAULT_CHUNK_SIZE, ge=1)
    chunk_overlap: int = Field(default=DEFAULT_CHUNK_OVERLAP, ge=0)
    header_levels: frozenset[int] = Field(default_factory=lambda: DEFAULT_HEADER_LEVELS)


class IngestResult(BaseModel):
    source_dir: Path
    file_count: int
    chunk_count: int