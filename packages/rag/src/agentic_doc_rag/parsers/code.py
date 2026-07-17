from pathlib import Path

from agentic_doc_rag.chunk.chunker import _chunk_id, split_with_overlap
from agentic_doc_rag.ingest.models import IngestSettings
from agentic_doc_rag.models import DocumentChunk
from agentic_doc_rag.observability.tracing import get_tracer, mark_chain_span
from agentic_doc_rag.parsers.code_units import CodeUnit, split_code_into_units
from agentic_doc_rag.parsers.language_profiles import PROFILES, profile_for_path


def _code_extensions() -> frozenset[str]:
    extensions: set[str] = set()
    for profile in PROFILES:
        extensions.update(profile.extensions)
    return frozenset(extensions)


class CodeParser:
    """Parse source files using structure-aware, profile-based unit splitting."""

    extensions: frozenset[str] = _code_extensions()

    def can_parse(self, path: Path) -> bool:
        return path.suffix.casefold() in self.extensions

    def parse(self, path: Path, settings: IngestSettings) -> list[DocumentChunk]:
        with get_tracer(__name__).start_as_current_span("parser.code") as span:
            mark_chain_span(span)
            span.set_attribute("source", str(path))

            profile = profile_for_path(path)
            span.set_attribute("language", profile.language)

            text = path.read_text(encoding="utf-8")
            units = split_code_into_units(text, profile)
            chunks = _chunks_from_units(
                path,
                profile.language,
                units,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )

            span.set_attribute("unit_count", len(units))
            span.set_attribute("chunk_count", len(chunks))
            return chunks


def _chunks_from_units(
    path: Path,
    language: str,
    units: list[CodeUnit],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for unit in units:
        section_path = _section_path(path, unit)
        parts = split_with_overlap(unit.text, chunk_size, chunk_overlap)
        for index, part in enumerate(parts):
            metadata: dict[str, object] = {
                "source": str(path),
                "file_type": "code",
                "language": language,
                "section_path": section_path,
                "start_line": unit.start_line,
                "end_line": unit.end_line,
            }
            if unit.symbol is not None:
                metadata["symbol"] = unit.symbol
            chunks.append(
                DocumentChunk(
                    id=_chunk_id(path, f"{section_path}:{index}", index),
                    text=part,
                    metadata=metadata,
                )
            )
    return chunks


def _section_path(path: Path, unit: CodeUnit) -> str:
    if unit.symbol:
        return f"{path.name}::{unit.symbol}"
    return f"{path.name}:L{unit.start_line}-L{unit.end_line}"
