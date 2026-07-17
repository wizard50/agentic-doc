import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from agentic_doc_rag.models import DocumentChunk
from agentic_doc_rag.observability.tracing import get_tracer, mark_chain_span

HEADER_RE = re.compile(r"^(#{1,6})\s+(.+)$")

DEFAULT_CHUNK_SIZE = 1500
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_HEADER_LEVELS = frozenset({2, 3})


@dataclass
class Section:
    headers: dict[str, str] = field(default_factory=dict)
    lines: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.lines).strip()

    @property
    def section_path(self) -> str:
        return " > ".join(self.headers[f"h{i}"] for i in range(1, 7) if f"h{i}" in self.headers)


def _apply_heading(stack: dict[str, str], level: int, title: str) -> None:
    stack[f"h{level}"] = title
    for deeper in range(level + 1, 7):
        stack.pop(f"h{deeper}", None)


def split_by_headers(text: str, levels: set[int] | frozenset[int]) -> list[Section]:
    """Split markdown into sections at the given header levels (e.g. {2, 3} for ## / ###)."""
    sections: list[Section] = []
    stack: dict[str, str] = {}
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_lines
        if not current_lines:
            return
        sections.append(Section(headers=dict(stack), lines=current_lines))
        current_lines = []

    in_fence = False
    for line in text.splitlines():
        if line.startswith("```"):
            in_fence = not in_fence

        match = HEADER_RE.match(line)
        if match and not in_fence:
            level = len(match.group(1))
            title = match.group(2).strip()
            if level in levels:
                flush()
            _apply_heading(stack, level, title)

        current_lines.append(line)

    flush()
    return sections


def split_with_overlap(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split long text into overlapping character windows."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    parts: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        parts.append(text[start:end])
        if end >= len(text):
            break
        start = end - chunk_overlap

    return parts


def make_chunk_id(source: str | Path, section_path: str, index: int) -> str:
    """Return a stable short id for a chunk within a source section."""
    key = f"{source}:{section_path}:{index}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def chunk_markdown_text(
    text: str,
    source: str | Path,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    header_levels: set[int] | None = None,
) -> list[DocumentChunk]:
    """Chunk a single markdown document."""
    levels = header_levels or DEFAULT_HEADER_LEVELS
    chunks: list[DocumentChunk] = []

    for section in split_by_headers(text, levels):
        if not section.text:
            continue

        for index, part in enumerate(split_with_overlap(section.text, chunk_size, chunk_overlap)):
            chunks.append(
                DocumentChunk(
                    id=make_chunk_id(source, section.section_path, index),
                    text=part,
                    metadata={
                        "source": str(source),
                        "section_path": section.section_path,
                        **section.headers,
                    },
                )
            )

    return chunks


def chunk_markdown_file(
    path: str | Path,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    header_levels: set[int] | None = None,
) -> list[DocumentChunk]:
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    return chunk_markdown_text(
        text,
        file_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        header_levels=header_levels,
    )


def chunk_markdown_dir(
    root: str | Path,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    header_levels: set[int] | None = None,
    skip_files: frozenset[str] | set[str] | None = None,
) -> list[DocumentChunk]:
    root_path = Path(root)
    excluded = skip_files or frozenset()

    with get_tracer(__name__).start_as_current_span("chunk.markdown_dir") as span:
        mark_chain_span(span)
        span.set_attribute("source_dir", str(root_path))

        chunks: list[DocumentChunk] = []
        file_count = 0
        for file_path in sorted(root_path.rglob("*.md")):
            if file_path.name in excluded:
                continue
            file_count += 1
            chunks.extend(
                chunk_markdown_file(
                    file_path,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    header_levels=header_levels,
                )
            )

        span.set_attribute("file_count", file_count)
        span.set_attribute("chunk_count", len(chunks))
        return chunks
