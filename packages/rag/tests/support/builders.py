from agentic_doc_rag.models import DocumentChunk, SearchResult


def search_result(
    chunk_id: str,
    source: str,
    *,
    text: str = "example",
    score: float = 0.1,
    section_path: str | None = None,
) -> SearchResult:
    metadata: dict[str, str] = {"source": source}
    if section_path is not None:
        metadata["section_path"] = section_path
    return SearchResult(
        chunk=DocumentChunk(id=chunk_id, text=text, metadata=metadata),
        score=score,
    )