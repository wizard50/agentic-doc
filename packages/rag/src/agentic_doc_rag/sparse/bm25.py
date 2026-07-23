import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from agentic_doc_rag.models import DocumentChunk, SearchResult
from agentic_doc_rag.observability.tracing import (
    get_tracer,
    mark_chain_span,
    mark_retriever_span,
    record_search_results,
)

_CHUNKS_FILE = "chunks.json"
_MANIFEST_FILE = "manifest.json"
_INDEX_VERSION = 1
_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class Bm25Index:
    """Local BM25 index with JSON persistence alongside the vector store."""

    def __init__(self, persist_dir: Path) -> None:
        self._persist_dir = persist_dir
        self._chunks: list[DocumentChunk] = []
        self._bm25: BM25Okapi | None = None
        self._load_if_present()

    def build(self, chunks: list[DocumentChunk]) -> None:
        with get_tracer(__name__).start_as_current_span("sparse.build") as span:
            mark_chain_span(span)
            span.set_attribute("persist_dir", str(self._persist_dir))
            span.set_attribute("chunk_count", len(chunks))

            self._chunks = list(chunks)
            if not chunks:
                self._bm25 = None
                self._persist()
                return

            tokenized_corpus = [_tokenize(chunk.text) for chunk in chunks]
            self._bm25 = BM25Okapi(tokenized_corpus)
            self._persist()

    def search(self, query: str, k: int) -> list[SearchResult]:
        with get_tracer(__name__).start_as_current_span("sparse.search") as span:
            mark_retriever_span(span, query=query, top_k=k)
            span.set_attribute("persist_dir", str(self._persist_dir))

            if self._bm25 is None or not self._chunks or k <= 0:
                record_search_results(span, [])
                return []

            query_tokens = _tokenize(query)
            if not query_tokens:
                record_search_results(span, [])
                return []

            scores = self._bm25.get_scores(query_tokens)
            top_indices = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[
                :k
            ]
            search_results = [
                SearchResult(chunk=self._chunks[index], score=float(scores[index]))
                for index in top_indices
            ]
            record_search_results(span, search_results)
            return search_results

    def count(self) -> int:
        return len(self._chunks)

    def _load_if_present(self) -> None:
        chunks_path = self._persist_dir / _CHUNKS_FILE
        if not chunks_path.exists():
            return

        chunks_data = json.loads(chunks_path.read_text(encoding="utf-8"))
        self._chunks = [DocumentChunk.model_validate(item) for item in chunks_data]
        if not self._chunks:
            self._bm25 = None
            return

        tokenized_corpus = [_tokenize(chunk.text) for chunk in self._chunks]
        self._bm25 = BM25Okapi(tokenized_corpus)

    def _persist(self) -> None:
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        chunks_path = self._persist_dir / _CHUNKS_FILE
        manifest_path = self._persist_dir / _MANIFEST_FILE

        chunks_path.write_text(
            json.dumps([chunk.model_dump() for chunk in self._chunks], indent=2),
            encoding="utf-8",
        )
        manifest_path.write_text(
            json.dumps({"version": _INDEX_VERSION, "chunk_count": len(self._chunks)}, indent=2),
            encoding="utf-8",
        )
