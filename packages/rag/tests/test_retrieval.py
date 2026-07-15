from pathlib import Path

from agentic_doc_rag.chunk.chunker import chunk_markdown_dir
from agentic_doc_rag.models import DocumentChunk, SearchResult
from agentic_doc_rag.retrieval import (
    MetadataFilterStage,
    PipelineRetriever,
    RetrievalRequest,
    RetrieveStage,
    TopKStage,
    create_retriever,
)
from agentic_doc_rag.sparse.bm25 import Bm25Index
from agentic_doc_rag.vectorstore.chroma import ChromaVectorStore

FIXTURES_DIR = Path(__file__).parent / "fixtures"
CORPUS_DIR = FIXTURES_DIR / "corpus"


class _StubVectorStore:
    def __init__(self, responses: dict[str, list[SearchResult]], *, count: int) -> None:
        self._responses = responses
        self._count = count
        self.search_calls: list[tuple[str, int]] = []

    def upsert(self, chunks: list[DocumentChunk]) -> None:
        raise NotImplementedError

    def search(self, query: str, k: int) -> list[SearchResult]:
        self.search_calls.append((query, k))
        return self._responses.get(query, [])

    def delete(self, ids: list[str]) -> None:
        raise NotImplementedError

    def count(self) -> int:
        return self._count


def _search_result(chunk_id: str, source: str) -> SearchResult:
    return SearchResult(
        chunk=DocumentChunk(id=chunk_id, text="example", metadata={"source": source}),
        score=0.1,
    )


class _UnusedSparseIndex:
    def build(self, chunks: list[DocumentChunk]) -> None:
        raise NotImplementedError

    def search(self, query: str, k: int) -> list[SearchResult]:
        raise AssertionError("sparse search should not be called")

    def count(self) -> int:
        return 0


def _pipeline_retriever(store: _StubVectorStore) -> PipelineRetriever:
    return PipelineRetriever(
        stages=[
            RetrieveStage(store, _UnusedSparseIndex()),
            MetadataFilterStage(),
            TopKStage(),
        ],
        vectorstore=store,
    )


def test_pipeline_retriever_delegates_semantic_search() -> None:
    store = _StubVectorStore({"ownership rules": [_search_result("1", "ownership.md")]}, count=1)
    retriever = _pipeline_retriever(store)

    results = retriever.retrieve(RetrievalRequest(query="ownership rules", top_k=3))

    assert len(results) == 1
    assert store.search_calls == [("ownership rules", 20)]


def test_pipeline_retriever_exposes_chunk_count() -> None:
    retriever = _pipeline_retriever(_StubVectorStore({}, count=7))

    assert retriever.count() == 7


def test_create_retriever_runs_fixture_corpus_search(tmp_path: Path) -> None:
    chunks = chunk_markdown_dir(CORPUS_DIR)
    store = ChromaVectorStore(tmp_path / "chroma", "retrieval-fixture")
    store.upsert(chunks)

    sparse = Bm25Index(tmp_path / "bm25")
    sparse.build(chunks)
    retriever = PipelineRetriever(
        stages=[RetrieveStage(store, sparse), MetadataFilterStage(), TopKStage()],
        vectorstore=store,
    )
    results = retriever.retrieve(RetrievalRequest(query="ownership", top_k=2))

    assert results
    assert retriever.count() == len(chunks)


def test_create_retriever_factory_builds_pipeline(tmp_path: Path) -> None:
    from agentic_doc_rag.config import RagSettings

    settings = RagSettings(
        chroma_persist_dir=tmp_path / "chroma",
        chroma_collection_name="factory",
        bm25_persist_dir=tmp_path / "bm25",
    )
    retriever = create_retriever(settings)

    assert retriever.count() == 0