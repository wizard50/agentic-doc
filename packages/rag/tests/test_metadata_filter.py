from agentic_doc_rag.models import DocumentChunk, SearchResult
from agentic_doc_rag.retrieval import (
    MetadataFilter,
    PipelineRetriever,
    RetrievalRequest,
    RetrieveStage,
)
from agentic_doc_rag.retrieval.filters import MetadataFilterStage, matches_filter
from agentic_doc_rag.retrieval.topk import TopKStage


def _result(source: str, section_path: str, chunk_id: str) -> SearchResult:
    return SearchResult(
        chunk=DocumentChunk(
            id=chunk_id,
            text="example",
            metadata={"source": source, "section_path": section_path},
        ),
        score=1.0,
    )


def test_matches_filter_source_contains() -> None:
    metadata = {"source": "data/corpus/borrowing.md", "section_path": "Borrowing"}
    filters = MetadataFilter(source_contains="borrowing.md")

    assert matches_filter(metadata, filters)


def test_matches_filter_source_suffix_requires_path_ending() -> None:
    metadata = {"source": "data/corpus/prefix-borrowing.md-suffix", "section_path": "Borrowing"}
    filters = MetadataFilter(source_suffix="borrowing.md")

    assert not matches_filter(metadata, filters)


def test_matches_filter_section_path_is_case_insensitive() -> None:
    metadata = {"source": "data/corpus/borrowing.md", "section_path": "References and Borrowing"}
    filters = MetadataFilter(section_path_contains="borrowing")

    assert matches_filter(metadata, filters)


def test_matches_filter_requires_all_specified_fields() -> None:
    metadata = {"source": "data/corpus/borrowing.md", "section_path": "Borrowing"}
    filters = MetadataFilter(source_contains="borrowing.md", section_path_contains="Ownership")

    assert not matches_filter(metadata, filters)


def test_metadata_filter_stage_passes_through_when_filters_are_none() -> None:
    stage = MetadataFilterStage()
    results = [_result("ownership.md", "Ownership", "1")]

    assert stage.run(RetrievalRequest(query="ownership", top_k=3), results) == results


def test_metadata_filter_stage_trims_non_matching_results() -> None:
    stage = MetadataFilterStage()
    results = [
        _result("borrowing.md", "Borrowing", "1"),
        _result("ownership.md", "Ownership", "2"),
    ]
    request = RetrievalRequest(
        query="rules",
        top_k=3,
        filters=MetadataFilter(source_suffix="borrowing.md"),
    )

    filtered = stage.run(request, results)

    assert len(filtered) == 1
    assert filtered[0].chunk.id == "1"


def test_pipeline_filter_and_topk_limit_final_results() -> None:
    class _StubVectorStore:
        def upsert(self, chunks: list[DocumentChunk]) -> None:
            raise NotImplementedError

        def search(self, query: str, k: int) -> list[SearchResult]:
            return [
                _result("borrowing.md", "Borrowing", "1"),
                _result("ownership.md", "Ownership", "2"),
                _result("traits.md", "Traits", "3"),
            ][:k]

        def delete(self, ids: list[str]) -> None:
            raise NotImplementedError

        def count(self) -> int:
            return 3

    class _UnusedSparseIndex:
        def build(self, chunks: list[DocumentChunk]) -> None:
            raise NotImplementedError

        def search(self, query: str, k: int) -> list[SearchResult]:
            raise AssertionError("sparse search should not be called")

        def count(self) -> int:
            return 0

    store = _StubVectorStore()
    retriever = PipelineRetriever(
        stages=[
            RetrieveStage(store, _UnusedSparseIndex()),
            MetadataFilterStage(),
            TopKStage(),
        ],
        vectorstore=store,
    )

    results = retriever.retrieve(
        RetrievalRequest(
            query="rules",
            top_k=1,
            candidate_k=3,
            filters=MetadataFilter(source_contains=".md"),
        )
    )

    assert len(results) == 1