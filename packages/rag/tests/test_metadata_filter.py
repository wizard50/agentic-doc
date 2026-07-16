from support.builders import search_result
from support.fakes import StubVectorStore
from support.pipelines import semantic_pipeline_retriever

from agentic_doc_rag.retrieval import MetadataFilter, RetrievalRequest
from agentic_doc_rag.retrieval.filters import MetadataFilterStage, matches_filter


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
    results = [search_result("1", "ownership.md", section_path="Ownership")]

    assert stage.run(RetrievalRequest(query="ownership", top_k=3), results) == results


def test_metadata_filter_stage_trims_non_matching_results() -> None:
    stage = MetadataFilterStage()
    results = [
        search_result("1", "borrowing.md", section_path="Borrowing"),
        search_result("2", "ownership.md", section_path="Ownership"),
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
    store = StubVectorStore(
        fixed_results=[
            search_result("1", "borrowing.md", section_path="Borrowing"),
            search_result("2", "ownership.md", section_path="Ownership"),
            search_result("3", "traits.md", section_path="Traits"),
        ],
        count=3,
    )
    retriever = semantic_pipeline_retriever(store)

    results = retriever.retrieve(
        RetrievalRequest(
            query="rules",
            top_k=1,
            candidate_k=3,
            filters=MetadataFilter(source_contains=".md"),
        )
    )

    assert len(results) == 1
