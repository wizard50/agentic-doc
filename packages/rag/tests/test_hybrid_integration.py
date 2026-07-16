from pathlib import Path

from support.paths import CORPUS_DIR
from support.pipelines import indexed_pipeline_retriever

from agentic_doc_rag.chunk.chunker import chunk_markdown_dir
from agentic_doc_rag.retrieval import MetadataFilter, RetrievalRequest, SearchMode
from agentic_doc_rag.sparse.bm25 import Bm25Index
from agentic_doc_rag.vectorstore.chroma import ChromaVectorStore


def test_hybrid_search_surfaces_borrowing_section(tmp_path: Path) -> None:
    chunks = chunk_markdown_dir(CORPUS_DIR)
    store = ChromaVectorStore(tmp_path / "chroma", "hybrid-fixture")
    store.upsert(chunks)
    sparse = Bm25Index(tmp_path / "bm25")
    sparse.build(chunks)

    retriever = indexed_pipeline_retriever(store, sparse)
    results = retriever.retrieve(
        RetrievalRequest(query="borrowing", mode=SearchMode.HYBRID, top_k=3, candidate_k=10)
    )

    assert results
    assert "Borrowing" in results[0].chunk.metadata.get("section_path", "")


def test_hybrid_search_with_source_filter_limits_results(tmp_path: Path) -> None:
    chunks = chunk_markdown_dir(CORPUS_DIR)
    store = ChromaVectorStore(tmp_path / "chroma", "hybrid-filter-fixture")
    store.upsert(chunks)
    sparse = Bm25Index(tmp_path / "bm25")
    sparse.build(chunks)

    retriever = indexed_pipeline_retriever(store, sparse)
    results = retriever.retrieve(
        RetrievalRequest(
            query="ownership",
            mode=SearchMode.HYBRID,
            top_k=5,
            candidate_k=10,
            filters=MetadataFilter(source_contains="borrowing.md"),
        )
    )

    assert results
    assert all("borrowing.md" in result.chunk.metadata.get("source", "") for result in results)
