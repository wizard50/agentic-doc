from pathlib import Path

from support.builders import search_result
from support.fakes import StubVectorStore
from support.paths import CORPUS_DIR
from support.pipelines import indexed_pipeline_retriever, semantic_pipeline_retriever
from support.vectorstore import chroma_vector_store

from agentic_doc_rag.chunk.chunker import chunk_markdown_dir
from agentic_doc_rag.retrieval import RetrievalRequest, create_retriever
from agentic_doc_rag.sparse.bm25 import Bm25Index


def test_pipeline_retriever_delegates_semantic_search() -> None:
    store = StubVectorStore({"ownership rules": [search_result("1", "ownership.md")]}, count=1)
    retriever = semantic_pipeline_retriever(store)

    results = retriever.retrieve(RetrievalRequest(query="ownership rules", top_k=3))

    assert len(results) == 1
    assert store.search_calls == [("ownership rules", 20)]


def test_pipeline_retriever_exposes_chunk_count() -> None:
    retriever = semantic_pipeline_retriever(StubVectorStore(count=7))

    assert retriever.count() == 7


def test_create_retriever_runs_fixture_corpus_search(tmp_path: Path) -> None:
    chunks = chunk_markdown_dir(CORPUS_DIR)
    store = chroma_vector_store(tmp_path / "chroma", "retrieval-fixture")
    store.upsert(chunks)

    sparse = Bm25Index(tmp_path / "bm25")
    sparse.build(chunks)
    retriever = indexed_pipeline_retriever(store, sparse)
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
