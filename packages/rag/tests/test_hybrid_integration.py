from pathlib import Path

from agentic_doc_rag.chunk.chunker import chunk_markdown_dir
from agentic_doc_rag.retrieval import PipelineRetriever, RetrievalRequest, RetrieveStage, SearchMode
from agentic_doc_rag.sparse.bm25 import Bm25Index
from agentic_doc_rag.vectorstore.chroma import ChromaVectorStore

FIXTURES_DIR = Path(__file__).parent / "fixtures"
CORPUS_DIR = FIXTURES_DIR / "corpus"


def test_hybrid_search_surfaces_borrowing_section(tmp_path: Path) -> None:
    chunks = chunk_markdown_dir(CORPUS_DIR)
    store = ChromaVectorStore(tmp_path / "chroma", "hybrid-fixture")
    store.upsert(chunks)
    sparse = Bm25Index(tmp_path / "bm25")
    sparse.build(chunks)

    retriever = PipelineRetriever(stages=[RetrieveStage(store, sparse)], vectorstore=store)
    results = retriever.retrieve(
        RetrievalRequest(query="borrowing", mode=SearchMode.HYBRID, top_k=3, candidate_k=10)
    )

    assert results
    assert "Borrowing" in results[0].chunk.metadata.get("section_path", "")