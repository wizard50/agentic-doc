import pytest
from support.fakes import StubEmbeddings

from agentic_doc_rag.config import EmbeddingType, RagSettings
from agentic_doc_rag.embeddings import (
    ChromaDefaultEmbeddings,
    EmbeddingModelMismatchError,
    SentenceTransformerEmbeddings,
    create_embeddings,
)
from agentic_doc_rag.models import DocumentChunk
from agentic_doc_rag.vectorstore.chroma import ChromaVectorStore


def test_create_embeddings_defaults_to_chroma_backend() -> None:
    embeddings = create_embeddings(RagSettings())

    assert isinstance(embeddings, ChromaDefaultEmbeddings)
    assert embeddings.model_name == "all-MiniLM-L6-v2"
    assert embeddings.dimensions == 384


def test_create_embeddings_sentence_transformers_backend() -> None:
    settings = RagSettings(
        embedding_type=EmbeddingType.SENTENCE_TRANSFORMERS,
        embedding_model="all-MiniLM-L6-v2",
    )

    embeddings = create_embeddings(settings)

    assert isinstance(embeddings, SentenceTransformerEmbeddings)
    assert embeddings.model_name == "all-MiniLM-L6-v2"


def test_chroma_default_embeddings_embed_documents_and_query() -> None:
    embeddings = ChromaDefaultEmbeddings()

    docs = embeddings.embed_documents(["ownership", "borrowing"])
    query = embeddings.embed_query("ownership")

    assert len(docs) == 2
    assert len(docs[0]) == embeddings.dimensions
    assert len(query) == embeddings.dimensions


def test_stub_embeddings_returns_deterministic_vectors() -> None:
    embeddings = StubEmbeddings()

    assert embeddings.embed_documents(["ab", "abcd"]) == [[2.0, 1.0, 0.0], [4.0, 1.0, 0.0]]
    assert embeddings.embed_query("hi") == [2.0, 1.0, 0.0]


def test_chroma_vector_store_uses_injected_embeddings(tmp_path) -> None:
    store = ChromaVectorStore(tmp_path / "chroma", "stub-embeddings", StubEmbeddings())
    store.upsert(
        [
            DocumentChunk(id="1", text="own", metadata={"source": "a.md"}),
            DocumentChunk(id="2", text="borrow", metadata={"source": "b.md"}),
        ]
    )

    results = store.search("own", k=1)

    assert len(results) == 1
    assert results[0].chunk.id == "1"


def test_chroma_vector_store_raises_on_embedding_model_mismatch(tmp_path) -> None:
    persist_dir = tmp_path / "chroma"
    first = ChromaVectorStore(persist_dir, "mismatch", StubEmbeddings())
    first.upsert([DocumentChunk(id="1", text="example", metadata={"source": "a.md"})])

    class OtherEmbeddings:
        model_name = "other"
        dimensions = 3

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return StubEmbeddings().embed_documents(texts)

        def embed_query(self, query: str) -> list[float]:
            return StubEmbeddings().embed_query(query)

    with pytest.raises(EmbeddingModelMismatchError, match="other"):
        ChromaVectorStore(persist_dir, "mismatch", OtherEmbeddings())