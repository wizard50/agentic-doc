from pathlib import Path

from agentic_doc_rag.config import EmbeddingType, RagSettings, get_rag_settings


def test_settings_loads() -> None:
    settings = get_rag_settings()
    assert isinstance(settings, RagSettings)


def test_rerank_defaults() -> None:
    settings = RagSettings()
    assert settings.rerank_enabled is False
    assert settings.rerank_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"


def test_embedding_defaults() -> None:
    settings = RagSettings()
    assert settings.embedding_type == EmbeddingType.CHROMA_DEFAULT
    assert settings.embedding_model == "all-MiniLM-L6-v2"


def test_ingest_defaults() -> None:
    settings = RagSettings()
    assert settings.ingest_source_dir == Path("corpora/rust-book/src")
    assert settings.ingest_skip_files == "SUMMARY.md,title-page.md"
    assert settings.ingest_on_startup is False
