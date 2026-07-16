from agentic_doc_rag.config import RagSettings, get_rag_settings


def test_settings_loads() -> None:
    settings = get_rag_settings()
    assert isinstance(settings, RagSettings)


def test_rerank_defaults() -> None:
    settings = RagSettings()
    assert settings.rerank_enabled is False
    assert settings.rerank_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"
