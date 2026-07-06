from agentic_doc_rag.config import RagSettings, get_rag_settings


def test_settings_loads() -> None:
    settings = get_rag_settings()
    assert isinstance(settings, RagSettings)
