from agentic_doc_core.config import CoreSettings, get_core_settings


def test_settings_loads() -> None:
    settings = get_core_settings()
    assert isinstance(settings, CoreSettings)
