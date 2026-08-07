from agentic_doc_studio.constants import (
    EXAMPLE_GOALS,
    PHOENIX_UI_URL,
)


def test_example_goals_defined() -> None:
    assert len(EXAMPLE_GOALS) >= 3
    assert all(goal.strip() for goal in EXAMPLE_GOALS)


def test_phoenix_ui_url() -> None:
    assert PHOENIX_UI_URL == "http://localhost:6006"
