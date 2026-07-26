from agentic_doc_studio.constants import (
    EXAMPLE_GOALS,
    PHOENIX_UI_URL,
    WORKFLOW_CARDS,
)


def test_example_goals_defined() -> None:
    assert len(EXAMPLE_GOALS) >= 3
    assert all(goal.strip() for goal in EXAMPLE_GOALS)


def test_phoenix_ui_url() -> None:
    assert PHOENIX_UI_URL == "http://localhost:6006"


def test_workflow_cards_include_live_answer_and_placeholders() -> None:
    by_id = {card["id"]: card for card in WORKFLOW_CARDS}
    assert set(by_id) == {"answer", "compare", "gap_report"}

    assert by_id["answer"]["status"] == "live"
    assert by_id["compare"]["status"] == "coming_soon"
    assert by_id["gap_report"]["status"] == "planned"

    for card in WORKFLOW_CARDS:
        assert card["title"]
        assert card["badge"]
        assert card["blurb"].strip()
