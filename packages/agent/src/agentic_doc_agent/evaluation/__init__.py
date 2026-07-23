"""Generation-quality evaluation (faithfulness and related metrics)."""

from agentic_doc_agent.evaluation.faithfulness import (
    FaithfulnessVerdict,
    build_faithfulness_messages,
    score_faithfulness,
)

__all__ = [
    "FaithfulnessVerdict",
    "build_faithfulness_messages",
    "score_faithfulness",
]
