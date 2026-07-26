PHOENIX_UI_URL = "http://localhost:6006"
PREVIEW_LENGTH = 400

EXAMPLE_GOALS = (
    "What is ownership in Rust?",
    "How do borrowing and references relate?",
    "What is the difference between String and &str?",
    "How do I handle recoverable errors with Result?",
    "How do traits work in Rust?",
)

ARCHITECTURE_SUMMARY = """
**Doc Agent** sits on the M2 agent package and M1 RAG core:

```
UI (this app)
  → agentic_doc_agent.run_workflow
    → LangGraph: plan → retrieve → generate ⇄ re-retrieve → evaluate
    → RetrieveTool → agentic_doc_rag (Chroma + BM25)
```

Phoenix / OpenInference spans (`agent.plan`, `agent.tool.retrieve`,
`agent.generate`, `agent.evaluate`) are emitted when `PHOENIX_ENABLED=true`.
"""
