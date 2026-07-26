PHOENIX_UI_URL = "http://localhost:6006"
PREVIEW_LENGTH = 400

EXAMPLE_GOALS = (
    "What is ownership in Rust?",
    "How do borrowing and references relate?",
    "What is the difference between String and &str?",
    "How do I handle recoverable errors with Result?",
    "How do traits work in Rust?",
)

# Product map for the gallery — only Answer is runnable in v1.
WORKFLOW_CARDS = (
    {
        "id": "answer",
        "title": "Answer",
        "status": "live",
        "badge": "Live",
        "blurb": (
            "Multi-step grounded Q&A: plan a search query, retrieve evidence, "
            "generate with citations, optional re-retrieve, faithfulness score."
        ),
    },
    {
        "id": "compare",
        "title": "Compare",
        "status": "coming_soon",
        "badge": "Coming soon",
        "blurb": (
            "Side-by-side analysis of two documentation topics or APIs — "
            "shared evidence, differences, and structured findings."
        ),
    },
    {
        "id": "gap_report",
        "title": "Gap report",
        "status": "planned",
        "badge": "Planned",
        "blurb": (
            "Find missing or thin coverage in the corpus relative to a goal "
            "or checklist — useful for docs quality reviews."
        ),
    },
)

ARCHITECTURE_SUMMARY = """
**What this demo is**

An agent answers questions about technical documentation (here: *The Rust
Programming Language* book). Answers are grounded in retrieved passages —
not free-form chat without sources.

**How a question is handled**

1. **Plan** — rewrite your goal into a better search query  
2. **Retrieve** — find relevant chunks in the documentation index  
3. **Generate** — write an answer using only that context, with citations  
4. **Re-retrieve** (if needed) — search again when context is thin  
5. **Evaluate** — score how faithful the answer is to the retrieved text  

**Stack (under the hood)**

```
This Streamlit app
  → agent workflow (LangGraph)
    → documentation search (vector + keyword hybrid)
    → local indexes (Chroma + BM25)
```

Optional **Phoenix** tracing shows each plan / retrieve / generate / evaluate
step when observability is enabled in the environment.
"""
