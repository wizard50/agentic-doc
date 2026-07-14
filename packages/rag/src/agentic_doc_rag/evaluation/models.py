from pydantic import BaseModel, Field, model_validator


class EvalQuery(BaseModel):
    """A golden retrieval query with deterministic ground-truth labels."""

    id: str
    query: str
    expected_sources: list[str] = Field(default_factory=list)
    expected_section_paths: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_ground_truth(self) -> "EvalQuery":
        if not self.expected_sources and not self.expected_section_paths:
            msg = "EvalQuery must specify expected_sources and/or expected_section_paths"
            raise ValueError(msg)
        return self


class QueryEvalResult(BaseModel):
    """Per-query retrieval evaluation outcome."""

    query_id: str
    query: str
    tags: list[str] = Field(default_factory=list)
    hit_at_k: bool
    source_hit_at_k: bool
    section_hit_at_k: bool
    first_match_rank: int | None = None
    recall_at_k: float
    matched_sources: list[str] = Field(default_factory=list)
    matched_sections: list[str] = Field(default_factory=list)


class TagMetrics(BaseModel):
    """Aggregate retrieval metrics for a dataset tag."""

    tag: str
    query_count: int
    hit_at_k: float
    mrr: float
    recall_at_k: float


class EvalReport(BaseModel):
    """Aggregate retrieval evaluation report."""

    dataset_name: str
    query_count: int
    top_k: int
    hit_at_k: float
    mrr: float
    recall_at_k: float
    source_match_at_k: float
    section_match_at_k: float
    by_tag: list[TagMetrics] = Field(default_factory=list)
    results: list[QueryEvalResult] = Field(default_factory=list)


class LlmRelevanceScore(BaseModel):
    """LLM relevance judgment for a single retrieved document."""

    query_id: str
    query: str
    document_position: int
    score: float
    label: str
    explanation: str | None = None
    span_id: str | None = None


class LlmEvalReport(BaseModel):
    """Aggregate LLM-based retrieval evaluation metrics."""

    model: str
    precision_at_k: float
    llm_hit_at_k: float
    scores: list[LlmRelevanceScore] = Field(default_factory=list)
    annotations_uploaded: bool = False
