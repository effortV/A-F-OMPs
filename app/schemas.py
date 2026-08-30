from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class KnowledgeBaseCreate(BaseModel):
    name: str = "默认纳滤知识库"
    description: str | None = None


class KnowledgeBaseRead(ORMModel):
    id: str
    name: str
    description: str | None
    index_version: int
    created_at: datetime


class ConversationCreate(BaseModel):
    knowledge_base_id: str | None = None
    title: str = "新对话"


class ConversationRead(ORMModel):
    id: str
    knowledge_base_id: str
    title: str
    rolling_summary: str | None
    current_task: str | None
    index_version: int
    created_at: datetime
    updated_at: datetime


class ConversationDeleteResponse(BaseModel):
    id: str
    deleted_messages: int
    deleted_training_traces: int
    preserved_insights: int
    knowledge_base_preserved: bool = True


class MessageRead(ORMModel):
    id: str
    conversation_id: str
    role: str
    content: str
    evidence: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    model_name: str | None = None
    created_at: datetime


class ChatRequest(BaseModel):
    conversation_id: str
    question: str = Field(min_length=1, max_length=20_000)
    proactive_literature: bool = True
    desired_new_count: int = Field(default=50, ge=0, le=200)
    knowledge_discovery: bool = True
    research_mode: Literal["deep_research", "evidence_strict", "rapid"] = "deep_research"
    deep_thinking: bool = True


class ChatResponse(BaseModel):
    message: MessageRead
    route: str
    evidence_count: int
    standalone_question: str
    needs_literature: bool = False
    literature_query: str | None = None
    literature_reason: str | None = None
    discovery: dict[str, Any] | None = None
    literature_error: str | None = None
    insight_count: int = 0


class DiscoveryRequest(BaseModel):
    conversation_id: str
    query: str = Field(min_length=1, max_length=1000)
    limit: int = Field(default=100, ge=1, le=500)
    year_from: int | None = Field(default=None, ge=1900, le=2100)
    year_to: int | None = Field(default=None, ge=1900, le=2100)
    include_citation_expansion: bool = True


class CandidateRead(BaseModel):
    candidate_id: str
    doi: str | None = None
    openalex_id: str | None = None
    title: str
    authors: list[dict[str, Any]] = Field(default_factory=list)
    publication_year: int | None = None
    venue: str | None = None
    abstract: str | None = None
    landing_url: str | None = None
    fulltext_url: str | None = None
    is_open_access: bool = False
    relevance_score: float = 0.0
    relevance_reasons: list[str] = Field(default_factory=list)
    already_exists: bool = False
    duplicate_reason: str | None = None
    source: str


class DiscoveryResponse(BaseModel):
    job_id: str
    expanded_terms: dict[str, Any]
    connector_status: dict[str, dict[str, Any]] = Field(default_factory=dict)
    total_found: int
    existing_count: int
    new_count: int
    candidates: list[CandidateRead]


class ImportSelectionRequest(BaseModel):
    candidate_ids: list[str] = Field(default_factory=list, max_length=200)
    requested_count: int = Field(default=0, ge=0, le=200)

    @field_validator("candidate_ids", mode="before")
    @classmethod
    def deduplicate_candidate_ids(cls, value: Any) -> Any:
        if not isinstance(value, (list, tuple)):
            return value
        unique: list[Any] = []
        for candidate_id in value:
            if candidate_id not in unique:
                unique.append(candidate_id)
        return unique

    @model_validator(mode="after")
    def selection_matches_count(self) -> ImportSelectionRequest:
        # The server derives the billable/import count from unique IDs instead
        # of rejecting a whole batch because a UI rerun repeated an option.
        self.requested_count = len(self.candidate_ids)
        return self


class JobRead(ORMModel):
    id: str
    conversation_id: str | None
    knowledge_base_id: str
    query: str
    requested_count: int
    status: str
    stage: str
    progress: float
    counts: dict[str, Any]
    log: list[dict[str, Any]]
    error_message: str | None
    control_state: str = "active"
    deleted: bool = False
    created_at: datetime
    updated_at: datetime


class JobControlRead(BaseModel):
    job_id: str
    state: str
    deleted: bool
    message: str


class UploadResponse(BaseModel):
    job_id: str
    document_id: str
    sha256: str
    message: str


class UploadMetadata(BaseModel):
    title: str
    doi: str | None = None
    authors: list[str] = Field(default_factory=list)
    publication_year: int | None = None
    rights_confirmed: Literal[True]


class PublicUrlImportRequest(BaseModel):
    source: str = Field(min_length=3, max_length=4000, description="公开文章页、PDF 地址或 DOI")
    title: str | None = Field(default=None, max_length=2000)
    doi: str | None = Field(default=None, max_length=300)
    authors: list[str] = Field(default_factory=list, max_length=200)
    publication_year: int | None = Field(default=None, ge=1800, le=2100)
    public_access_confirmed: Literal[True]


class RetryMetadataRequest(BaseModel):
    max_documents: int = Field(default=50, ge=1, le=200)


class AutomationCreate(BaseModel):
    conversation_id: str
    query: str = Field(min_length=1, max_length=1000)
    name: str | None = Field(default=None, max_length=240)
    batch_size: int = Field(default=50, ge=50, le=200)
    interval_minutes: int = Field(default=60, ge=5, le=10_080)
    max_total: int | None = Field(default=None, ge=50, le=100_000)


class AutomationRead(ORMModel):
    id: str
    conversation_id: str | None
    knowledge_base_id: str
    name: str
    query: str
    batch_size: int
    interval_minutes: int
    max_total: int | None
    imported_total: int
    cycles: int
    status: str
    stop_requested: bool
    next_run_at: datetime | None
    last_run_at: datetime | None
    last_job_id: str | None
    error_message: str | None
    created_at: datetime
    updated_at: datetime


class DocumentRead(ORMModel):
    id: str
    knowledge_base_id: str
    doi: str | None
    openalex_id: str | None
    title: str
    authors: list[dict[str, Any]]
    abstract: str | None
    publication_year: int | None
    venue: str | None
    landing_url: str | None
    fulltext_url: str | None
    fulltext_source: str | None
    license: str | None
    object_key: str | None
    is_open_access: bool
    relevance_score: float
    status: str
    metadata_json: dict[str, Any]
    error_message: str | None
    created_at: datetime
    updated_at: datetime


class ChunkRead(ORMModel):
    id: str
    chunk_index: int
    section: str | None
    block_kind: str | None
    source_label: str | None
    page_start: int | None
    page_end: int | None
    text: str


class FactRead(ORMModel):
    id: str
    fact_type: str
    subject: str
    predicate: str
    object_text: str | None
    value: float | None
    unit: str | None
    normalized_value: float | None
    normalized_unit: str | None
    conditions: dict[str, Any]
    source_sentence: str
    page: int | None
    table_id: str | None
    confidence: float


class KnowledgeInsightRead(ORMModel):
    id: str
    knowledge_base_id: str
    conversation_id: str | None
    question: str
    insight_type: str
    title: str
    claim: str
    rationale: str
    evidence_document_ids: list[str]
    evidence_refs: list[dict[str, Any]]
    assumptions: list[str]
    boundary_conditions: list[str]
    validation_plan: dict[str, Any]
    confidence: float
    novelty_score: float
    status: str
    model_name: str | None
    review_note: str | None
    reviewed_at: datetime | None
    created_at: datetime
    updated_at: datetime


class KnowledgeInsightReview(BaseModel):
    status: Literal["reviewed", "validated", "rejected"]
    review_note: str | None = Field(default=None, max_length=5000)


class DocumentDetail(BaseModel):
    document: DocumentRead
    chunks: list[ChunkRead]
    facts: list[FactRead]


class FeedbackRequest(BaseModel):
    message_id: str
    rating: int = Field(ge=1, le=5)
    human_revision: str | None = None
    approved_for_training: bool = False


class TrainingTraceRead(BaseModel):
    id: str
    conversation_id: str
    message_id: str | None
    instruction: str
    input_text: str
    output_text: str
    retrieval_evidence: list[dict[str, Any]]
    tool_trace: list[dict[str, Any]]
    human_revision: str | None
    rating: int | None
    approved_for_training: bool
    quality_score: float
    quality_label: str
    created_at: datetime


class TrainingTraceReview(BaseModel):
    rating: int = Field(ge=1, le=5)
    human_revision: str | None = Field(default=None, max_length=50_000)
    approved_for_training: bool = False


class TrainingStatsRead(BaseModel):
    total: int
    rated: int
    approved: int
    evidence_backed: int
    high_quality: int


class HealthRead(BaseModel):
    status: str
    database: str
    llm_configured: bool
    llm_model: str
    chat_llm_model: str
    openalex_configured: bool
    unpaywall_configured: bool
    elsevier_configured: bool
    neo4j_configured: bool
    chroma_path: str
    embedding_model: str
    embedding_download_allowed: bool
    embedding_fallback_allowed: bool
    hf_endpoint: str
    queue_mode: str
    missing_recommended_settings: list[str] = Field(default_factory=list)
