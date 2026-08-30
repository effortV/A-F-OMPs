from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Enum, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_uuid() -> str:
    return str(uuid.uuid4())


class JobStatus(str, enum.Enum):
    queued = "queued"
    discovering = "discovering"
    awaiting_selection = "awaiting_selection"
    downloading = "downloading"
    parsing = "parsing"
    extracting = "extracting"
    indexing = "indexing"
    completed = "completed"
    failed = "failed"


class DocumentStatus(str, enum.Enum):
    candidate = "candidate"
    selected = "selected"
    downloading = "downloading"
    downloaded = "downloaded"
    parsed = "parsed"
    extracted = "extracted"
    indexed = "indexed"
    unavailable = "unavailable"
    failed = "failed"


class AutomationStatus(str, enum.Enum):
    active = "active"
    running = "running"
    stopping = "stopping"
    stopped = "stopped"
    failed = "failed"


class KnowledgeBase(Base):
    __tablename__ = "knowledge_bases"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    name: Mapped[str] = mapped_column(String(200), default="默认纳滤知识库")
    description: Mapped[str | None] = mapped_column(Text)
    index_version: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    conversations: Mapped[list[Conversation]] = relationship(back_populates="knowledge_base")
    documents: Mapped[list[Document]] = relationship(back_populates="knowledge_base")


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    knowledge_base_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id"), index=True)
    title: Mapped[str] = mapped_column(String(240), default="新对话")
    rolling_summary: Mapped[str | None] = mapped_column(Text)
    current_task: Mapped[str | None] = mapped_column(Text)
    index_version: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    knowledge_base: Mapped[KnowledgeBase] = relationship(back_populates="conversations")
    messages: Mapped[list[Message]] = relationship(back_populates="conversation", cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (Index("ix_messages_conversation_created", "conversation_id", "created_at"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id", ondelete="CASCADE"))
    role: Mapped[str] = mapped_column(String(20))
    content: Mapped[str] = mapped_column(Text)
    evidence: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    tool_calls: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    model_name: Mapped[str | None] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    conversation: Mapped[Conversation] = relationship(back_populates="messages")


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("knowledge_base_id", "doi_normalized", name="uq_document_kb_doi"),
        UniqueConstraint("knowledge_base_id", "openalex_id", name="uq_document_kb_openalex"),
        UniqueConstraint("knowledge_base_id", "title_author_fingerprint", name="uq_document_kb_fingerprint"),
        UniqueConstraint("knowledge_base_id", "file_sha256", name="uq_document_kb_sha256"),
        Index("ix_document_title", "title"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    knowledge_base_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id"), index=True)
    doi: Mapped[str | None] = mapped_column(String(300))
    doi_normalized: Mapped[str | None] = mapped_column(String(300))
    openalex_id: Mapped[str | None] = mapped_column(String(100))
    semantic_scholar_id: Mapped[str | None] = mapped_column(String(100))
    title: Mapped[str] = mapped_column(Text)
    title_author_fingerprint: Mapped[str] = mapped_column(String(64))
    authors: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    abstract: Mapped[str | None] = mapped_column(Text)
    publication_year: Mapped[int | None] = mapped_column(Integer)
    venue: Mapped[str | None] = mapped_column(String(500))
    landing_url: Mapped[str | None] = mapped_column(Text)
    fulltext_url: Mapped[str | None] = mapped_column(Text)
    fulltext_source: Mapped[str | None] = mapped_column(String(80))
    license: Mapped[str | None] = mapped_column(String(120))
    is_open_access: Mapped[bool] = mapped_column(Boolean, default=False)
    relevance_score: Mapped[float] = mapped_column(Float, default=0.0)
    relevance_reasons: Mapped[list[str]] = mapped_column(JSON, default=list)
    status: Mapped[DocumentStatus] = mapped_column(Enum(DocumentStatus), default=DocumentStatus.candidate)
    file_sha256: Mapped[str | None] = mapped_column(String(64))
    object_key: Mapped[str | None] = mapped_column(Text)
    parsed_object_key: Mapped[str | None] = mapped_column(Text)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    knowledge_base: Mapped[KnowledgeBase] = relationship(back_populates="documents")
    chunks: Mapped[list[Chunk]] = relationship(back_populates="document", cascade="all, delete-orphan")
    facts: Mapped[list[ExtractedFact]] = relationship(back_populates="document", cascade="all, delete-orphan")


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (Index("ix_chunk_document_order", "document_id", "chunk_index"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    chunk_index: Mapped[int] = mapped_column(Integer)
    section: Mapped[str | None] = mapped_column(String(500))
    block_kind: Mapped[str | None] = mapped_column(String(80))
    source_label: Mapped[str | None] = mapped_column(String(120))
    page_start: Mapped[int | None] = mapped_column(Integer)
    page_end: Mapped[int | None] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    token_count: Mapped[int | None] = mapped_column(Integer)
    vector_id: Mapped[str | None] = mapped_column(String(100), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    document: Mapped[Document] = relationship(back_populates="chunks")


class ExtractedFact(Base):
    __tablename__ = "extracted_facts"
    __table_args__ = (Index("ix_fact_document_type", "document_id", "fact_type"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    fact_type: Mapped[str] = mapped_column(String(80))
    subject: Mapped[str] = mapped_column(String(500))
    predicate: Mapped[str] = mapped_column(String(200))
    object_text: Mapped[str | None] = mapped_column(Text)
    value: Mapped[float | None] = mapped_column(Float)
    unit: Mapped[str | None] = mapped_column(String(80))
    normalized_value: Mapped[float | None] = mapped_column(Float)
    normalized_unit: Mapped[str | None] = mapped_column(String(80))
    conditions: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    source_sentence: Mapped[str] = mapped_column(Text)
    page: Mapped[int | None] = mapped_column(Integer)
    table_id: Mapped[str | None] = mapped_column(String(80))
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    review_status: Mapped[str] = mapped_column(String(30), default="pending")
    raw_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    document: Mapped[Document] = relationship(back_populates="facts")


class KnowledgeInsight(Base):
    """AI-derived cross-document synthesis kept separate from extracted facts."""

    __tablename__ = "knowledge_insights"
    __table_args__ = (
        Index("ix_insight_kb_created", "knowledge_base_id", "created_at"),
        Index("ix_insight_status_type", "status", "insight_type"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    knowledge_base_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id"), index=True)
    conversation_id: Mapped[str | None] = mapped_column(ForeignKey("conversations.id"), index=True)
    question: Mapped[str] = mapped_column(Text)
    insight_type: Mapped[str] = mapped_column(String(40))
    title: Mapped[str] = mapped_column(String(500))
    claim: Mapped[str] = mapped_column(Text)
    rationale: Mapped[str] = mapped_column(Text)
    evidence_document_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    evidence_refs: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    assumptions: Mapped[list[str]] = mapped_column(JSON, default=list)
    boundary_conditions: Mapped[list[str]] = mapped_column(JSON, default=list)
    validation_plan: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    novelty_score: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[str] = mapped_column(String(40), default="ai_hypothesis")
    model_name: Mapped[str | None] = mapped_column(String(200))
    review_note: Mapped[str | None] = mapped_column(Text)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class ImportJob(Base):
    __tablename__ = "import_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    conversation_id: Mapped[str | None] = mapped_column(ForeignKey("conversations.id"))
    knowledge_base_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id"), index=True)
    query: Mapped[str] = mapped_column(Text)
    expanded_terms: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    requested_count: Mapped[int] = mapped_column(Integer, default=0)
    selected_document_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    status: Mapped[JobStatus] = mapped_column(Enum(JobStatus), default=JobStatus.queued)
    stage: Mapped[str] = mapped_column(String(80), default="queued")
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    counts: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    log: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    task_control: Mapped[TaskControl | None] = relationship(
        back_populates="job",
        cascade="all, delete-orphan",
        passive_deletes=True,
        uselist=False,
    )

    @property
    def control_state(self) -> str:
        return self.task_control.state if self.task_control else "active"

    @property
    def deleted(self) -> bool:
        return bool(self.task_control and self.task_control.deleted)


class TaskControl(Base):
    """Durable cooperative control for a literature import job.

    A separate row avoids losing pause/cancel requests when a Worker concurrently
    writes progress into ImportJob.counts. Creating this new table is backward
    compatible with existing PostgreSQL and SQLite knowledge bases.
    """

    __tablename__ = "task_controls"

    job_id: Mapped[str] = mapped_column(
        ForeignKey("import_jobs.id", ondelete="CASCADE"),
        primary_key=True,
    )
    state: Mapped[str] = mapped_column(String(40), default="active", index=True)
    deleted: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    requested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    job: Mapped[ImportJob] = relationship(back_populates="task_control")


class DiscoveryCandidate(Base):
    __tablename__ = "discovery_candidates"
    __table_args__ = (
        UniqueConstraint("job_id", "candidate_id", name="uq_candidate_job_id"),
        Index("ix_candidate_job_rank", "job_id", "rank"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    job_id: Mapped[str] = mapped_column(ForeignKey("import_jobs.id", ondelete="CASCADE"), index=True)
    candidate_id: Mapped[str] = mapped_column(String(64))
    rank: Mapped[int] = mapped_column(Integer)
    source: Mapped[str] = mapped_column(String(80))
    doi: Mapped[str | None] = mapped_column(String(300))
    openalex_id: Mapped[str | None] = mapped_column(String(100))
    semantic_scholar_id: Mapped[str | None] = mapped_column(String(100))
    title: Mapped[str] = mapped_column(Text)
    title_author_fingerprint: Mapped[str] = mapped_column(String(64))
    authors: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    publication_year: Mapped[int | None] = mapped_column(Integer)
    venue: Mapped[str | None] = mapped_column(String(500))
    abstract: Mapped[str | None] = mapped_column(Text)
    landing_url: Mapped[str | None] = mapped_column(Text)
    fulltext_url: Mapped[str | None] = mapped_column(Text)
    license: Mapped[str | None] = mapped_column(String(120))
    is_open_access: Mapped[bool] = mapped_column(Boolean, default=False)
    relevance_score: Mapped[float] = mapped_column(Float, default=0.0)
    relevance_reasons: Mapped[list[str]] = mapped_column(JSON, default=list)
    already_exists: Mapped[bool] = mapped_column(Boolean, default=False)
    duplicate_reason: Mapped[str | None] = mapped_column(String(120))
    raw_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class TrainingTrace(Base):
    __tablename__ = "training_traces"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id"), index=True)
    message_id: Mapped[str | None] = mapped_column(ForeignKey("messages.id"))
    instruction: Mapped[str] = mapped_column(Text)
    input_text: Mapped[str] = mapped_column(Text)
    output_text: Mapped[str] = mapped_column(Text)
    retrieval_evidence: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    tool_trace: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    human_revision: Mapped[str | None] = mapped_column(Text)
    rating: Mapped[int | None] = mapped_column(Integer)
    approved_for_training: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class LiteratureAutomation(Base):
    """A persistent recurring literature-discovery task driven by the RQ scheduler."""

    __tablename__ = "literature_automations"
    __table_args__ = (Index("ix_automation_kb_status", "knowledge_base_id", "status"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    conversation_id: Mapped[str | None] = mapped_column(ForeignKey("conversations.id"), index=True)
    knowledge_base_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id"), index=True)
    name: Mapped[str] = mapped_column(String(240))
    query: Mapped[str] = mapped_column(Text)
    batch_size: Mapped[int] = mapped_column(Integer, default=50)
    interval_minutes: Mapped[int] = mapped_column(Integer, default=60)
    max_total: Mapped[int | None] = mapped_column(Integer)
    imported_total: Mapped[int] = mapped_column(Integer, default=0)
    cycles: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[AutomationStatus] = mapped_column(Enum(AutomationStatus), default=AutomationStatus.active)
    stop_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    next_run_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_run_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_job_id: Mapped[str | None] = mapped_column(ForeignKey("import_jobs.id"))
    rq_job_id: Mapped[str | None] = mapped_column(String(100))
    error_message: Mapped[str | None] = mapped_column(Text)
    settings_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
