from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Annotated
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Response, UploadFile
from sqlalchemy import delete, func, or_, select, update
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import get_db
from app.models import (
    AutomationStatus,
    Chunk,
    Conversation,
    DiscoveryCandidate,
    Document,
    DocumentStatus,
    ExtractedFact,
    ImportJob,
    JobStatus,
    KnowledgeInsight,
    KnowledgeBase,
    LiteratureAutomation,
    Message,
    TrainingTrace,
)
from app.schemas import (
    AutomationCreate,
    AutomationRead,
    CandidateRead,
    ChatRequest,
    ChatResponse,
    ConversationCreate,
    ConversationDeleteResponse,
    ConversationRead,
    DiscoveryRequest,
    DiscoveryResponse,
    DocumentDetail,
    DocumentRead,
    FeedbackRequest,
    HealthRead,
    ImportSelectionRequest,
    JobControlRead,
    JobRead,
    KnowledgeInsightRead,
    KnowledgeInsightReview,
    KnowledgeBaseCreate,
    KnowledgeBaseRead,
    MessageRead,
    PublicUrlImportRequest,
    RetryMetadataRequest,
    TrainingStatsRead,
    TrainingTraceRead,
    TrainingTraceReview,
    UploadResponse,
)
from app.services.dedupe import extract_doi, find_duplicate, normalize_doi, sha256_bytes, title_author_fingerprint
from app.services.discovery_service import DiscoveryService, candidates_for_job
from app.services.automation import (
    enqueue_automation_cycle,
    enqueue_automation_resume_cycle,
)
from app.services.import_service import ImportSelectionError, create_documents_from_candidates
from app.services.fulltext import FullTextResolver
from app.services.pipeline import apply_control_checkpoint, append_job_log, run_import_job
from app.services.queue import enqueue_import
from app.services.rag import NanofiltrationRAGAgent
from app.services.graph_store import GraphStore
from app.services.storage import ObjectStorage
from app.services.training import trace_to_read, traces_to_jsonl, training_quality
from app.services.task_control import (
    CANCEL_STATES,
    CANCEL_REQUESTED,
    PAUSE_REQUESTED,
    PAUSED,
    automation_is_deleted,
    clear_job_control,
    get_job_control,
    import_rq_call_is_running,
    mark_automation_deleted,
    remove_queued_automation_calls,
    remove_queued_import_calls,
    request_job_control,
)


router = APIRouter(prefix="/api")
Db = Annotated[Session, Depends(get_db)]


def get_or_404(db: Session, model: type, identifier: str):
    value = db.get(model, identifier)
    if value is None:
        raise HTTPException(status_code=404, detail=f"{model.__name__} 不存在")
    return value


def get_job_or_404(db: Session, job_id: str) -> ImportJob:
    job = get_or_404(db, ImportJob, job_id)
    if job.deleted:
        raise HTTPException(status_code=404, detail="ImportJob 不存在或已删除")
    return job


def is_priority_job(job: ImportJob) -> bool:
    if bool((job.counts or {}).get("priority")):
        return True
    return (job.query or "").strip().startswith(("用户上传：", "公开网址/DOI 导入：", "重新获取公开全文："))


def candidate_to_schema(item: DiscoveryCandidate) -> CandidateRead:
    return CandidateRead(
        candidate_id=item.candidate_id,
        doi=item.doi,
        openalex_id=item.openalex_id,
        title=item.title,
        authors=item.authors,
        publication_year=item.publication_year,
        venue=item.venue,
        abstract=item.abstract,
        landing_url=item.landing_url,
        fulltext_url=item.fulltext_url,
        is_open_access=item.is_open_access,
        relevance_score=item.relevance_score,
        relevance_reasons=item.relevance_reasons,
        already_exists=item.already_exists,
        duplicate_reason=item.duplicate_reason,
        source=item.source,
    )


def discovery_to_schema(job: ImportJob, rows: list[DiscoveryCandidate]) -> DiscoveryResponse:
    new_rows = [row for row in rows if not row.already_exists]
    return DiscoveryResponse(
        job_id=job.id,
        expanded_terms=job.expanded_terms,
        connector_status=(job.counts or {}).get("connectors", {}),
        total_found=len(rows),
        existing_count=len(rows) - len(new_rows),
        new_count=len(new_rows),
        candidates=[candidate_to_schema(row) for row in rows],
    )


def is_metadata_only(document: Document) -> bool:
    return document.fulltext_source == "metadata-only" or bool((document.metadata_json or {}).get("metadata_only"))


def create_single_document_job(
    db: Session,
    conversation: Conversation | None,
    knowledge_base_id: str,
    *,
    query: str,
    stage: str,
    upgraded: bool = False,
) -> ImportJob:
    job = ImportJob(
        conversation_id=conversation.id if conversation else None,
        knowledge_base_id=knowledge_base_id,
        query=query,
        requested_count=1,
        status=JobStatus.queued,
        stage=stage,
        counts={
            "priority": True,
            "execution_state": "queued",
            "worker_queue": get_settings().priority_queue_name,
            "selected": 1,
            "upgraded": 1 if upgraded else 0,
            "downloaded": 0,
            "parsed": 0,
            "indexed": 0,
            "failed": 0,
        },
    )
    db.add(job)
    db.flush()
    return job


def mark_document_for_job(document: Document, job: ImportJob, *, upgrade_pending: bool) -> None:
    metadata = dict(document.metadata_json or {})
    metadata["import_job_id"] = job.id
    metadata["upgrade_pending"] = upgrade_pending
    document.metadata_json = metadata
    document.status = DocumentStatus.selected
    job.selected_document_ids = [document.id]


def enqueue_or_background(job: ImportJob, background_tasks: BackgroundTasks) -> None:
    # User-provided documents and explicit full-text retries should not wait behind
    # a 50–200 paper discovery batch.
    if not enqueue_import(job.id, high_priority=True):
        background_tasks.add_task(run_import_job, job.id)


@router.get("/health", response_model=HealthRead)
def health() -> HealthRead:
    settings = get_settings()
    missing = []
    if not settings.openalex_api_key:
        missing.append("OPENALEX_API_KEY")
    if not settings.unpaywall_email:
        missing.append("UNPAYWALL_EMAIL")
    if not settings.openalex_email:
        missing.append("OPENALEX_EMAIL")
    return HealthRead(
        status="ok",
        database="sqlite" if settings.database_url.startswith("sqlite") else "postgresql",
        llm_configured=bool(settings.siliconflow_api_key),
        llm_model=settings.llm_model,
        chat_llm_model=settings.chat_llm_model,
        openalex_configured=bool(settings.openalex_api_key),
        unpaywall_configured=bool(settings.unpaywall_email),
        elsevier_configured=bool(settings.elsevier_api_key),
        neo4j_configured=bool(settings.neo4j_uri and settings.neo4j_password),
        chroma_path=str(settings.chroma_path),
        embedding_model=settings.embedding_model,
        embedding_download_allowed=settings.allow_embedding_download,
        embedding_fallback_allowed=settings.allow_embedding_fallback,
        hf_endpoint=settings.hf_endpoint,
        queue_mode="redis/rq" if settings.use_rq else "fastapi-background",
        missing_recommended_settings=missing,
    )


@router.post("/knowledge-bases", response_model=KnowledgeBaseRead, status_code=201)
def create_knowledge_base(payload: KnowledgeBaseCreate, db: Db) -> KnowledgeBase:
    knowledge_base = KnowledgeBase(name=payload.name, description=payload.description)
    db.add(knowledge_base)
    db.commit()
    db.refresh(knowledge_base)
    return knowledge_base


@router.get("/knowledge-bases", response_model=list[KnowledgeBaseRead])
def list_knowledge_bases(db: Db) -> list[KnowledgeBase]:
    return list(db.scalars(select(KnowledgeBase).order_by(KnowledgeBase.updated_at.desc())))


@router.post("/conversations", response_model=ConversationRead, status_code=201)
def create_conversation(payload: ConversationCreate, db: Db) -> Conversation:
    knowledge_base = db.get(KnowledgeBase, payload.knowledge_base_id) if payload.knowledge_base_id else None
    if payload.knowledge_base_id and not knowledge_base:
        raise HTTPException(404, "知识库不存在")
    if not knowledge_base:
        knowledge_base = db.scalar(select(KnowledgeBase).order_by(KnowledgeBase.created_at).limit(1))
    if not knowledge_base:
        knowledge_base = KnowledgeBase()
        db.add(knowledge_base)
        db.flush()
    conversation = Conversation(
        knowledge_base_id=knowledge_base.id,
        title=payload.title,
        index_version=knowledge_base.index_version,
    )
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    return conversation


@router.get("/conversations", response_model=list[ConversationRead])
def list_conversations(db: Db) -> list[Conversation]:
    return list(db.scalars(select(Conversation).order_by(Conversation.updated_at.desc()).limit(200)))


@router.get("/conversations/{conversation_id}", response_model=ConversationRead)
def get_conversation(conversation_id: str, db: Db) -> Conversation:
    return get_or_404(db, Conversation, conversation_id)


@router.delete("/conversations/{conversation_id}", response_model=ConversationDeleteResponse)
def delete_conversation(conversation_id: str, db: Db) -> ConversationDeleteResponse:
    """Delete chat-only data while preserving the shared knowledge base and its documents."""
    conversation = get_or_404(db, Conversation, conversation_id)
    deleted_messages = db.scalar(
        select(func.count()).select_from(Message).where(Message.conversation_id == conversation.id)
    ) or 0
    deleted_traces = db.scalar(
        select(func.count()).select_from(TrainingTrace).where(TrainingTrace.conversation_id == conversation.id)
    ) or 0
    preserved_insights = db.scalar(
        select(func.count()).select_from(KnowledgeInsight).where(KnowledgeInsight.conversation_id == conversation.id)
    ) or 0

    db.execute(delete(TrainingTrace).where(TrainingTrace.conversation_id == conversation.id))
    db.execute(delete(Message).where(Message.conversation_id == conversation.id))
    db.execute(
        update(KnowledgeInsight)
        .where(KnowledgeInsight.conversation_id == conversation.id)
        .values(conversation_id=None)
    )
    db.execute(
        update(ImportJob).where(ImportJob.conversation_id == conversation.id).values(conversation_id=None)
    )
    db.execute(
        update(LiteratureAutomation)
        .where(LiteratureAutomation.conversation_id == conversation.id)
        .values(conversation_id=None)
    )
    db.delete(conversation)
    db.commit()
    return ConversationDeleteResponse(
        id=conversation_id,
        deleted_messages=deleted_messages,
        deleted_training_traces=deleted_traces,
        preserved_insights=preserved_insights,
    )


@router.get("/conversations/{conversation_id}/messages", response_model=list[MessageRead])
def list_messages(conversation_id: str, db: Db) -> list[Message]:
    get_or_404(db, Conversation, conversation_id)
    return list(
        db.scalars(select(Message).where(Message.conversation_id == conversation_id).order_by(Message.created_at))
    )


@router.post("/discover", response_model=DiscoveryResponse)
async def discover(payload: DiscoveryRequest, db: Db) -> DiscoveryResponse:
    conversation = get_or_404(db, Conversation, payload.conversation_id)
    job = ImportJob(
        conversation_id=conversation.id,
        knowledge_base_id=conversation.knowledge_base_id,
        query=payload.query,
        status=JobStatus.queued,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    try:
        rows = await DiscoveryService().discover(
            db,
            job,
            limit=payload.limit,
            year_from=payload.year_from,
            year_to=payload.year_to,
            include_citation_expansion=payload.include_citation_expansion,
        )
    except Exception as exc:
        job.status = JobStatus.failed
        job.stage = "检索失败"
        job.error_message = f"{type(exc).__name__}: {exc}"
        append_job_log(job, "failed", job.error_message)
        db.commit()
        raise HTTPException(502, f"文献检索失败：{exc}") from exc
    return discovery_to_schema(job, rows)


@router.get("/jobs/{job_id}", response_model=JobRead)
def get_job(job_id: str, db: Db) -> ImportJob:
    return get_job_or_404(db, job_id)


@router.get("/knowledge-bases/{knowledge_base_id}/jobs", response_model=list[JobRead])
def list_knowledge_base_jobs(
    knowledge_base_id: str,
    db: Db,
    active_only: bool = False,
    limit: int = 50,
) -> list[ImportJob]:
    get_or_404(db, KnowledgeBase, knowledge_base_id)
    statement = select(ImportJob).where(ImportJob.knowledge_base_id == knowledge_base_id)
    if active_only:
        statement = statement.where(ImportJob.status.not_in((JobStatus.completed, JobStatus.failed)))
    requested_limit = max(1, min(limit, 200))
    rows = list(db.scalars(statement.order_by(ImportJob.updated_at.desc()).limit(1000)))
    return [item for item in rows if not item.deleted][:requested_limit]


@router.post("/jobs/{job_id}/pause", response_model=JobControlRead)
def pause_job(job_id: str, db: Db) -> JobControlRead:
    job = get_job_or_404(db, job_id)
    if job.status in {JobStatus.completed, JobStatus.failed, JobStatus.awaiting_selection}:
        raise HTTPException(409, "该任务当前状态不能暂停")
    execution_state = str((job.counts or {}).get("execution_state") or "queued")
    state = PAUSE_REQUESTED if execution_state == "running" else PAUSED
    control = request_job_control(db, job, state)
    counts = dict(job.counts or {})
    counts["execution_state"] = "pausing" if state == PAUSE_REQUESTED else "paused"
    job.counts = counts
    job.stage = "正在完成当前单篇后暂停" if state == PAUSE_REQUESTED else "已暂停；可继续任务"
    append_job_log(job, "pause_requested", "用户请求暂停；将在当前单篇安全点生效")
    automation = db.scalar(select(LiteratureAutomation).where(LiteratureAutomation.last_job_id == job.id))
    if automation and not automation_is_deleted(automation):
        automation.stop_requested = True
        automation.status = AutomationStatus.stopping
        automation.next_run_at = None
        automation.error_message = "当前轮次已暂停；继续该任务后自动采集将恢复"
    db.commit()
    removed = remove_queued_import_calls(job.id)
    message = "正在完成当前单篇，随后暂停" if state == PAUSE_REQUESTED else "任务已暂停"
    if removed:
        message += f"；已从队列移除 {removed} 个待执行调用"
    return JobControlRead(job_id=job.id, state=control.state, deleted=control.deleted, message=message)


@router.post("/jobs/{job_id}/start", response_model=JobControlRead)
def start_job(job_id: str, background_tasks: BackgroundTasks, db: Db) -> JobControlRead:
    job = get_job_or_404(db, job_id)
    if job.status in {JobStatus.completed, JobStatus.failed, JobStatus.awaiting_selection}:
        raise HTTPException(409, "该任务当前状态不能开始；等待选择的任务请先确认文献")
    control = get_job_control(db, job.id)
    if control and (control.deleted or control.state in CANCEL_STATES):
        raise HTTPException(409, "该任务已经取消或删除，不能再次开始")
    automation = db.scalar(select(LiteratureAutomation).where(LiteratureAutomation.last_job_id == job.id))
    if import_rq_call_is_running(job):
        if control and control.state == PAUSE_REQUESTED:
            clear_job_control(db, job)
            counts = dict(job.counts or {})
            counts["execution_state"] = "running"
            job.counts = counts
            current = dict(counts.get("current_document") or {})
            job.stage = current.get("stage") or "已撤销暂停，继续当前任务"
            append_job_log(job, "started", "用户点击开始/继续，已撤销尚未生效的暂停请求")
            if automation and not automation_is_deleted(automation):
                automation.stop_requested = False
                automation.status = AutomationStatus.running
                automation.error_message = None
            db.commit()
            return JobControlRead(
                job_id=job.id,
                state="active",
                deleted=False,
                message="已撤销暂停，当前任务继续执行",
            )
        raise HTTPException(409, "Worker 仍在处理当前文献；到达安全暂停点后即可开始/继续")
    clear_job_control(db, job)
    job.status = JobStatus.queued
    job.stage = "已开始，等待 Worker"
    job.error_message = None
    counts = dict(job.counts or {})
    counts["execution_state"] = "queued"
    job.counts = counts
    append_job_log(job, "started", "用户点击开始/继续，任务已重新进入后台队列")
    if automation and not automation_is_deleted(automation):
        automation.stop_requested = False
        automation.status = AutomationStatus.running
        automation.error_message = None
        automation.next_run_at = None
    db.commit()
    remove_queued_import_calls(job.id)
    if automation and not automation_is_deleted(automation):
        try:
            automation.rq_job_id = enqueue_automation_resume_cycle(automation.id)
            db.commit()
        except Exception as exc:
            request_job_control(db, job, PAUSED)
            counts = dict(job.counts or {})
            counts["execution_state"] = "paused"
            job.counts = counts
            job.stage = "开始失败，任务仍保持暂停"
            automation.stop_requested = True
            automation.status = AutomationStatus.stopped
            automation.error_message = f"{type(exc).__name__}: 无法连接 Redis/RQ"
            db.commit()
            raise HTTPException(502, automation.error_message) from exc
    elif not enqueue_import(job.id, high_priority=is_priority_job(job)):
        background_tasks.add_task(run_import_job, job.id)
    return JobControlRead(job_id=job.id, state="active", deleted=False, message="任务已开始")


@router.post("/jobs/{job_id}/resume", response_model=JobControlRead)
def resume_job(job_id: str, background_tasks: BackgroundTasks, db: Db) -> JobControlRead:
    job = get_job_or_404(db, job_id)
    control = get_job_control(db, job.id)
    if control is None or control.state not in {PAUSED, PAUSE_REQUESTED}:
        raise HTTPException(409, "该任务没有处于暂停或等待暂停状态")
    return start_job(job_id, background_tasks, db)


@router.delete("/jobs/{job_id}", response_model=JobControlRead)
def delete_job(job_id: str, db: Db) -> JobControlRead:
    job = get_job_or_404(db, job_id)
    execution_state = str((job.counts or {}).get("execution_state") or "queued")
    control = request_job_control(db, job, CANCEL_REQUESTED, deleted=True)
    counts = dict(job.counts or {})
    counts["execution_state"] = "cancelling" if execution_state == "running" else "cancel_requested"
    job.counts = counts
    job.stage = "正在完成当前单篇后删除任务" if execution_state == "running" else "正在取消并删除任务"
    append_job_log(job, "cancel_requested", "用户删除任务；已经入库的文献和知识不会删除")

    automation = db.scalar(select(LiteratureAutomation).where(LiteratureAutomation.last_job_id == job.id))
    if automation and not automation_is_deleted(automation):
        automation.stop_requested = True
        automation.status = AutomationStatus.stopping if execution_state == "running" else AutomationStatus.stopped
        automation.next_run_at = None
        automation.error_message = "当前轮次由用户删除，自动采集已停止"
    db.commit()
    remove_queued_import_calls(job.id)
    if execution_state != "running":
        apply_control_checkpoint(db, job)
    return JobControlRead(
        job_id=job.id,
        state=control.state,
        deleted=True,
        message="任务已从列表移除；已入库文献和知识继续保留",
    )


@router.get("/jobs/{job_id}/candidates", response_model=list[CandidateRead])
def get_candidates(job_id: str, db: Db, new_only: bool = False) -> list[CandidateRead]:
    get_job_or_404(db, job_id)
    return [candidate_to_schema(item) for item in candidates_for_job(db, job_id, new_only=new_only)]


@router.post("/jobs/{job_id}/selection", response_model=JobRead)
def select_candidates(
    job_id: str,
    payload: ImportSelectionRequest,
    background_tasks: BackgroundTasks,
    db: Db,
) -> ImportJob:
    job = get_job_or_404(db, job_id)
    if job.status != JobStatus.awaiting_selection:
        raise HTTPException(409, f"任务当前状态 {job.status.value}，不能选择")
    if payload.requested_count == 0:
        job.requested_count = 0
        job.selected_document_ids = []
        job.status = JobStatus.completed
        job.stage = "未新增，继续使用现有知识库"
        job.progress = 1.0
        job.completed_at = datetime.now(timezone.utc)
        append_job_log(job, "completed", "用户选择新增 0 篇")
        db.commit()
        return job

    try:
        document_ids = create_documents_from_candidates(db, job, payload.candidate_ids)
    except ImportSelectionError as exc:
        raise HTTPException(409, str(exc)) from exc
    if not document_ids:
        raise HTTPException(409, "所选文献均已在知识库中")
    db.commit()
    if not enqueue_import(job.id):
        background_tasks.add_task(run_import_job, job.id)
    return job


@router.post("/automations", response_model=AutomationRead, status_code=201)
def create_automation(payload: AutomationCreate, db: Db) -> LiteratureAutomation:
    settings = get_settings()
    if not settings.use_rq:
        raise HTTPException(409, "持续自动采集需要 Redis/RQ，请使用完整 Docker 模式启动")
    conversation = get_or_404(db, Conversation, payload.conversation_id)
    automation = LiteratureAutomation(
        conversation_id=conversation.id,
        knowledge_base_id=conversation.knowledge_base_id,
        name=payload.name or f"自动采集：{payload.query[:80]}",
        query=payload.query,
        batch_size=payload.batch_size,
        interval_minutes=payload.interval_minutes,
        max_total=payload.max_total,
        status=AutomationStatus.active,
    )
    db.add(automation)
    db.commit()
    db.refresh(automation)
    try:
        automation.rq_job_id = enqueue_automation_cycle(automation.id)
        db.commit()
    except Exception as exc:
        automation.status = AutomationStatus.failed
        automation.error_message = f"{type(exc).__name__}: 无法连接 Redis/RQ"
        db.commit()
        raise HTTPException(502, automation.error_message) from exc
    db.refresh(automation)
    return automation


@router.get("/automations", response_model=list[AutomationRead])
def list_automations(db: Db, knowledge_base_id: str | None = None) -> list[LiteratureAutomation]:
    statement = select(LiteratureAutomation)
    if knowledge_base_id:
        statement = statement.where(LiteratureAutomation.knowledge_base_id == knowledge_base_id)
    rows = list(db.scalars(statement.order_by(LiteratureAutomation.created_at.desc()).limit(500)))
    return [item for item in rows if not automation_is_deleted(item)][:200]


@router.get("/automations/{automation_id}", response_model=AutomationRead)
def get_automation(automation_id: str, db: Db) -> LiteratureAutomation:
    automation = get_or_404(db, LiteratureAutomation, automation_id)
    if automation_is_deleted(automation):
        raise HTTPException(404, "LiteratureAutomation 不存在或已删除")
    return automation


@router.post("/automations/{automation_id}/stop", response_model=AutomationRead)
def stop_automation(automation_id: str, db: Db) -> LiteratureAutomation:
    automation = get_automation(automation_id, db)
    automation.stop_requested = True
    if automation.status == AutomationStatus.running:
        automation.status = AutomationStatus.stopping
    else:
        automation.status = AutomationStatus.stopped
        automation.next_run_at = None
    if automation.last_job_id:
        last_job = db.get(ImportJob, automation.last_job_id)
        if last_job and last_job.status not in {JobStatus.completed, JobStatus.failed}:
            execution_state = str((last_job.counts or {}).get("execution_state") or "queued")
            request_job_control(db, last_job, CANCEL_REQUESTED)
            append_job_log(last_job, "stop_requested", "持续采集已停止；当前单篇完成后结束本轮")
            if execution_state != "running":
                apply_control_checkpoint(db, last_job)
    db.commit()
    remove_queued_automation_calls(automation.id)
    db.refresh(automation)
    return automation


@router.post("/automations/{automation_id}/restart", response_model=AutomationRead)
def restart_automation(automation_id: str, db: Db) -> LiteratureAutomation:
    automation = get_automation(automation_id, db)
    if automation.status in {AutomationStatus.active, AutomationStatus.running, AutomationStatus.stopping}:
        raise HTTPException(409, "该自动任务已经在运行或等待运行")
    automation.stop_requested = False
    automation.status = AutomationStatus.active
    automation.error_message = None
    automation.next_run_at = None
    db.commit()
    try:
        automation.rq_job_id = enqueue_automation_cycle(automation.id)
    except Exception as exc:
        automation.status = AutomationStatus.failed
        automation.error_message = f"{type(exc).__name__}: 无法连接 Redis/RQ"
        db.commit()
        raise HTTPException(502, automation.error_message) from exc
    db.commit()
    db.refresh(automation)
    return automation


@router.delete("/automations/{automation_id}", status_code=204)
def delete_automation(automation_id: str, db: Db) -> Response:
    automation = get_automation(automation_id, db)
    running = automation.status == AutomationStatus.running
    automation.stop_requested = True
    automation.status = AutomationStatus.stopping if running else AutomationStatus.stopped
    automation.next_run_at = None
    mark_automation_deleted(automation)
    if automation.last_job_id:
        last_job = db.get(ImportJob, automation.last_job_id)
        if last_job and last_job.status not in {JobStatus.completed, JobStatus.failed}:
            execution_state = str((last_job.counts or {}).get("execution_state") or "queued")
            request_job_control(db, last_job, CANCEL_REQUESTED, deleted=True)
            append_job_log(last_job, "cancel_requested", "所属持续采集任务已删除；已入库文献保留")
            if execution_state != "running":
                apply_control_checkpoint(db, last_job)
    db.commit()
    remove_queued_automation_calls(automation.id)
    return Response(status_code=204)


@router.get("/knowledge-bases/{knowledge_base_id}/documents", response_model=list[DocumentRead])
def list_documents(
    knowledge_base_id: str,
    db: Db,
    limit: int = 200,
    offset: int = 0,
    query: str | None = None,
) -> list[Document]:
    get_or_404(db, KnowledgeBase, knowledge_base_id)
    statement = select(Document).where(Document.knowledge_base_id == knowledge_base_id)
    if query:
        statement = statement.where(Document.title.ilike(f"%{query[:200]}%"))
    statement = statement.order_by(Document.updated_at.desc()).offset(max(0, offset)).limit(max(1, min(limit, 500)))
    return list(db.scalars(statement))


@router.get("/documents/{document_id}", response_model=DocumentDetail)
def get_document_detail(document_id: str, db: Db) -> DocumentDetail:
    document = get_or_404(db, Document, document_id)
    chunks = list(db.scalars(select(Chunk).where(Chunk.document_id == document.id).order_by(Chunk.chunk_index).limit(1000)))
    facts = list(
        db.scalars(
            select(ExtractedFact)
            .where(ExtractedFact.document_id == document.id)
            .order_by(ExtractedFact.confidence.desc())
            .limit(1000)
        )
    )
    return DocumentDetail(document=DocumentRead.model_validate(document), chunks=chunks, facts=facts)


@router.post("/conversations/{conversation_id}/upload", response_model=UploadResponse, status_code=201)
async def upload_licensed_document(
    conversation_id: str,
    background_tasks: BackgroundTasks,
    db: Db,
    file: Annotated[UploadFile, File(description="合法取得的 PDF 或 XML 全文")],
    title: Annotated[str, Form()],
    rights_confirmed: Annotated[bool, Form()],
    doi: Annotated[str | None, Form()] = None,
    authors_json: Annotated[str, Form()] = "[]",
    publication_year: Annotated[int | None, Form()] = None,
) -> UploadResponse:
    if not rights_confirmed:
        raise HTTPException(400, "必须确认你有权将该文献用于本知识库")
    conversation = get_or_404(db, Conversation, conversation_id)
    content = await file.read()
    max_size = 100 * 1024 * 1024
    if not content or len(content) > max_size:
        raise HTTPException(400, "文件为空或超过 100 MB")
    is_pdf = content.startswith(b"%PDF-")
    is_xml = content.lstrip().startswith(b"<")
    if not (is_pdf or is_xml):
        raise HTTPException(400, "仅支持真实 PDF/XML 文件")
    try:
        author_names = json.loads(authors_json)
        if not isinstance(author_names, list):
            raise ValueError
    except (json.JSONDecodeError, ValueError):
        raise HTTPException(400, "authors_json 必须是作者名字符串数组")
    authors = [{"name": str(name)} for name in author_names]
    fingerprint = title_author_fingerprint(title, authors)
    digest = sha256_bytes(content)
    duplicate, reason = find_duplicate(
        db,
        conversation.knowledge_base_id,
        doi=doi,
        fingerprint=fingerprint,
        file_sha256=digest,
    )
    upgraded = duplicate is not None
    if duplicate and not is_metadata_only(duplicate):
        raise HTTPException(409, f"文献全文已存在（查重层级：{reason}，document_id={duplicate.id}）")
    sha_duplicate, _ = find_duplicate(db, conversation.knowledge_base_id, file_sha256=digest)
    if sha_duplicate and (not duplicate or sha_duplicate.id != duplicate.id):
        raise HTTPException(409, f"相同全文文件已存在（document_id={sha_duplicate.id}）")
    job = create_single_document_job(
        db,
        conversation,
        conversation.knowledge_base_id,
        query=f"用户上传：{title}",
        stage="已接收合法上传",
        upgraded=upgraded,
    )
    if duplicate:
        document = duplicate
        document.doi = document.doi or doi
        document.doi_normalized = document.doi_normalized or normalize_doi(doi)
        document.authors = document.authors or authors
        document.publication_year = document.publication_year or publication_year
    else:
        document = Document(
            knowledge_base_id=conversation.knowledge_base_id,
            doi=doi,
            doi_normalized=normalize_doi(doi),
            title=title,
            title_author_fingerprint=fingerprint,
            authors=authors,
            publication_year=publication_year,
        )
        db.add(document)
        db.flush()
    document.file_sha256 = digest
    document.fulltext_source = "user-upload"
    document.license = "user-confirmed-rights"
    mark_document_for_job(document, job, upgrade_pending=upgraded)
    extension = "pdf" if is_pdf else "xml"
    key = f"kb/{conversation.knowledge_base_id}/documents/{document.id}/source.{extension}"
    ObjectStorage().put_bytes(key, content, "application/pdf" if is_pdf else "application/xml")
    document.object_key = key
    metadata = dict(document.metadata_json or {})
    metadata["original_filename"] = file.filename
    document.metadata_json = metadata
    action = "升级已有题名/摘要记录" if upgraded else "新增文献"
    append_job_log(job, "upload", f"已接收用户授权上传并{action}：{document.title}", document.id)
    db.commit()
    enqueue_or_background(job, background_tasks)
    message = "全文已进入后台升级解析队列" if upgraded else "文件已进入后台解析队列"
    return UploadResponse(job_id=job.id, document_id=document.id, sha256=digest, message=message)


@router.post("/conversations/{conversation_id}/url-import", response_model=UploadResponse, status_code=201)
async def import_public_url(
    conversation_id: str,
    payload: PublicUrlImportRequest,
    background_tasks: BackgroundTasks,
    db: Db,
) -> UploadResponse:
    conversation = get_or_404(db, Conversation, conversation_id)
    resolver = FullTextResolver()
    try:
        result = await resolver.resolve_public_source(payload.source)
    except (httpx.HTTPError, ValueError) as exc:
        detail = FullTextResolver._safe_error("公开网址读取失败", exc)
        raise HTTPException(422, detail) from exc
    finally:
        await resolver.close()

    parsed_path = urlparse(payload.source).path.rstrip("/")
    fallback_title = parsed_path.rsplit("/", 1)[-1].replace("-", " ") if parsed_path else "公开全文"
    title = (payload.title or result.title or result.doi or fallback_title).strip()[:2000]
    doi = normalize_doi(payload.doi) or result.doi or extract_doi(payload.source)
    authors = [{"name": item.strip()} for item in payload.authors if item.strip()]
    fingerprint = title_author_fingerprint(title, authors)
    duplicate, reason = find_duplicate(
        db,
        conversation.knowledge_base_id,
        doi=doi,
        fingerprint=fingerprint,
        file_sha256=result.sha256,
    )
    if not duplicate:
        duplicate = db.scalar(
            select(Document).where(
                Document.knowledge_base_id == conversation.knowledge_base_id,
                or_(Document.landing_url == result.url, Document.fulltext_url == result.url),
            )
        )
        reason = "source_url" if duplicate else reason
    upgraded = duplicate is not None
    if duplicate and not is_metadata_only(duplicate):
        raise HTTPException(409, f"文献全文已存在（查重层级：{reason}，document_id={duplicate.id}）")
    sha_duplicate, _ = find_duplicate(db, conversation.knowledge_base_id, file_sha256=result.sha256)
    if sha_duplicate and (not duplicate or sha_duplicate.id != duplicate.id):
        raise HTTPException(409, f"相同全文文件已存在（document_id={sha_duplicate.id}）")

    job = create_single_document_job(
        db,
        conversation,
        conversation.knowledge_base_id,
        query=f"公开网址/DOI 导入：{title}",
        stage="已获取公开全文",
        upgraded=upgraded,
    )
    if duplicate:
        document = duplicate
        document.doi = document.doi or doi
        document.doi_normalized = document.doi_normalized or doi
        document.authors = document.authors or authors
        document.publication_year = document.publication_year or payload.publication_year
    else:
        document = Document(
            knowledge_base_id=conversation.knowledge_base_id,
            doi=doi,
            doi_normalized=doi,
            title=title,
            title_author_fingerprint=fingerprint,
            authors=authors,
            publication_year=payload.publication_year,
        )
        db.add(document)
        db.flush()
    document.file_sha256 = result.sha256
    document.fulltext_source = result.source
    document.fulltext_url = result.url
    document.landing_url = document.landing_url or result.url
    document.license = result.license or document.license
    document.is_open_access = True
    mark_document_for_job(document, job, upgrade_pending=upgraded)
    key = f"kb/{conversation.knowledge_base_id}/documents/{document.id}/source.{result.extension}"
    ObjectStorage().put_bytes(key, result.content, result.content_type)
    document.object_key = key
    metadata = dict(document.metadata_json or {})
    metadata.update({"public_access_confirmed": True, "direct_public_source": result.source})
    document.metadata_json = metadata
    action = "升级已有题名/摘要记录" if upgraded else "新增文献"
    append_job_log(job, "public_download", f"已直接取得公开全文并{action}：{document.title}", document.id)
    db.commit()
    enqueue_or_background(job, background_tasks)
    return UploadResponse(
        job_id=job.id,
        document_id=document.id,
        sha256=result.sha256,
        message="公开全文已进入后台升级解析队列" if upgraded else "公开全文已进入后台解析队列",
    )


@router.post("/documents/{document_id}/retry-fulltext", response_model=JobRead)
def retry_document_fulltext(
    document_id: str,
    background_tasks: BackgroundTasks,
    db: Db,
) -> ImportJob:
    document = get_or_404(db, Document, document_id)
    if not is_metadata_only(document):
        raise HTTPException(409, "该文献不是题名/摘要模式，无需重新获取全文")
    if not (document.doi_normalized or document.fulltext_url or document.landing_url):
        raise HTTPException(409, "该文献没有 DOI 或公开网址，请使用公开网址/DOI 导入或上传全文")
    job = create_single_document_job(
        db,
        None,
        document.knowledge_base_id,
        query=f"重新获取公开全文：{document.title}",
        stage="等待重新检查公开全文",
        upgraded=True,
    )
    mark_document_for_job(document, job, upgrade_pending=True)
    append_job_log(job, "retry", f"重新检查公开 PDF/HTML/XML：{document.title}", document.id)
    db.commit()
    enqueue_or_background(job, background_tasks)
    return job


@router.post("/knowledge-bases/{knowledge_base_id}/retry-metadata-only", response_model=JobRead)
def retry_metadata_only_documents(
    knowledge_base_id: str,
    payload: RetryMetadataRequest,
    background_tasks: BackgroundTasks,
    db: Db,
) -> ImportJob:
    get_or_404(db, KnowledgeBase, knowledge_base_id)
    candidates = list(
        db.scalars(
            select(Document)
            .where(Document.knowledge_base_id == knowledge_base_id)
            .order_by(Document.updated_at)
            .limit(1000)
        )
    )
    documents = [
        item
        for item in candidates
        if is_metadata_only(item) and (item.doi_normalized or item.fulltext_url or item.landing_url)
    ][: payload.max_documents]
    if not documents:
        raise HTTPException(409, "没有带 DOI/网址、可重新检查的题名/摘要文献")
    job = ImportJob(
        knowledge_base_id=knowledge_base_id,
        query="批量重新检查 metadata-only 公开全文",
        requested_count=len(documents),
        status=JobStatus.queued,
        stage="等待批量重新检查公开全文",
        counts={
            "execution_state": "queued",
            "worker_queue": get_settings().queue_name,
            "selected": len(documents),
            "upgraded": len(documents),
            "downloaded": 0,
            "parsed": 0,
            "indexed": 0,
            "failed": 0,
        },
    )
    db.add(job)
    db.flush()
    for document in documents:
        mark_document_for_job(document, job, upgrade_pending=True)
    job.selected_document_ids = [item.id for item in documents]
    append_job_log(job, "retry", f"将重新检查 {len(documents)} 篇题名/摘要文献的公开全文")
    db.commit()
    if not enqueue_import(job.id):
        background_tasks.add_task(run_import_job, job.id)
    return job


@router.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, db: Db) -> ChatResponse:
    conversation = get_or_404(db, Conversation, payload.conversation_id)
    kb = get_or_404(db, KnowledgeBase, conversation.knowledge_base_id)
    user_message = Message(conversation_id=conversation.id, role="user", content=payload.question)
    db.add(user_message)
    if conversation.title == "新对话":
        conversation.title = payload.question[:80]
    conversation.current_task = payload.question
    conversation.index_version = kb.index_version
    db.commit()

    agent = NanofiltrationRAGAgent(db, deep_thinking=payload.deep_thinking)
    try:
        result = await agent.answer(
            conversation,
            payload.question,
            enable_knowledge_discovery=payload.knowledge_discovery,
            research_mode=payload.research_mode,
        )
    except Exception as exc:
        raise HTTPException(502, f"问答流程失败：{type(exc).__name__}: {exc}") from exc
    discovery_payload: dict | None = None
    literature_error: str | None = None
    answer = result.answer
    if payload.proactive_literature and payload.desired_new_count > 0 and result.needs_literature:
        discovery_job = ImportJob(
            conversation_id=conversation.id,
            knowledge_base_id=conversation.knowledge_base_id,
            query=result.literature_query or result.standalone_question,
            status=JobStatus.queued,
            stage="对话 Agent 主动发现",
        )
        db.add(discovery_job)
        db.commit()
        db.refresh(discovery_job)
        try:
            rows = await DiscoveryService().discover(
                db,
                discovery_job,
                limit=min(500, max(80, payload.desired_new_count * 3)),
                year_from=None,
                year_to=None,
                include_citation_expansion=True,
            )
            discovery_model = discovery_to_schema(discovery_job, rows)
            discovery_payload = discovery_model.model_dump(mode="json")
            selectable = min(payload.desired_new_count, discovery_model.new_count)
            answer += (
                f"\n\n我还发现了 {discovery_model.new_count} 篇去重后的新候选，"
                f"已按相关性预选前 {selectable} 篇；请在当前页面确认后再入库。"
            )
            result.tool_calls.append(
                {"tool": "proactive_literature_discovery", "job_id": discovery_job.id, "new_count": discovery_model.new_count}
            )
        except Exception as exc:
            db.rollback()
            discovery_job = db.get(ImportJob, discovery_job.id)
            if discovery_job:
                discovery_job.status = JobStatus.failed
                discovery_job.stage = "主动检索失败"
                discovery_job.error_message = f"{type(exc).__name__}: 上游检索失败"
                db.commit()
            literature_error = f"{type(exc).__name__}: 主动检索暂时失败，可稍后在文献模块重试"
            answer += "\n\n主动文献检索本轮未完成；现有知识库回答仍然保留，可稍后从文献模块重试。"
    assistant_message = Message(
        conversation_id=conversation.id,
        role="assistant",
        content=answer,
        evidence=result.evidence,
        tool_calls=result.tool_calls,
        model_name=agent.llm.model_name,
    )
    db.add(assistant_message)
    db.flush()
    trace = TrainingTrace(
        conversation_id=conversation.id,
        message_id=assistant_message.id,
        instruction=(
            "作为纳滤知识发现 Agent，仅以知识库文献、结构化事实和明确标注的 AI 假设为依据，"
            "回答问题并逐项引用 DOI 与页码；不得把模型参数记忆当作证据。"
        ),
        input_text=payload.question,
        output_text=answer,
        retrieval_evidence=result.evidence,
        tool_trace=result.tool_calls,
    )
    db.add(trace)
    conversation.current_task = None
    db.commit()
    await agent.maybe_update_rolling_summary(conversation)
    db.refresh(assistant_message)
    return ChatResponse(
        message=MessageRead.model_validate(assistant_message),
        route=result.route,
        evidence_count=len(result.evidence),
        standalone_question=result.standalone_question,
        needs_literature=result.needs_literature,
        literature_query=result.literature_query,
        literature_reason=result.literature_reason,
        discovery=discovery_payload,
        literature_error=literature_error,
        insight_count=len(result.insights),
    )


@router.get("/knowledge-bases/{knowledge_base_id}/insights", response_model=list[KnowledgeInsightRead])
def list_knowledge_insights(
    knowledge_base_id: str,
    db: Db,
    limit: int = 200,
    status: str | None = None,
    conversation_id: str | None = None,
) -> list[KnowledgeInsight]:
    get_or_404(db, KnowledgeBase, knowledge_base_id)
    statement = select(KnowledgeInsight).where(KnowledgeInsight.knowledge_base_id == knowledge_base_id)
    if status:
        statement = statement.where(KnowledgeInsight.status == status[:40])
    if conversation_id:
        statement = statement.where(KnowledgeInsight.conversation_id == conversation_id)
    statement = statement.order_by(KnowledgeInsight.created_at.desc()).limit(max(1, min(limit, 500)))
    return list(db.scalars(statement))


@router.post("/insights/{insight_id}/review", response_model=KnowledgeInsightRead)
def review_knowledge_insight(
    insight_id: str,
    payload: KnowledgeInsightReview,
    db: Db,
) -> KnowledgeInsight:
    insight = get_or_404(db, KnowledgeInsight, insight_id)
    if payload.status == "validated" and not (payload.review_note or "").strip():
        raise HTTPException(400, "标记为实验验证时必须填写验证依据或实验记录")
    insight.status = payload.status
    insight.review_note = payload.review_note
    insight.reviewed_at = datetime.now(timezone.utc)
    db.commit()
    graph = GraphStore(get_settings())
    try:
        graph.update_insight_status(insight.id, insight.status, insight.review_note)
    finally:
        graph.close()
    db.refresh(insight)
    return insight


@router.post("/feedback", status_code=204)
def record_feedback(payload: FeedbackRequest, db: Db) -> None:
    trace = db.scalar(select(TrainingTrace).where(TrainingTrace.message_id == payload.message_id))
    if not trace:
        raise HTTPException(404, "该回答没有训练轨迹")
    trace.rating = payload.rating
    trace.human_revision = payload.human_revision
    trace.approved_for_training = payload.approved_for_training
    db.commit()


def training_traces_for_query(
    db: Session,
    *,
    knowledge_base_id: str | None,
    conversation_id: str | None,
    approved_only: bool,
    min_rating: int,
    limit: int | None,
) -> list[TrainingTrace]:
    statement = select(TrainingTrace).join(Conversation, Conversation.id == TrainingTrace.conversation_id)
    if knowledge_base_id:
        statement = statement.where(Conversation.knowledge_base_id == knowledge_base_id)
    if conversation_id:
        statement = statement.where(TrainingTrace.conversation_id == conversation_id)
    if approved_only:
        statement = statement.where(TrainingTrace.approved_for_training.is_(True))
    if min_rating > 0:
        statement = statement.where(TrainingTrace.rating >= min(5, min_rating))
    statement = statement.order_by(TrainingTrace.created_at.desc())
    if limit is not None:
        statement = statement.limit(max(1, min(limit, 1000)))
    return list(db.scalars(statement))


@router.get("/training/traces", response_model=list[TrainingTraceRead])
def list_training_traces(
    db: Db,
    knowledge_base_id: str | None = None,
    conversation_id: str | None = None,
    approved_only: bool = False,
    min_rating: int = 0,
    limit: int = 200,
) -> list[dict]:
    rows = training_traces_for_query(
        db,
        knowledge_base_id=knowledge_base_id,
        conversation_id=conversation_id,
        approved_only=approved_only,
        min_rating=max(0, min_rating),
        limit=limit,
    )
    return [trace_to_read(row) for row in rows]


@router.get("/training/stats", response_model=TrainingStatsRead)
def get_training_stats(db: Db, knowledge_base_id: str | None = None) -> TrainingStatsRead:
    rows = training_traces_for_query(
        db,
        knowledge_base_id=knowledge_base_id,
        conversation_id=None,
        approved_only=False,
        min_rating=0,
        limit=None,
    )
    return TrainingStatsRead(
        total=len(rows),
        rated=sum(row.rating is not None for row in rows),
        approved=sum(row.approved_for_training for row in rows),
        evidence_backed=sum(bool(row.retrieval_evidence) for row in rows),
        high_quality=sum(training_quality(row)[0] >= 0.75 for row in rows),
    )


@router.post("/training/traces/{trace_id}/review", response_model=TrainingTraceRead)
def review_training_trace(trace_id: str, payload: TrainingTraceReview, db: Db) -> dict:
    trace = get_or_404(db, TrainingTrace, trace_id)
    trace.rating = payload.rating
    trace.human_revision = (payload.human_revision or "").strip() or None
    trace.approved_for_training = payload.approved_for_training
    db.commit()
    db.refresh(trace)
    return trace_to_read(trace)


@router.get("/training/export")
def export_training_data(
    db: Db,
    knowledge_base_id: str | None = None,
    conversation_id: str | None = None,
    approved_only: bool = True,
    min_rating: int = 0,
) -> Response:
    rows = training_traces_for_query(
        db,
        knowledge_base_id=knowledge_base_id,
        conversation_id=conversation_id,
        approved_only=approved_only,
        min_rating=max(0, min_rating),
        limit=None,
    )
    return Response(
        content=traces_to_jsonl(reversed(rows)),
        media_type="application/x-ndjson; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="nf-atlas-training.jsonl"'},
    )
