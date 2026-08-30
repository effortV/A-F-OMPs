from __future__ import annotations

import asyncio
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import SessionLocal
from app.models import Chunk, Document, DocumentStatus, ImportJob, JobStatus, KnowledgeBase
from app.services.dedupe import find_duplicate
from app.services.extractor import FactExtractor
from app.services.fulltext import FullTextResolver, FullTextResult, FullTextUnavailable
from app.services.graph_store import GraphStore
from app.services.parser import DocumentParser, ParsedBlock, ParsedDocument, chunk_document
from app.services.storage import ObjectStorage
from app.services.task_control import (
    ACTIVE,
    CANCEL_REQUESTED,
    CANCEL_STATES,
    CANCELLED,
    PAUSE_STATES,
    PAUSED,
    get_job_control,
)
from app.services.vector_store import VectorStore


def document_processing_priority(document: Document) -> tuple[int, int, int, int, float]:
    """Process owned bytes and strong public-fulltext signals before metadata fallbacks."""
    fulltext_hint = (document.fulltext_url or "").casefold().split("?", 1)[0]
    direct_file_hint = fulltext_hint.endswith((".pdf", ".xml", ".jats"))
    return (
        0 if document.object_key else 1,
        0 if direct_file_hint else 1,
        0 if document.is_open_access and document.doi_normalized else 1,
        0 if document.doi_normalized else 1,
        -float(document.relevance_score or 0),
    )


def append_job_log(job: ImportJob, stage: str, message: str, document_id: str | None = None) -> None:
    entry = {
        "time": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "message": message,
    }
    if document_id:
        entry["document_id"] = document_id
    job.log = [*job.log, entry][-500:]


def update_count(job: ImportJob, key: str, increment: int = 1) -> None:
    counts = dict(job.counts)
    counts[key] = int(counts.get(key, 0)) + increment
    job.counts = counts


def content_mode_label(result: FullTextResult) -> str:
    extension = (result.extension or "").casefold()
    if extension == "pdf":
        return "全文 PDF"
    if extension in {"xml", "jats"}:
        return "全文 XML/JATS"
    if extension in {"html", "htm"}:
        return "全文 HTML"
    return "全文/解析文本"


def update_current_document(
    job: ImportJob,
    document: Document,
    *,
    position: int,
    total: int,
    stage: str,
    document_progress: float,
    content_mode: str | None = None,
    chunks: int | None = None,
    facts: int | None = None,
    state: str = "processing",
) -> None:
    counts = dict(job.counts or {})
    current = dict(counts.get("current_document") or {})
    current.update(
        {
            "document_id": document.id,
            "title": document.title,
            "doi": document.doi_normalized,
            "position": position,
            "total": total,
            "stage": stage,
            "document_progress": round(max(0.0, min(1.0, document_progress)), 4),
            "document_status": getattr(document.status, "value", str(document.status)),
            "content_mode": content_mode or current.get("content_mode") or "正在判断全文可用性",
            "fulltext_source": document.fulltext_source,
            "has_saved_file": bool(document.object_key),
            "state": state,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    if chunks is not None:
        current["chunks"] = chunks
    if facts is not None:
        current["facts"] = facts
    counts["current_document"] = current
    counts["processing_position"] = position
    counts["processing_total"] = total
    job.counts = counts
    job.stage = stage
    job.progress = min(0.99, 0.25 + 0.7 * ((position - 1 + document_progress) / max(1, total)))


def apply_control_checkpoint(db: Session, job: ImportJob, *, allow_pause: bool = True) -> str:
    """Apply a durable user request only between document transactions.

    A running PDF is allowed to finish its current parse/extract/index transaction.
    This keeps PostgreSQL, Neo4j and Chroma consistent while still stopping before
    the next paper. The return value is active, paused or cancelled.
    """

    control = get_job_control(db, job.id)
    if control is None:
        return ACTIVE
    if control.deleted and control.state not in CANCEL_STATES:
        control.state = CANCEL_REQUESTED
    if control.state in CANCEL_STATES:
        was_terminal = control.state == CANCELLED
        control.state = CANCELLED
        counts = dict(job.counts or {})
        counts["execution_state"] = "cancelled"
        current = dict(counts.get("current_document") or {})
        if current:
            current.update(
                {
                    "state": "cancelled",
                    "stage": "已在单篇安全点停止",
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            counts["current_document"] = current
        job.counts = counts
        job.status = JobStatus.completed
        job.stage = "已取消；已入库文献保留"
        job.completed_at = datetime.now(timezone.utc)
        if not was_terminal:
            append_job_log(job, "cancelled", "用户取消任务；已完成入库的文献和知识保留")
        db.commit()
        return CANCELLED
    if control.state in PAUSE_STATES:
        if not allow_pause:
            control.state = ACTIVE
            db.commit()
            return ACTIVE
        was_paused = control.state == PAUSED
        control.state = PAUSED
        counts = dict(job.counts or {})
        counts["execution_state"] = "paused"
        current = dict(counts.get("current_document") or {})
        if current:
            current.update(
                {
                    "state": "paused",
                    "stage": "已在单篇安全点暂停",
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            counts["current_document"] = current
        job.counts = counts
        job.status = JobStatus.queued
        job.stage = "已暂停；可继续任务"
        job.completed_at = None
        if not was_paused:
            append_job_log(job, "paused", "任务已在单篇安全点暂停；已入库文献保留")
        db.commit()
        return PAUSED
    return ACTIVE


async def process_document(
    db: Session,
    job: ImportJob,
    document: Document,
    *,
    position: int = 1,
    total: int = 1,
) -> bool:
    settings = get_settings()
    resolver = FullTextResolver(settings)
    storage = ObjectStorage(settings)
    document_id = document.id
    job_id = job.id
    document_title = document.title
    try:
        job.status = JobStatus.downloading
        document.status = DocumentStatus.downloading
        update_current_document(
            job,
            document,
            position=position,
            total=total,
            stage="检查公开全文/已上传文件",
            document_progress=0.05,
        )
        db.commit()
        if document.object_key and document.fulltext_source not in {None, "metadata-only"}:
            content = storage.get_bytes(document.object_key)
            extension = Path(document.object_key).suffix.lstrip(".").lower() or "pdf"
            content_types = {"pdf": "application/pdf", "xml": "application/xml", "html": "text/html", "htm": "text/html"}
            result = FullTextResult(
                content=content,
                content_type=content_types.get(extension, "application/octet-stream"),
                extension=extension,
                source=document.fulltext_source,
                url=document.fulltext_url or document.fulltext_source,
                license=document.license,
                sha256=document.file_sha256 or "",
            )
        else:
            result = await resolver.resolve_and_download(
                doi=document.doi_normalized,
                hinted_url=document.fulltext_url,
                landing_url=document.landing_url,
                hinted_open_access=document.is_open_access,
            )
        duplicate, reason = find_duplicate(
            db,
            document.knowledge_base_id,
            file_sha256=result.sha256,
        )
        if duplicate and duplicate.id != document.id:
            append_job_log(job, "dedupe", f"全文 SHA-256 与已有文献重复，跳过：{duplicate.title}", document.id)
            db.delete(document)
            update_count(job, "existing")
            db.commit()
            return False
        key = document.object_key or f"kb/{document.knowledge_base_id}/documents/{document.id}/source.{result.extension}"
        if not document.object_key:
            storage.put_bytes(key, result.content, result.content_type)
        document.object_key = key
        document.file_sha256 = result.sha256
        document.fulltext_source = result.source
        document.fulltext_url = result.url
        document.license = result.license or document.license
        document.is_open_access = document.is_open_access or result.source.startswith(
            ("direct-public", "doi-public", "unpaywall", "europe-pmc")
        )
        document.status = DocumentStatus.downloaded
        document.error_message = None
        update_count(job, "downloaded")
        update_count(job, "fulltext")
        mode = content_mode_label(result)
        update_current_document(
            job,
            document,
            position=position,
            total=total,
            stage="全文已保存，准备解析",
            document_progress=0.22,
            content_mode=mode,
        )
        db.commit()

        job.status = JobStatus.parsing
        document.status = DocumentStatus.downloaded
        update_current_document(
            job,
            document,
            position=position,
            total=total,
            stage="解析正文/表格/图注",
            document_progress=0.35,
            content_mode=mode,
        )
        db.commit()
        suffix = "." + result.extension
        with tempfile.TemporaryDirectory(prefix="nf-parse-") as temp_dir:
            source_path = Path(temp_dir) / f"source{suffix}"
            source_path.write_bytes(result.content)
            parsed = DocumentParser(settings).parse(source_path, result.content_type)
        return await _index_parsed_document(
            db,
            job,
            document,
            parsed,
            storage,
            metadata_only=False,
            position=position,
            total=total,
            content_mode=mode,
        )
    except FullTextUnavailable as exc:
        # A paper with no legally available full text is still useful for discovery and RAG.
        # The metadata-only marker prevents abstract claims from being presented as full-text facts.
        blocks = [ParsedBlock(kind="title", section="Title", text=document.title)]
        if document.abstract:
            blocks.append(ParsedBlock(kind="abstract", section="Abstract", text=document.abstract))
        parsed = ParsedDocument(
            title=document.title,
            blocks=blocks,
            parser="metadata-only",
            warnings=["全文不可用；仅使用题名和摘要，页码不可解析"],
        )
        metadata = dict(document.metadata_json or {})
        reason = str(exc).strip()[:2000] or type(exc).__name__
        metadata.update({"metadata_only": True, "fulltext_unavailable_reason": reason})
        document.metadata_json = metadata
        document.fulltext_source = "metadata-only"
        document.error_message = f"全文不可用，已使用题名/摘要入库：{reason}"
        update_count(job, "metadata_only")
        update_current_document(
            job,
            document,
            position=position,
            total=total,
            stage="全文不可用，构建题名/摘要切片",
            document_progress=0.35,
            content_mode="题名/摘要",
        )
        append_job_log(job, "metadata_only", f"全文不可用，已以题名/摘要入库：{document.title}；原因：{reason}", document.id)
        try:
            return await _index_parsed_document(
                db,
                job,
                document,
                parsed,
                storage,
                metadata_only=True,
                position=position,
                total=total,
                content_mode="题名/摘要",
            )
        except Exception as index_exc:
            db.rollback()
            document = db.get(Document, document.id)
            job = db.get(ImportJob, job.id)
            if document and job:
                document.status = DocumentStatus.failed
                detail = str(index_exc).strip()[:1500]
                document.error_message = f"{type(index_exc).__name__}: 摘要入库失败；{detail}"
                update_count(job, "failed")
                update_current_document(
                    job,
                    document,
                    position=position,
                    total=total,
                    stage="题名/摘要入库失败",
                    document_progress=1.0,
                    content_mode="题名/摘要",
                    state="failed",
                )
                append_job_log(
                    job,
                    "failed",
                    f"题名/摘要入库失败：{document.title}；{type(index_exc).__name__}: {detail}",
                    document.id,
                )
                db.commit()
            return False
    except Exception as exc:
        # A flush/commit error leaves SQLAlchemy in pending-rollback state. Roll
        # back before touching ORM attributes so the original error remains visible.
        db.rollback()
        document = db.get(Document, document_id)
        job = db.get(ImportJob, job_id)
        if document and job:
            detail = str(exc).strip()[:2000]
            document.status = DocumentStatus.failed
            document.error_message = f"{type(exc).__name__}: {detail}"
            update_count(job, "failed")
            update_current_document(
                job,
                document,
                position=position,
                total=total,
                stage="文献处理失败",
                document_progress=1.0,
                state="failed",
            )
            append_job_log(
                job,
                "failed",
                f"处理失败：{document_title}；{type(exc).__name__}: {detail}",
                document_id,
            )
            db.commit()
        return False
    finally:
        await resolver.close()


async def _index_parsed_document(
    db: Session,
    job: ImportJob,
    document: Document,
    parsed: ParsedDocument,
    storage: ObjectStorage,
    *,
    metadata_only: bool,
    position: int,
    total: int,
    content_mode: str,
) -> bool:
    settings = get_settings()
    parsed_key = f"kb/{document.knowledge_base_id}/documents/{document.id}/parsed.json"
    storage.put_json(parsed_key, parsed.to_dict())
    document.parsed_object_key = parsed_key
    db.execute(delete(Chunk).where(Chunk.document_id == document.id))
    rows: list[Chunk] = []
    for index, item in enumerate(chunk_document(parsed, settings.chunk_size, settings.chunk_overlap)):
        row = Chunk(
            document_id=document.id,
            chunk_index=index,
            section=item["section"],
            block_kind=item["metadata"].get("kind"),
            source_label=item["metadata"].get("label"),
            page_start=item["page_start"],
            page_end=item["page_end"],
            text=item["text"],
            token_count=max(1, len(item["text"]) // 3),
            vector_id=f"{document.id}:{index}",
        )
        db.add(row)
        rows.append(row)
    document.status = DocumentStatus.parsed
    update_count(job, "parsed")
    update_current_document(
        job,
        document,
        position=position,
        total=total,
        stage="切片完成，准备结构化抽取",
        document_progress=0.52,
        content_mode=content_mode,
        chunks=len(rows),
    )
    db.flush()
    db.commit()

    job.status = JobStatus.extracting
    update_current_document(
        job,
        document,
        position=position,
        total=total,
        stage="DeepSeek 摘要抽取" if metadata_only else "DeepSeek 全文结构化抽取",
        document_progress=0.62,
        content_mode=content_mode,
        chunks=len(rows),
    )
    db.commit()
    facts = await FactExtractor().extract_document(db, document)
    document.status = DocumentStatus.extracted
    update_current_document(
        job,
        document,
        position=position,
        total=total,
        stage="结构化事实完成，准备图谱/向量入库",
        document_progress=0.78,
        content_mode=content_mode,
        chunks=len(rows),
        facts=len(facts),
    )
    db.commit()

    job.status = JobStatus.indexing
    update_current_document(
        job,
        document,
        position=position,
        total=total,
        stage="Neo4j 与 Chroma/bge-m3 入库",
        document_progress=0.86,
        content_mode=content_mode,
        chunks=len(rows),
        facts=len(facts),
    )
    db.commit()
    graph = GraphStore(settings)
    try:
        graph.replace_document(document, facts)
    finally:
        graph.close()
    vector_store = VectorStore(settings)
    vector_store.replace_document(document, rows)
    metadata = dict(document.metadata_json or {})
    metadata["embedding_backend"] = vector_store.embedder.backend
    metadata.pop("upgrade_pending", None)
    if metadata_only:
        metadata["metadata_only"] = True
    else:
        metadata.pop("metadata_only", None)
        metadata.pop("fulltext_unavailable_reason", None)
        metadata["fulltext_mode"] = "fulltext"
    document.metadata_json = metadata
    document.error_message = None if not metadata_only else document.error_message
    document.status = DocumentStatus.indexed
    update_count(job, "indexed")
    update_current_document(
        job,
        document,
        position=position,
        total=total,
        stage="当前文献已完成入库",
        document_progress=1.0,
        content_mode=content_mode,
        chunks=len(rows),
        facts=len(facts),
        state="completed",
    )
    suffix = "，题名/摘要模式" if metadata_only else ""
    append_job_log(
        job,
        "indexed",
        (
            f"已入库：{document.title}（{len(rows)} 个切片，{len(facts)} 条事实{suffix}；"
            f"向量模型 {vector_store.embedder.backend}）"
        ),
        document.id,
    )
    db.commit()
    return True


async def run_import_job_async(job_id: str) -> None:
    db = SessionLocal()
    try:
        job = db.get(ImportJob, job_id)
        if not job:
            return
        if apply_control_checkpoint(db, job) != ACTIVE:
            return
        counts = dict(job.counts or {})
        counts["execution_state"] = "running"
        try:
            from rq import get_current_job

            rq_job = get_current_job()
            counts["worker_queue"] = rq_job.origin if rq_job else counts.get("worker_queue")
            counts["rq_job_id"] = rq_job.id if rq_job else counts.get("rq_job_id")
        except Exception:
            counts["worker_queue"] = counts.get("worker_queue") or "fastapi-background"
        job.counts = counts
        db.commit()
        # Filtering application-side is portable across SQLite and PostgreSQL JSON implementations.
        all_documents = [
            item
            for item in db.scalars(select(Document).where(Document.knowledge_base_id == job.knowledge_base_id))
            if item.metadata_json.get("import_job_id") == job.id
        ]
        documents = sorted(
            (item for item in all_documents if item.status != DocumentStatus.indexed),
            key=document_processing_priority,
        )
        already_indexed = len(all_documents) - len(documents)
        if already_indexed:
            append_job_log(
                job,
                "resume",
                f"断点恢复：跳过 {already_indexed} 篇已经完成索引的文献，不重复解析或调用模型",
            )
            db.commit()
        total = max(1, len(documents))
        successful = 0
        for index, document in enumerate(documents, 1):
            db.expire_all()
            job = db.get(ImportJob, job_id)
            if not job or apply_control_checkpoint(db, job) != ACTIVE:
                return
            db.refresh(document)
            if (document.metadata_json or {}).get("import_job_id") != job.id:
                append_job_log(job, "reprioritized", f"文献已转交其他优先任务，当前批次跳过：{document.title}", document.id)
                db.commit()
                continue
            update_current_document(
                job,
                document,
                position=index,
                total=total,
                stage="等待处理当前文献",
                document_progress=0.0,
            )
            db.commit()
            successful += int(await process_document(db, job, document, position=index, total=total))
            db.expire_all()
            job = db.get(ImportJob, job_id)
            if not job or apply_control_checkpoint(db, job, allow_pause=index < len(documents)) != ACTIVE:
                return
        db.expire_all()
        job = db.get(ImportJob, job_id)
        if not job or apply_control_checkpoint(db, job, allow_pause=False) != ACTIVE:
            return
        all_documents = [
            item
            for item in db.scalars(select(Document).where(Document.knowledge_base_id == job.knowledge_base_id))
            if item.metadata_json.get("import_job_id") == job.id
        ]
        counts = dict(job.counts or {})
        counts["selected"] = len(all_documents)
        counts["indexed"] = sum(item.status == DocumentStatus.indexed for item in all_documents)
        counts["failed"] = sum(item.status == DocumentStatus.failed for item in all_documents)
        counts["metadata_only"] = sum(
            item.status == DocumentStatus.indexed and bool((item.metadata_json or {}).get("metadata_only"))
            for item in all_documents
        )
        counts["fulltext"] = sum(
            item.status == DocumentStatus.indexed and not bool((item.metadata_json or {}).get("metadata_only"))
            for item in all_documents
        )
        current = dict(counts.get("current_document") or {})
        current.update({"state": "job_completed", "stage": "任务全部完成", "document_progress": 1.0})
        counts["current_document"] = current
        counts["execution_state"] = "completed"
        job.counts = counts
        kb = db.get(KnowledgeBase, job.knowledge_base_id)
        if kb and successful:
            kb.index_version += 1
        job.status = JobStatus.completed
        job.stage = "完成"
        job.progress = 1.0
        job.completed_at = datetime.now(timezone.utc)
        append_job_log(
            job,
            "completed",
            f"任务结束：本次处理成功 {successful} 篇，累计完成 {counts['indexed']} 篇",
        )
        db.commit()
    except Exception as exc:
        db.rollback()
        job = db.get(ImportJob, job_id)
        if job:
            job.status = JobStatus.failed
            job.stage = "任务失败"
            job.error_message = f"{type(exc).__name__}: {exc}"
            counts = dict(job.counts or {})
            counts["execution_state"] = "failed"
            job.counts = counts
            append_job_log(job, "failed", job.error_message)
            db.commit()
    finally:
        db.close()


def run_import_job(job_id: str) -> None:
    asyncio.run(run_import_job_async(job_id))
