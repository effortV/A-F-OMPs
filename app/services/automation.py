from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from app.config import get_settings
from app.db import SessionLocal
from app.models import AutomationStatus, ImportJob, JobStatus, LiteratureAutomation
from app.services.discovery_service import DiscoveryService
from app.services.import_service import create_documents_from_candidates
from app.services.pipeline import append_job_log, run_import_job_async
from app.services.task_control import automation_is_deleted


def enqueue_automation_cycle(automation_id: str, *, delay_minutes: int = 0) -> str:
    settings = get_settings()
    if not settings.use_rq:
        raise RuntimeError("持续自动采集需要 Redis/RQ；请使用完整 Docker 模式启动")
    from redis import Redis
    from rq import Queue

    queue = Queue(settings.queue_name, connection=Redis.from_url(settings.redis_url))
    if delay_minutes > 0:
        rq_job = queue.enqueue_in(
            timedelta(minutes=delay_minutes),
            "app.services.automation.run_automation_cycle",
            automation_id,
            job_timeout="24h",
            result_ttl=86400,
            failure_ttl=604800,
        )
    else:
        rq_job = queue.enqueue(
            "app.services.automation.run_automation_cycle",
            automation_id,
            job_timeout="24h",
            result_ttl=86400,
            failure_ttl=604800,
        )
    return rq_job.id


def enqueue_automation_resume_cycle(automation_id: str) -> str:
    settings = get_settings()
    if not settings.use_rq:
        raise RuntimeError("持续自动采集需要 Redis/RQ；请使用完整 Docker 模式启动")
    from redis import Redis
    from rq import Queue

    queue = Queue(settings.queue_name, connection=Redis.from_url(settings.redis_url))
    rq_job = queue.enqueue(
        "app.services.automation.resume_automation_cycle",
        automation_id,
        job_timeout="24h",
        result_ttl=86400,
        failure_ttl=604800,
    )
    return rq_job.id


def _account_finished_job(automation: LiteratureAutomation, job: ImportJob | None) -> None:
    """Count an automation import exactly once, including after a worker restart."""
    if not job or job.status != JobStatus.completed:
        return
    counts = dict(job.counts or {})
    if counts.get("automation_accounted"):
        return
    automation.imported_total += int(counts.get("indexed", 0))
    counts["automation_accounted"] = True
    job.counts = counts


def _finish_or_reschedule(db, automation: LiteratureAutomation) -> None:
    reached_limit = automation.max_total is not None and automation.imported_total >= automation.max_total
    if automation.stop_requested or automation_is_deleted(automation) or reached_limit:
        automation.status = AutomationStatus.stopped
        automation.next_run_at = None
        if reached_limit:
            automation.error_message = "已达到自动采集总量上限"
    else:
        automation.status = AutomationStatus.active
        automation.next_run_at = datetime.now(timezone.utc) + timedelta(minutes=automation.interval_minutes)
        automation.rq_job_id = enqueue_automation_cycle(
            automation.id,
            delay_minutes=automation.interval_minutes,
        )
    db.commit()


async def run_automation_cycle_async(automation_id: str) -> None:
    db = SessionLocal()
    automation: LiteratureAutomation | None = None
    try:
        automation = db.get(LiteratureAutomation, automation_id)
        if (
            not automation
            or automation.stop_requested
            or automation_is_deleted(automation)
            or automation.status == AutomationStatus.stopped
        ):
            if automation:
                automation.status = AutomationStatus.stopped
                automation.next_run_at = None
                db.commit()
            return
        automation.status = AutomationStatus.running
        automation.last_run_at = datetime.now(timezone.utc)
        automation.next_run_at = None
        automation.cycles += 1
        automation.error_message = None
        job = ImportJob(
            conversation_id=automation.conversation_id,
            knowledge_base_id=automation.knowledge_base_id,
            query=automation.query,
            status=JobStatus.queued,
            stage=f"自动采集第 {automation.cycles} 轮",
        )
        db.add(job)
        db.flush()
        automation.last_job_id = job.id
        db.commit()

        await DiscoveryService().discover(
            db,
            job,
            limit=min(500, max(150, automation.batch_size * 3)),
            year_from=None,
            year_to=None,
            include_citation_expansion=True,
        )
        db.refresh(automation)
        if automation.stop_requested or automation_is_deleted(automation):
            job.status = JobStatus.completed
            job.stage = "自动采集已由用户停止"
            job.progress = 1.0
            job.completed_at = datetime.now(timezone.utc)
            append_job_log(job, "stopped", "检索完成后收到停止/删除请求，本轮未继续导入")
            db.commit()
            _finish_or_reschedule(db, automation)
            return
        remaining = automation.batch_size
        if automation.max_total is not None:
            remaining = min(remaining, max(0, automation.max_total - automation.imported_total))
        document_ids = create_documents_from_candidates(db, job, top_n=remaining)
        if document_ids:
            db.commit()
            await run_import_job_async(job.id)
            db.expire_all()
            finished_job = db.get(ImportJob, job.id)
            _account_finished_job(automation, finished_job)
        else:
            job.status = JobStatus.completed
            job.stage = "本轮没有去重后的新文献"
            job.progress = 1.0
            job.completed_at = datetime.now(timezone.utc)
            append_job_log(job, "completed", "本轮没有去重后的新文献，将按计划继续检查")
        db.commit()

        _finish_or_reschedule(db, automation)
    except Exception as exc:
        db.rollback()
        automation = db.get(LiteratureAutomation, automation_id)
        if automation:
            automation.status = AutomationStatus.failed
            automation.next_run_at = None
            automation.error_message = f"{type(exc).__name__}: {str(exc)[:500]}"
            db.commit()
    finally:
        db.close()


def run_automation_cycle(automation_id: str) -> None:
    asyncio.run(run_automation_cycle_async(automation_id))


async def resume_automation_cycle_async(automation_id: str) -> None:
    """Resume the last import and complete automation bookkeeping after an interrupted worker."""
    db = SessionLocal()
    try:
        automation = db.get(LiteratureAutomation, automation_id)
        if not automation or automation_is_deleted(automation):
            return
        if automation.stop_requested and not automation.last_job_id:
            automation.status = AutomationStatus.stopped
            automation.next_run_at = None
            db.commit()
            return
        last_job_id = automation.last_job_id
        last_job = db.get(ImportJob, last_job_id) if last_job_id else None
        if not last_job:
            automation.status = AutomationStatus.active
            automation.error_message = "恢复时未找到上一轮任务，已安排下一轮"
            _finish_or_reschedule(db, automation)
            return
        if last_job.status not in {JobStatus.completed, JobStatus.failed}:
            db.close()
            await run_import_job_async(last_job.id)
            db = SessionLocal()
            automation = db.get(LiteratureAutomation, automation_id)
            last_job = db.get(ImportJob, last_job_id)
        if not automation or not last_job:
            return
        if last_job.status == JobStatus.failed:
            automation.status = AutomationStatus.failed
            automation.next_run_at = None
            automation.error_message = f"中断任务恢复失败：{last_job.error_message or last_job.stage}"
            db.commit()
            return
        _account_finished_job(automation, last_job)
        automation.error_message = None
        _finish_or_reschedule(db, automation)
    except Exception as exc:
        db.rollback()
        automation = db.get(LiteratureAutomation, automation_id)
        if automation:
            automation.status = AutomationStatus.failed
            automation.next_run_at = None
            automation.error_message = f"恢复失败：{type(exc).__name__}: {str(exc)[:500]}"
            db.commit()
    finally:
        db.close()


def resume_automation_cycle(automation_id: str) -> None:
    asyncio.run(resume_automation_cycle_async(automation_id))
