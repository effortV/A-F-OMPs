from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from sqlalchemy import select

from app.config import get_settings
from app.db import SessionLocal
from app.models import AutomationStatus, ImportJob, JobStatus, LiteratureAutomation
from app.services.pipeline import append_job_log
from app.services.task_control import (
    CANCEL_STATES,
    PAUSE_REQUESTED,
    PAUSE_STATES,
    automation_is_deleted,
    get_job_control,
    normalize_interrupted_pause,
)


RECOVERABLE_IMPORT_STATES = {
    JobStatus.queued,
    JobStatus.downloading,
    JobStatus.parsing,
    JobStatus.extracting,
    JobStatus.indexing,
}

RETRYABLE_DATABASE_ERROR_MARKERS = (
    "DuplicatePreparedStatement",
    "duplicate prepared statement",
    "PendingRollbackError",
)


def is_retryable_database_error(value: str | None) -> bool:
    text = value or ""
    return any(marker.casefold() in text.casefold() for marker in RETRYABLE_DATABASE_ERROR_MARKERS)


def is_priority_import(job: ImportJob) -> bool:
    if bool((job.counts or {}).get("priority")):
        return True
    query = (job.query or "").strip()
    return query.startswith(("用户上传：", "公开网址/DOI 导入：", "重新获取公开全文："))


def _queued_calls(queue: Any, connection: Any) -> tuple[set[str], set[str]]:
    from rq.job import Job
    from rq.registry import ScheduledJobRegistry

    import_ids: set[str] = set()
    automation_ids: set[str] = set()
    ids = [*queue.job_ids, *ScheduledJobRegistry(queue.name, connection=connection).get_job_ids()]
    for rq_id in ids:
        try:
            job = Job.fetch(rq_id, connection=connection)
        except Exception:
            continue
        args = job.args or ()
        if not args:
            continue
        if job.func_name == "app.services.pipeline.run_import_job":
            import_ids.add(str(args[0]))
        elif job.func_name in {
            "app.services.automation.run_automation_cycle",
            "app.services.automation.resume_automation_cycle",
        }:
            automation_ids.add(str(args[0]))
    return import_ids, automation_ids


def _remove_queued_import_calls(queue: Any, connection: Any, import_job_id: str) -> int:
    from rq.job import Job

    removed = 0
    for rq_id in list(queue.job_ids):
        try:
            rq_job = Job.fetch(rq_id, connection=connection)
        except Exception:
            continue
        args = rq_job.args or ()
        if rq_job.func_name != "app.services.pipeline.run_import_job" or not args:
            continue
        if str(args[0]) != import_job_id:
            continue
        try:
            queue.remove(rq_id, delete_job=True)
            removed += 1
        except Exception:
            continue
    return removed


def recover_interrupted_tasks(connection: Any, queue: Any) -> dict[str, int]:
    """Requeue durable database work that was abandoned when the sole worker stopped."""
    settings = get_settings()
    if not settings.use_rq:
        return {"imports": 0, "automations": 0}
    from rq import Queue

    queued_imports: set[str] = set()
    queued_imports_by_queue: dict[str, set[str]] = {}
    queued_automations: set[str] = set()
    queue_names = list(dict.fromkeys([settings.queue_name, settings.priority_queue_name]))
    for queue_name in queue_names:
        imports, automations_in_queue = _queued_calls(Queue(queue_name, connection=connection), connection)
        queued_imports_by_queue[queue_name] = imports
        queued_imports.update(imports)
        queued_automations.update(automations_in_queue)
    recovered_imports = 0
    recovered_automations = 0
    db = SessionLocal()
    try:
        automations = list(db.scalars(select(LiteratureAutomation)))
        automations = [
            item
            for item in automations
            if not automation_is_deleted(item)
            and (
                item.status in {AutomationStatus.active, AutomationStatus.running, AutomationStatus.stopping}
                or (item.status == AutomationStatus.failed and is_retryable_database_error(item.error_message))
            )
        ]
        automation_job_ids = {item.last_job_id for item in automations if item.last_job_id}
        for automation in automations:
            if automation.id in queued_automations:
                continue
            if automation.status == AutomationStatus.failed:
                last_job = db.get(ImportJob, automation.last_job_id) if automation.last_job_id else None
                if last_job and last_job.status == JobStatus.failed and is_retryable_database_error(
                    last_job.error_message
                ):
                    last_job.status = JobStatus.queued
                    last_job.stage = "数据库连接修复后恢复"
                    last_job.error_message = None
                    automation.status = AutomationStatus.running
                else:
                    if last_job and last_job.status not in {JobStatus.completed, JobStatus.failed}:
                        last_job.status = JobStatus.failed
                        last_job.stage = "数据库连接中断，自动任务将重新检索"
                        last_job.error_message = "旧轮次在数据库连接异常时中断，已安排新轮次"
                    automation.status = AutomationStatus.active
                automation.error_message = None
            if automation.status in {AutomationStatus.running, AutomationStatus.stopping}:
                rq_job = queue.enqueue(
                    "app.services.automation.resume_automation_cycle",
                    automation.id,
                    job_id=f"recover-auto-{automation.id}-{uuid4().hex[:8]}",
                    job_timeout="24h",
                    result_ttl=86400,
                    failure_ttl=604800,
                )
                automation.rq_job_id = rq_job.id
            else:
                delay_seconds = 0
                if automation.next_run_at:
                    target = automation.next_run_at
                    if target.tzinfo is None:
                        target = target.replace(tzinfo=timezone.utc)
                    delay_seconds = max(0, int((target - datetime.now(timezone.utc)).total_seconds()))
                if delay_seconds:
                    rq_job = queue.enqueue_in(
                        timedelta(seconds=delay_seconds),
                        "app.services.automation.run_automation_cycle",
                        automation.id,
                        job_timeout="24h",
                        result_ttl=86400,
                        failure_ttl=604800,
                    )
                else:
                    rq_job = queue.enqueue(
                        "app.services.automation.run_automation_cycle",
                        automation.id,
                        job_timeout="24h",
                        result_ttl=86400,
                        failure_ttl=604800,
                    )
                automation.rq_job_id = rq_job.id
            recovered_automations += 1

        imports = list(db.scalars(select(ImportJob)))
        imports = [
            item
            for item in imports
            if item.status in RECOVERABLE_IMPORT_STATES
            or (item.status == JobStatus.failed and is_retryable_database_error(item.error_message))
        ]
        for import_job in imports:
            control = get_job_control(db, import_job.id)
            if control and control.state == PAUSE_REQUESTED:
                if normalize_interrupted_pause(import_job, control):
                    append_job_log(
                        import_job,
                        "paused",
                        "Worker 重启时任务正在安全暂停，现已转为可点击开始/继续的暂停状态",
                    )
                continue
            if control and (control.deleted or control.state in PAUSE_STATES or control.state in CANCEL_STATES):
                continue
            if import_job.id in automation_job_ids:
                continue
            priority = is_priority_import(import_job)
            if priority and import_job.id in queued_imports_by_queue.get(settings.priority_queue_name, set()):
                counts = dict(import_job.counts or {})
                counts.update({"execution_state": "queued", "worker_queue": settings.priority_queue_name})
                import_job.counts = counts
                continue
            if not priority and import_job.id in queued_imports:
                queue_name = next(
                    (
                        name
                        for name, import_ids in queued_imports_by_queue.items()
                        if import_job.id in import_ids
                    ),
                    settings.queue_name,
                )
                counts = dict(import_job.counts or {})
                counts.update({"execution_state": "queued", "worker_queue": queue_name})
                import_job.counts = counts
                continue
            if import_job.status == JobStatus.failed:
                import_job.status = JobStatus.queued
                import_job.stage = "数据库连接修复后恢复"
                import_job.error_message = None
            target_queue = Queue(
                settings.priority_queue_name if priority else settings.queue_name,
                connection=connection,
            )
            if priority:
                _remove_queued_import_calls(
                    Queue(settings.queue_name, connection=connection),
                    connection,
                    import_job.id,
                )
            target_queue.enqueue(
                "app.services.pipeline.run_import_job",
                import_job.id,
                at_front=True,
                job_id=f"recover-import-{import_job.id}-{uuid4().hex[:8]}",
                job_timeout="12h",
                result_ttl=86400,
                failure_ttl=604800,
            )
            counts = dict(import_job.counts or {})
            counts["execution_state"] = "queued"
            counts["worker_queue"] = target_queue.name
            import_job.counts = counts
            queue_label = "优先队列" if priority else "普通队列"
            append_job_log(import_job, "resume", f"Worker 重启后已自动恢复到{queue_label}；已清除旧数据库连接")
            recovered_imports += 1
        db.commit()
    finally:
        db.close()
    return {"imports": recovered_imports, "automations": recovered_automations}
