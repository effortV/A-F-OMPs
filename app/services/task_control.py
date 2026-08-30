from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import ImportJob, JobStatus, LiteratureAutomation, TaskControl


ACTIVE = "active"
PAUSE_REQUESTED = "pause_requested"
PAUSED = "paused"
CANCEL_REQUESTED = "cancel_requested"
CANCELLED = "cancelled"

PAUSE_STATES = {PAUSE_REQUESTED, PAUSED}
CANCEL_STATES = {CANCEL_REQUESTED, CANCELLED}
TERMINAL_CONTROL_STATES = {CANCELLED}


def get_job_control(db: Session, job_id: str, *, create: bool = False) -> TaskControl | None:
    control = db.scalar(
        select(TaskControl)
        .where(TaskControl.job_id == job_id)
        .execution_options(populate_existing=True)
    )
    if control is None and create:
        control = TaskControl(job_id=job_id)
        db.add(control)
        db.flush()
    return control


def request_job_control(
    db: Session,
    job: ImportJob,
    state: str,
    *,
    deleted: bool | None = None,
) -> TaskControl:
    control = get_job_control(db, job.id, create=True)
    assert control is not None
    control.state = state
    if deleted is not None:
        control.deleted = deleted
    control.requested_at = datetime.now(timezone.utc)
    return control


def clear_job_control(db: Session, job: ImportJob) -> TaskControl:
    control = get_job_control(db, job.id, create=True)
    assert control is not None
    control.state = ACTIVE
    control.requested_at = datetime.now(timezone.utc)
    return control


def normalize_interrupted_pause(job: ImportJob, control: TaskControl) -> bool:
    """Turn an in-flight pause request into a resumable pause after Worker restart."""

    if control.state != PAUSE_REQUESTED:
        return False
    control.state = PAUSED
    counts = dict(job.counts or {})
    counts["execution_state"] = "paused"
    job.counts = counts
    job.status = JobStatus.queued
    job.stage = "已暂停；点击开始/继续任务"
    return True


def import_rq_call_is_running(job: ImportJob) -> bool:
    """Best-effort duplicate guard for the explicit Start button."""

    settings = get_settings()
    execution_state = str((job.counts or {}).get("execution_state") or "")
    if not settings.use_rq:
        return execution_state in {"running", "pausing", "cancelling"}
    rq_id = str((job.counts or {}).get("rq_job_id") or "").strip()
    if not rq_id:
        return execution_state == "running"
    try:
        from redis import Redis
        from rq.job import Job

        rq_job = Job.fetch(rq_id, connection=Redis.from_url(settings.redis_url))
        status = rq_job.get_status(refresh=True)
        value = getattr(status, "value", str(status)).casefold()
        return value == "started"
    except Exception:
        return execution_state == "running"


def automation_is_deleted(automation: LiteratureAutomation) -> bool:
    return bool((automation.settings_json or {}).get("deleted"))


def mark_automation_deleted(automation: LiteratureAutomation) -> None:
    payload = dict(automation.settings_json or {})
    payload.update(
        {
            "deleted": True,
            "deleted_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    automation.settings_json = payload


def _matching_rq_job(rq_job: Any, function_names: set[str], target_id: str) -> bool:
    args = rq_job.args or ()
    return bool(args) and rq_job.func_name in function_names and str(args[0]) == target_id


def _remove_from_queue(
    queue: Any,
    connection: Any,
    function_names: set[str],
    target_id: str,
) -> int:
    from rq.job import Job

    removed = 0
    for rq_id in list(queue.job_ids):
        try:
            rq_job = Job.fetch(rq_id, connection=connection)
            if not _matching_rq_job(rq_job, function_names, target_id):
                continue
            queue.remove(rq_id, delete_job=True)
            removed += 1
        except Exception:
            continue
    return removed


def _remove_from_scheduled_registry(
    queue: Any,
    connection: Any,
    function_names: set[str],
    target_id: str,
) -> int:
    from rq.job import Job
    from rq.registry import ScheduledJobRegistry

    registry = ScheduledJobRegistry(queue.name, connection=connection)
    removed = 0
    for rq_id in list(registry.get_job_ids()):
        try:
            rq_job = Job.fetch(rq_id, connection=connection)
            if not _matching_rq_job(rq_job, function_names, target_id):
                continue
            registry.remove(rq_id, delete_job=True)
            removed += 1
        except Exception:
            continue
    return removed


def remove_queued_calls(function_names: Iterable[str], target_id: str, *, scheduled: bool = False) -> int:
    settings = get_settings()
    if not settings.use_rq:
        return 0
    try:
        from redis import Redis
        from rq import Queue

        connection = Redis.from_url(settings.redis_url)
        names = list(dict.fromkeys([settings.queue_name, settings.priority_queue_name]))
        removed = 0
        for name in names:
            queue = Queue(name, connection=connection)
            removed += _remove_from_queue(queue, connection, set(function_names), target_id)
            if scheduled:
                removed += _remove_from_scheduled_registry(
                    queue,
                    connection,
                    set(function_names),
                    target_id,
                )
        return removed
    except Exception:
        # The durable database control remains authoritative even if Redis is
        # temporarily unavailable. A queued Worker will observe it on startup.
        return 0


def remove_queued_import_calls(job_id: str) -> int:
    return remove_queued_calls({"app.services.pipeline.run_import_job"}, job_id)


def remove_queued_automation_calls(automation_id: str) -> int:
    return remove_queued_calls(
        {
            "app.services.automation.run_automation_cycle",
            "app.services.automation.resume_automation_cycle",
        },
        automation_id,
        scheduled=True,
    )
