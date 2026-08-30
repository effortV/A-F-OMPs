from __future__ import annotations

from app.config import get_settings


def enqueue_import(job_id: str, *, high_priority: bool = False) -> bool:
    settings = get_settings()
    if not settings.use_rq:
        return False
    from redis import Redis
    from rq import Queue

    queue_name = settings.priority_queue_name if high_priority else settings.queue_name
    queue = Queue(queue_name, connection=Redis.from_url(settings.redis_url))
    queue.enqueue(
        "app.services.pipeline.run_import_job",
        job_id,
        job_timeout="12h",
        result_ttl=86400,
        failure_ttl=604800,
    )
    return True
