from __future__ import annotations

import os

from redis import Redis
from rq import Queue, Worker

from app.config import get_settings
from app.db import engine, init_db
from app.services.recovery import recover_interrupted_tasks


def main() -> None:
    settings = get_settings()
    # The API normally creates backward-compatible tables first, but Workers may
    # start at nearly the same time after a Compose upgrade.
    init_db()
    connection = Redis.from_url(settings.redis_url)
    queue_names = [
        item.strip()
        for item in os.getenv("WORKER_QUEUE_NAMES", settings.queue_name).split(",")
        if item.strip()
    ]
    queues = [Queue(name, connection=connection) for name in queue_names]
    should_recover = os.getenv("WORKER_RECOVER_TASKS", "true").strip().casefold() in {"1", "true", "yes"}
    recovered = recover_interrupted_tasks(connection, Queue(settings.queue_name, connection=connection)) if should_recover else {
        "imports": 0,
        "automations": 0,
    }
    # Recovery queries run in the long-lived worker parent. Never let an open or
    # pooled PostgreSQL connection cross RQ's subsequent fork boundary.
    engine.dispose()
    if recovered["imports"] or recovered["automations"]:
        print(f"Recovered interrupted work: {recovered}", flush=True)
    worker = Worker(queues, connection=connection)
    worker.work(with_scheduler=True)


if __name__ == "__main__":
    main()
