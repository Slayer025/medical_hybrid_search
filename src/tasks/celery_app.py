"""Celery application configuration.

Uses Redis as broker and result backend, JSON serialization, and limits worker
concurrency to 2 to keep memory usage low on a 16GB RAM CPU-only machine.
"""

from __future__ import annotations

import os

from celery import Celery
from dotenv import load_dotenv


load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

app = Celery(
    "medical_hybrid_search",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["src.tasks.ingest_tasks"],
)

app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    worker_prefetch_multiplier=1,
    worker_concurrency=2,
    task_acks_late=True,
    result_expires=3600,
)

if __name__ == "__main__":
    app.start()
