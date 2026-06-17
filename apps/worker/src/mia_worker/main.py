"""ARQ worker entry point.

Start the worker:
    uv run python -m mia_worker.main

Or via Makefile:
    make worker

Phase 6: added ``run_query`` task (streams the multi-agent LangGraph pipeline
events to Redis pubsub for WebSocket forwarding to the browser).
"""

from __future__ import annotations

import logging
import urllib.parse

import structlog
from arq.connections import RedisSettings

from mia_shared.config import get_settings
from mia_worker.tasks.ingest import ingest_filing, ingest_ticker
from mia_worker.tasks.query import run_query


def _redis_settings() -> RedisSettings:
    """Build ARQ RedisSettings from the DATABASE_URL-style redis_url."""
    settings = get_settings()
    parsed = urllib.parse.urlparse(settings.redis_url)
    return RedisSettings(
        host=parsed.hostname or "localhost",
        port=parsed.port or 6379,
        password=parsed.password or None,
        database=int(parsed.path.lstrip("/") or "0"),
    )


async def startup(ctx: dict) -> None:
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        logger_factory=structlog.PrintLoggerFactory(),
    )
    logger = structlog.get_logger(__name__)
    logger.info("Worker starting", queues=["default"])


async def shutdown(ctx: dict) -> None:
    structlog.get_logger(__name__).info("Worker shutting down")


class WorkerSettings:
    """ARQ worker configuration.

    ARQ reads this class's attributes at startup to configure the worker.
    """

    functions = [ingest_ticker, ingest_filing, run_query]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = _redis_settings()
    # Allow up to 10 concurrent jobs (all I/O-bound via asyncio)
    max_jobs = 10
    # Jobs time out after 30 minutes — generous for large ticker batches
    job_timeout = 1800


if __name__ == "__main__":
    from arq import run_worker

    run_worker(WorkerSettings)
