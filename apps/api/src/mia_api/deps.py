"""FastAPI dependency providers for the MIA API.

All heavy resources (Redis, settings) are created once per process via
module-level lazy singletons and injected into route handlers via FastAPI's
``Depends`` mechanism.

The Redis client is NOT a singleton in the traditional sense: ``get_redis()``
returns a new client from the connection pool each time, which is the
recommended pattern for ``redis.asyncio``.  The pool itself is shared.
"""

from __future__ import annotations

from functools import lru_cache
from typing import AsyncIterator

import redis.asyncio as aioredis
from fastapi import Depends

from mia_shared.config import Settings, get_settings


# ── Settings ──────────────────────────────────────────────────────────────────

def get_api_settings() -> Settings:
    """Re-export the shared settings singleton for FastAPI DI."""
    return get_settings()


# ── Redis pool ────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_redis_pool(redis_url: str) -> aioredis.Redis:
    """Create (once) the shared async Redis client from *redis_url*."""
    return aioredis.from_url(redis_url, decode_responses=True)


def get_redis(
    settings: Settings = Depends(get_api_settings),
) -> aioredis.Redis:
    """Return the shared Redis client (connection pool backed)."""
    return _get_redis_pool(settings.redis_url)


# ── ARQ pool factory ──────────────────────────────────────────────────────────

async def get_arq_pool(
    settings: Settings = Depends(get_api_settings),
) -> AsyncIterator[aioredis.Redis]:
    """Yield a one-off ARQ Redis pool for enqueueing jobs.

    ARQ uses the same Redis instance as pubsub but through its own pool.
    We create a fresh pool per request to avoid ARQ-internal state leakage.
    """
    import urllib.parse  # noqa: PLC0415

    from arq.connections import ArqRedis, RedisSettings  # noqa: PLC0415

    parsed = urllib.parse.urlparse(settings.redis_url)
    arq_settings = RedisSettings(
        host=parsed.hostname or "localhost",
        port=parsed.port or 6379,
        password=parsed.password or None,
        database=int(parsed.path.lstrip("/") or "0"),
    )
    pool: ArqRedis = await aioredis.Redis(
        host=arq_settings.host,  # type: ignore[arg-type]
        port=arq_settings.port,
        password=arq_settings.password,
        db=arq_settings.database,
        decode_responses=False,  # ARQ needs bytes
    )
    try:
        yield pool
    finally:
        await pool.aclose()
