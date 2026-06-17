"""Session state management in Redis.

Sessions are stored as JSON blobs in Redis with a 2-hour TTL.
Key format: ``session:{session_id}``
Pubsub channel: ``events:{session_id}``

This module is intentionally thin — it only manages the session lifecycle
(create, read, update status).  Event emission is handled by the ARQ worker
task (``tasks/query.py``) which publishes ``AgentEvent`` JSON to the pubsub
channel.  The WebSocket endpoint subscribes to that channel and forwards
messages to the connected client.
"""

from __future__ import annotations

import json
from uuid import UUID

import redis.asyncio as aioredis

# Session TTL: 2 hours — enough for any realistic query + frontend browsing time
_SESSION_TTL_SECONDS = 7200

# Pubsub channel name pattern
EVENTS_CHANNEL = "events:{session_id}"


def _session_key(session_id: UUID | str) -> str:
    return f"session:{session_id}"


async def create_session(
    redis: aioredis.Redis,
    session_id: UUID,
    query: str,
    tickers: list[str],
) -> None:
    """Store session metadata in Redis with a 2-hour TTL."""
    data = {
        "status": "queued",
        "query": query,
        "tickers": tickers,
    }
    await redis.set(
        _session_key(session_id),
        json.dumps(data),
        ex=_SESSION_TTL_SECONDS,
    )


async def get_session(
    redis: aioredis.Redis,
    session_id: UUID | str,
) -> dict | None:
    """Return session metadata dict, or ``None`` if not found / expired."""
    raw = await redis.get(_session_key(session_id))
    if raw is None:
        return None
    return json.loads(raw)


async def update_session_status(
    redis: aioredis.Redis,
    session_id: UUID | str,
    status: str,
) -> None:
    """Update the ``status`` field of an existing session.

    If the session key no longer exists (expired or never created), this is a
    silent no-op — the worker may finish after the TTL or client disconnect.
    """
    raw = await redis.get(_session_key(session_id))
    if raw is None:
        return
    data = json.loads(raw)
    data["status"] = status
    # Preserve the remaining TTL (use KEEPTTL requires Redis 6.0+)
    await redis.set(_session_key(session_id), json.dumps(data), keepttl=True)


def events_channel(session_id: UUID | str) -> str:
    """Return the Redis pubsub channel name for *session_id*."""
    return EVENTS_CHANNEL.format(session_id=session_id)
