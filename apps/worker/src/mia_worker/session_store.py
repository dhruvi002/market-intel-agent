"""Session state helpers for the ARQ worker.

Mirrors the key functions from apps/api/src/mia_api/session_store.py so the
worker does not depend on mia-api.  Both modules reference the same Redis keys
and pubsub channels — any change to key/channel naming must be reflected here.
"""

from __future__ import annotations

import json
from uuid import UUID

import redis.asyncio as aioredis

_SESSION_TTL_SECONDS = 7200

EVENTS_CHANNEL = "events:{session_id}"


def _session_key(session_id: UUID | str) -> str:
    return f"session:{session_id}"


async def update_session_status(
    redis: aioredis.Redis,
    session_id: UUID | str,
    status: str,
) -> None:
    """Update ``status`` field of an existing session.  Silent no-op if gone."""
    raw = await redis.get(_session_key(session_id))
    if raw is None:
        return
    data = json.loads(raw)
    data["status"] = status
    await redis.set(_session_key(session_id), json.dumps(data), keepttl=True)


def events_channel(session_id: UUID | str) -> str:
    """Return the Redis pubsub channel name for *session_id*."""
    return EVENTS_CHANNEL.format(session_id=session_id)
