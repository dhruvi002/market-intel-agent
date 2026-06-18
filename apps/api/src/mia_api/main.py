"""MIA FastAPI gateway — Phase 6.

Endpoints
---------
POST /api/sessions
    Accept a ``QueryRequest``, create a session in Redis, enqueue the
    ``run_query`` ARQ task, return ``SessionResponse`` with the WebSocket URL.

GET  /ws/sessions/{session_id}/stream   (WebSocket)
    Subscribe to the Redis pubsub channel ``events:{session_id}`` and
    forward every ``AgentEvent`` JSON message to the connected WebSocket
    client.  Closes the connection on SESSION_DONE or ERROR.

GET  /api/sessions/{session_id}
    Return the current session metadata (status, query) from Redis.
    Useful for polling when WebSocket is unavailable.

GET  /api/health
    Liveness probe — returns ``{"status": "ok"}``.

Architecture
------------
- The FastAPI process does NOT run the agent graph directly.  It enqueues an
  ARQ task and immediately returns a ``session_id``.  The ARQ worker (a
  separate process) runs the graph and publishes ``AgentEvent`` objects to
  Redis pubsub.  The WebSocket endpoint subscribes to that channel and
  streams events to the browser.

- Redis pubsub uses the ``decode_responses=True`` client (events are JSON
  strings).  ARQ uses a ``decode_responses=False`` client (ARQ expects bytes).
  Both share the same Redis instance but via separate pools in ``deps.py``.

- CORS is configured for development (allow all origins).  In production,
  restrict ``allow_origins`` to the deployed frontend URL.
"""

from __future__ import annotations

import json
import logging
from uuid import UUID, uuid4

import redis.asyncio as aioredis
import structlog
from arq import create_pool
from arq.connections import RedisSettings
from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from mia_shared.config import Settings, get_settings
from mia_shared.schemas import EventType, QueryRequest, SessionResponse

from .deps import get_redis
from .session_store import create_session, events_channel, get_session

logger = structlog.get_logger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Market Intelligence Agent API",
    version="0.6.0",
    description="FastAPI gateway for the autonomous multi-agent market-intel system.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Lifecycle ─────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def _on_startup() -> None:
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        logger_factory=structlog.stdlib.LoggerFactory(),
    )
    logger.info("API started")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _arq_redis_settings(settings: Settings) -> RedisSettings:
    import urllib.parse

    parsed = urllib.parse.urlparse(settings.redis_url)
    return RedisSettings(
        host=parsed.hostname or "localhost",
        port=parsed.port or 6379,
        password=parsed.password or None,
        database=int(parsed.path.lstrip("/") or "0"),
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/api/sessions", response_model=SessionResponse)
async def create_session_endpoint(
    request: QueryRequest,
    redis: aioredis.Redis = Depends(get_redis),
    settings: Settings = Depends(get_settings),
) -> SessionResponse:
    """Create a new agent session and enqueue the query for processing.

    Returns the ``session_id`` and the WebSocket URL to connect to for
    streaming events.
    """
    session_id = uuid4()

    # Persist session metadata in Redis
    await create_session(
        redis=redis,
        session_id=session_id,
        query=request.query,
        tickers=request.tickers,
    )

    # Enqueue the ARQ task
    try:
        arq_pool = await create_pool(_arq_redis_settings(settings))
        await arq_pool.enqueue_job(
            "run_query",
            str(session_id),
            request.query,
            request.tickers,
        )
        await arq_pool.aclose()
    except Exception as exc:
        logger.error("Failed to enqueue run_query task", exc=str(exc))
        raise HTTPException(status_code=503, detail="Queue unavailable") from exc

    logger.info(
        "Session created",
        session_id=str(session_id),
        query=request.query[:80],
    )

    return SessionResponse(
        session_id=session_id,
        status="queued",
        ws_url=f"/ws/sessions/{session_id}/stream",
    )


@app.get("/api/sessions/{session_id}", response_model=dict)
async def get_session_endpoint(
    session_id: UUID,
    redis: aioredis.Redis = Depends(get_redis),
) -> dict:
    """Return session metadata (status, query) from Redis."""
    data = await get_session(redis, session_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": str(session_id), **data}


@app.websocket("/ws/sessions/{session_id}/stream")
async def ws_stream(
    websocket: WebSocket,
    session_id: UUID,
    settings: Settings = Depends(get_settings),
) -> None:
    """Stream agent events to the browser over WebSocket.

    Subscribes to Redis pubsub channel ``events:{session_id}`` and forwards
    every message to the client.  Closes automatically on SESSION_DONE, ERROR,
    or client disconnect.
    """
    await websocket.accept()

    # Use a fresh Redis connection for pubsub (separate from the shared pool
    # because pubsub connections are stateful and should not be reused)
    pubsub_redis = aioredis.from_url(settings.redis_url, decode_responses=True)
    pubsub = pubsub_redis.pubsub()
    channel = events_channel(session_id)

    try:
        await pubsub.subscribe(channel)
        logger.info("WebSocket subscribed", channel=channel)

        async for message in pubsub.listen():
            if message["type"] != "message":
                continue

            data: str = message["data"]
            await websocket.send_text(data)

            # Close connection after terminal events
            try:
                event = json.loads(data)
                if event.get("event_type") in (
                    EventType.SESSION_DONE.value,
                    EventType.ERROR.value,
                ):
                    break
            except (json.JSONDecodeError, KeyError):
                pass

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected", channel=channel)
    except Exception as exc:
        logger.warning("WebSocket error", exc=str(exc), channel=channel)
        try:
            await websocket.close(code=1011)
        except Exception:
            pass
    finally:
        await pubsub.unsubscribe(channel)
        await pubsub_redis.aclose()
        logger.info("WebSocket cleaned up", channel=channel)
