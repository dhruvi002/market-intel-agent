"""ARQ task: run the multi-agent LangGraph pipeline for a query session.

This module is the bridge between the FastAPI gateway (which enqueues the job)
and the LangGraph agent graph (which does the work).

Architecture
------------
1. ``run_query(ctx, session_id, query, tickers)`` is enqueued by the API
   immediately after the user submits a query.
2. The task builds a ``Retriever`` and ``build_graph()`` with an
   ``event_callback`` that publishes ``AgentEvent`` JSON to the Redis pubsub
   channel ``events:{session_id}``.
3. The browser's WebSocket subscription to that channel receives the events
   in real-time.
4. After the graph completes (or on error), the task emits SESSION_DONE / ERROR
   and updates the session status in Redis.

Event emission
--------------
The ``event_callback`` is an async function that serialises an ``AgentEvent``
with Pydantic's ``model_dump_json()`` and publishes it to Redis via
``redis.publish(channel, json_str)``.  The WebSocket endpoint subscribes to
the same channel and forwards the raw JSON string to the browser.

Langfuse tracing (Phase 9)
--------------------------
``observe_run`` wraps the full graph invocation in a top-level Langfuse trace,
and ``langchain_callbacks()`` attaches a ``CallbackHandler`` to LangChain's
config so every nested LLM call becomes a child span.  Both degrade to no-ops
when Langfuse keys are absent — a fresh clone runs unchanged.

Retriever initialisation
------------------------
The Retriever is built once per ARQ worker task rather than per-worker-process
because the embedder and reranker models are large; building them in the task
function means the first query per worker process will load the models from
disk (~2–5 seconds) while subsequent queries in the same process reuse the
cached singletons via ``@lru_cache`` in the retrieval package.
"""

from __future__ import annotations

import logging
from uuid import UUID

import redis.asyncio as aioredis
import structlog

from mia_shared.config import get_settings
from mia_shared.schemas import AgentEvent, AgentState, EventType
from mia_shared.tracing import langchain_callbacks, observe_run

from ..session_store import events_channel, update_session_status

logger = structlog.get_logger(__name__)

# Module-level lazy retriever cache — one per process, reused across tasks
_retriever = None


def _get_retriever():
    """Lazy singleton: build the Retriever once per worker process."""
    global _retriever  # noqa: PLW0603
    if _retriever is None:
        from pathlib import Path  # noqa: PLC0415

        from mia_retrieval.retriever import build_retriever  # noqa: PLC0415

        settings = get_settings()
        _retriever = build_retriever(
            qdrant_url=settings.qdrant_url,
            qdrant_collection=settings.qdrant_collection,
            bm25_path=Path("data/bm25_index.pkl"),
        )
        logging.getLogger(__name__).info("Retriever initialised")
    return _retriever


async def run_query(
    ctx: dict,
    session_id: str,
    query: str,
    tickers: list[str],
) -> None:
    """ARQ task: run the multi-agent graph for *query* and stream events.

    Parameters
    ----------
    ctx        : ARQ task context (provides ``ctx["redis"]`` — an ArqRedis)
    session_id : UUID string identifying the session
    query      : natural-language user query
    tickers    : optional ticker hints (may be empty list)
    """
    settings = get_settings()
    sid = UUID(session_id)
    channel = events_channel(sid)
    log = logger.bind(session_id=session_id)

    # Open a pubsub-capable Redis client for event publishing
    pub_redis = aioredis.from_url(settings.redis_url, decode_responses=True)

    async def _emit(event: AgentEvent) -> None:
        """Publish one AgentEvent as JSON to the session's Redis channel."""
        try:
            await pub_redis.publish(channel, event.model_dump_json())
        except Exception as exc:  # noqa: BLE001
            log.warning("Event publish failed", exc=str(exc))

    try:
        await update_session_status(pub_redis, sid, "running")
        log.info("run_query: started", query=query[:80])

        # ── Build graph ────────────────────────────────────────────────────
        from mia_agents.graph import build_graph  # noqa: PLC0415
        from mia_agents.llm import get_llm  # noqa: PLC0415

        retriever = _get_retriever()
        llm = get_llm()
        graph = build_graph(
            retriever=retriever,
            llm=llm,
            event_callback=_emit,
        )

        # ── Run graph (wrapped in a top-level Langfuse trace) ─────────────
        initial_state = AgentState(
            session_id=sid,
            query=query,
        )
        async with observe_run(
            "run_query",
            session_id=session_id,
            metadata={"query": query[:200], "tickers": tickers},
        ):
            await graph.ainvoke(
                initial_state.model_dump(),
                config={"callbacks": langchain_callbacks()},
            )

        # ── Emit terminal event ────────────────────────────────────────────
        await _emit(
            AgentEvent(
                session_id=sid,
                event_type=EventType.SESSION_DONE,
                payload={},
            )
        )
        await update_session_status(pub_redis, sid, "done")
        log.info("run_query: completed")

    except Exception as exc:
        log.error("run_query: error", exc=str(exc))
        await _emit(
            AgentEvent(
                session_id=sid,
                event_type=EventType.ERROR,
                payload={"message": str(exc)},
            )
        )
        await update_session_status(pub_redis, sid, "error")

    finally:
        await pub_redis.aclose()
