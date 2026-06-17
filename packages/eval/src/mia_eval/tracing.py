"""Langfuse v2 observability helpers.

Closes the Phase 3–4 deferred items ("Langfuse tracing in RAGAgent.run / graph
nodes once keys are configured").  The design goal is **zero friction when
unconfigured**: if no Langfuse keys are present in settings, every helper here
degrades to a no-op so the agent runs unchanged on a fresh clone — tracing is
strictly additive.

Two integration surfaces:

1. ``get_langfuse_handler()`` → a LangChain ``CallbackHandler`` (or ``None``).
   Pass it in ``config={"callbacks": [handler]}`` to any ``ainvoke`` /
   ``astream`` and Langfuse captures the full span tree, token counts and
   (synthetic) cost for that call.
2. ``observe_run(name, **trace_kwargs)`` → an async context manager that opens
   a single top-level trace around a whole query (so all nested LLM spans roll
   up under one session/trace id you can find in the Langfuse UI).

Langfuse is **v2** here (Postgres + Redis only) per the locked decision — v3's
ClickHouse requirement is too heavy for local dev.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Any, AsyncIterator

from mia_shared.config import get_settings

logger = logging.getLogger(__name__)


def langfuse_enabled() -> bool:
    """True iff both Langfuse keys are configured in settings."""
    cfg = get_settings()
    return bool(cfg.langfuse_public_key and cfg.langfuse_secret_key)


@lru_cache(maxsize=1)
def get_langfuse_handler() -> Any | None:
    """Return a cached LangChain ``CallbackHandler`` for Langfuse, or ``None``.

    Cached as a singleton: the handler holds an HTTP client and a background
    flush thread, so one per process is correct.  Returns ``None`` (no-op) when
    keys are absent or the ``langfuse`` package is not installed — callers can
    unconditionally do ``callbacks = [h] if h else []``.
    """
    if not langfuse_enabled():
        logger.debug("Langfuse disabled (no keys) — tracing is a no-op")
        return None

    cfg = get_settings()
    try:
        from langfuse.callback import CallbackHandler  # noqa: PLC0415
    except ImportError:  # pragma: no cover - optional dep
        logger.warning("langfuse not installed — tracing disabled")
        return None

    handler = CallbackHandler(
        public_key=cfg.langfuse_public_key,
        secret_key=cfg.langfuse_secret_key.get_secret_value(),
        host=cfg.langfuse_host,
    )
    logger.info("Langfuse tracing enabled → %s", cfg.langfuse_host)
    return handler


def langchain_callbacks() -> list[Any]:
    """Convenience: ``[handler]`` if enabled else ``[]`` for ``config=``."""
    handler = get_langfuse_handler()
    return [handler] if handler is not None else []


@lru_cache(maxsize=1)
def _get_client() -> Any | None:
    """Cached low-level Langfuse client (for top-level traces / scores)."""
    if not langfuse_enabled():
        return None
    cfg = get_settings()
    try:
        from langfuse import Langfuse  # noqa: PLC0415
    except ImportError:  # pragma: no cover - optional dep
        return None
    return Langfuse(
        public_key=cfg.langfuse_public_key,
        secret_key=cfg.langfuse_secret_key.get_secret_value(),
        host=cfg.langfuse_host,
    )


@asynccontextmanager
async def observe_run(
    name: str, **trace_kwargs: Any
) -> AsyncIterator[Any | None]:
    """Open a top-level Langfuse trace around a whole agent run.

    Yields the trace object (or ``None`` when disabled).  Always flushes on
    exit so short-lived CLI / eval processes don't drop spans.

    Example
    -------
    >>> async with observe_run("graph_run", session_id=str(sid)) as trace:
    ...     await graph.ainvoke(state, config={"callbacks": langchain_callbacks()})
    """
    client = _get_client()
    trace = None
    if client is not None:
        try:
            trace = client.trace(name=name, **trace_kwargs)
        except Exception:  # pragma: no cover - never break the run on tracing
            logger.exception("Langfuse trace() failed — continuing untraced")
            trace = None
    try:
        yield trace
    finally:
        if client is not None:
            try:
                client.flush()
            except Exception:  # pragma: no cover
                logger.exception("Langfuse flush() failed")
