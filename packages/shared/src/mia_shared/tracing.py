"""Langfuse v2 observability helpers — shared across worker and scripts.

This module lives in ``mia-shared`` so the ARQ worker can attach Langfuse
traces without pulling in the heavy ``mia-eval`` package (ragas, pandas, etc.).
``mia_eval.tracing`` wraps these helpers directly.

Design goals
------------
- **Zero friction when unconfigured.** A fresh clone with no Langfuse keys
  must still ``make graph-run`` successfully.  Every helper degrades to a
  no-op when ``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY`` are absent.
- **One handler per process.**  ``get_langfuse_handler`` is ``lru_cache``'d
  so the HTTP client and background flush thread are singletons.
- **Additive, never crash.** ``observe_run`` swallows trace errors and always
  flushes, so a Langfuse outage never surfaces to the user.
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
    flush thread, so one per process is correct.  Returns ``None`` when keys
    are absent or the ``langfuse`` package is not installed — callers can
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
        secret_key=cfg.langfuse_secret_key.get_secret_value(),  # type: ignore[union-attr]
        host=cfg.langfuse_host,
    )
    logger.info("Langfuse tracing enabled → %s", cfg.langfuse_host)
    return handler


def langchain_callbacks() -> list[Any]:
    """Convenience: ``[handler]`` if enabled else ``[]`` for LangChain config."""
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
        secret_key=cfg.langfuse_secret_key.get_secret_value(),  # type: ignore[union-attr]
        host=cfg.langfuse_host,
    )


@asynccontextmanager
async def observe_run(
    name: str, **trace_kwargs: Any
) -> AsyncIterator[Any | None]:
    """Open a top-level Langfuse trace around a whole agent run.

    Yields the trace object (or ``None`` when disabled).  Always flushes on
    exit so short-lived CLI / ARQ processes don't drop spans.

    Example
    -------
    >>> async with observe_run("query", session_id=str(sid)) as trace:
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
