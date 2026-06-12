"""Summarizer node — synthesises accumulated evidence into a cited draft.

Design decisions:
- The summarizer is intentionally separate from the RAGAgent: by the time this
  node runs, evidence may have been gathered by multiple workers across several
  iterations, so we only need generation (no retrieval).
- Evidence is capped at ``settings.max_evidence_chunks`` before formatting to
  avoid exceeding the LLM's context window.  The chunks are passed in
  descending relevance-score order so the most relevant evidence is not
  truncated away.
- The draft is written to ``state.draft``; the Critic node reads it next.
"""

from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel

from mia_agents.rag_agent import _format_context
from mia_shared.config import get_settings
from mia_shared.schemas import AgentState

logger = logging.getLogger(__name__)


async def summarizer_node(state: AgentState, *, llm: BaseChatModel) -> dict:
    """Synthesise all accumulated evidence into a structured draft answer.

    Parameters
    ----------
    state : current AgentState — reads ``query`` and ``evidence``
    llm   : LangChain chat model

    Returns
    -------
    dict
        Update for ``draft``.
    """
    from mia_agents.prompts import build_summarizer_messages  # noqa: PLC0415

    settings = get_settings()

    # Sort by relevance score descending; None scores go last
    ranked = sorted(
        state.evidence,
        key=lambda ev: ev.relevance_score or 0.0,
        reverse=True,
    )
    capped = ranked[: settings.max_evidence_chunks]

    logger.info(
        "summarizer: synthesising from %d evidence chunks (cap=%d)",
        len(capped),
        settings.max_evidence_chunks,
    )

    if not capped:
        logger.warning("summarizer: no evidence — returning fallback draft")
        return {
            "draft": (
                "Insufficient evidence was retrieved to answer this question. "
                "Please try a different query or check that relevant filings have been ingested."
            )
        }

    context = _format_context(capped)
    messages = build_summarizer_messages(query=state.query, context=context)
    response = await llm.ainvoke(messages)
    draft: str = response.content  # type: ignore[attr-defined]

    logger.info("summarizer: draft produced (%d chars)", len(draft))
    return {"draft": draft}
