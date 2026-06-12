"""Critic node — fact-checks the draft against retrieved evidence.

Phase 4 implementation: LLM-based verification using ``with_structured_output``.

Phase 5 will layer on NLI (cross-encoder/nli-deberta-v3-base) for per-sentence
entailment scoring, which upgrades each Citation's ``is_verified`` flag and
``nli_score``.  The LLM-based critic here is the gating logic that issues the
PASS / REVISE / ESCALATE verdict; the NLI model will provide fine-grained
sentence-level support for the Critic's claims.

Design decisions:
- ``CritiqueResult`` is already defined in ``mia_shared.schemas`` — the critic
  returns exactly that schema, keeping the interface stable for Phase 5.
- ``iteration_count`` is incremented here (not in the supervisor) so the graph
  can enforce the cap inside the routing function without needing to track it
  across two nodes.
- ESCALATE is treated as a terminal verdict: the graph routes it to END without
  further retries, since more retrieval is unlikely to help.
"""

from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel

from mia_agents.rag_agent import _format_context
from mia_shared.schemas import (
    AgentState,
    CriticVerdict,
    CritiqueResult,
    FailingClaim,
)

logger = logging.getLogger(__name__)


# ── Structured-output contract ────────────────────────────────────────────────

class _FailingClaimOut(FailingClaim):
    """Re-export with no changes — used as the LLM output schema."""


class _CritiqueOut(CritiqueResult):
    """Re-export with no changes — used as the LLM output schema."""


# ── Node ──────────────────────────────────────────────────────────────────────

async def critic_node(state: AgentState, *, llm: BaseChatModel) -> dict:
    """Fact-check ``state.draft`` against ``state.evidence``.

    Parameters
    ----------
    state : current AgentState — reads ``query``, ``evidence``, ``draft``
    llm   : LangChain chat model

    Returns
    -------
    dict
        Updates for ``critique`` and ``iteration_count``.
    """
    from mia_agents.prompts import build_critic_messages  # noqa: PLC0415

    logger.info(
        "critic: verifying draft (%d chars) against %d evidence chunks",
        len(state.draft),
        len(state.evidence),
    )

    if not state.draft:
        logger.warning("critic: empty draft — issuing ESCALATE verdict")
        result = CritiqueResult(
            verdict=CriticVerdict.ESCALATE,
            summary="No draft was produced by the Summarizer.",
        )
        return {"critique": result, "iteration_count": state.iteration_count + 1}

    structured_llm = llm.with_structured_output(_CritiqueOut)
    context = _format_context(state.evidence)
    messages = build_critic_messages(
        query=state.query,
        context=context,
        draft=state.draft,
    )

    try:
        raw: _CritiqueOut = await structured_llm.ainvoke(messages)
        critique = CritiqueResult(
            verdict=raw.verdict,
            failing_claims=raw.failing_claims,
            summary=raw.summary,
        )
    except Exception as exc:  # noqa: BLE001
        # Structured-output parsing can fail if the LLM returns malformed JSON;
        # default to REVISE so the graph retries rather than surfaces an error.
        logger.warning("critic: structured-output parse failed (%s) — defaulting to REVISE", exc)
        critique = CritiqueResult(
            verdict=CriticVerdict.REVISE,
            summary=f"Critic could not parse LLM output: {exc}",
        )

    logger.info(
        "critic: verdict=%s  failing_claims=%d",
        critique.verdict.value,
        len(critique.failing_claims),
    )

    return {
        "critique": critique,
        "iteration_count": state.iteration_count + 1,
    }
