"""Critic node — fact-checks the draft by scoring each citation with NLI.

Phase 5 implementation: replaces the Phase-4 LLM-based critic with a local
NLI cross-encoder (``cross-encoder/nli-deberta-v3-base``).

For each ``Citation`` in ``state.citations``:

1. Look up the corresponding ``Evidence`` chunk by ``evidence_id``.
2. Run ``nli.score_pairs([(evidence.text, citation.claim_text)])`` to get
   the entailment probability.
3. Set ``citation.is_verified = (score >= settings.nli_entailment_threshold)``
   and ``citation.nli_score = score``.

Verdict derivation
------------------
- **ESCALATE** — draft is empty, OR no evidence and no citations exist.
- **PASS**     — all scored citations are verified (or there are no citations
                 but evidence + draft are present).
- **REVISE**   — at least one citation failed entailment.

Citations whose ``evidence_id`` does not match any entry in ``state.evidence``
(an edge case during multi-iteration runs) are left with ``is_verified=False``
and ``nli_score=None`` and are counted as failing.

The ``llm`` parameter is accepted but unused — it is kept for API
compatibility with ``graph.py``'s closure pattern and may be removed in
a future phase.

Design decisions
----------------
- ``nli.score_pairs`` is synchronous and CPU-bound; it is dispatched via
  ``asyncio.to_thread`` to avoid blocking the event loop.
- All N citation pairs are batched into a single ``score_pairs`` call so
  the NLI model only runs once per critic invocation.
- ``Citation`` objects are *replaced* (new instances with updated fields)
  rather than mutated in-place, keeping Pydantic immutability intact.
"""

from __future__ import annotations

import asyncio
import logging
from uuid import UUID

from langchain_core.language_models import BaseChatModel

from mia_agents import nli
from mia_shared.config import get_settings
from mia_shared.schemas import (
    AgentState,
    Citation,
    CriticVerdict,
    CritiqueResult,
    Evidence,
    FailingClaim,
)

logger = logging.getLogger(__name__)


# ── Node ──────────────────────────────────────────────────────────────────────

async def critic_node(
    state: AgentState,
    *,
    llm: BaseChatModel | None = None,  # accepted but unused in Phase 5
) -> dict:
    """NLI-based fact-checker: scores each citation and derives a verdict.

    Parameters
    ----------
    state : current AgentState — reads ``draft``, ``evidence``, ``citations``
    llm   : unused (kept for API compatibility with graph.py closures)

    Returns
    -------
    dict
        Updates for ``citations`` (with ``is_verified`` / ``nli_score``),
        ``critique``, and ``iteration_count``.
    """
    settings = get_settings()

    # ── Guard: empty draft ────────────────────────────────────────────────────
    if not state.draft:
        logger.warning("critic: empty draft — issuing ESCALATE verdict")
        return {
            "citations": state.citations,
            "critique": CritiqueResult(
                verdict=CriticVerdict.ESCALATE,
                summary="No draft was produced by the Summarizer.",
            ),
            "iteration_count": state.iteration_count + 1,
        }

    # ── Guard: no evidence at all ─────────────────────────────────────────────
    if not state.evidence and not state.citations:
        logger.warning("critic: no evidence and no citations — issuing ESCALATE")
        return {
            "citations": state.citations,
            "critique": CritiqueResult(
                verdict=CriticVerdict.ESCALATE,
                summary="No evidence was retrieved to verify claims against.",
            ),
            "iteration_count": state.iteration_count + 1,
        }

    # ── Fast-path: evidence exists but no citations (e.g. web_search results) ─
    if not state.citations:
        logger.info(
            "critic: no citations to score — evidence present, issuing PASS"
        )
        return {
            "citations": [],
            "critique": CritiqueResult(
                verdict=CriticVerdict.PASS,
                summary="No citations to verify; evidence is present.",
            ),
            "iteration_count": state.iteration_count + 1,
        }

    # ── Build evidence lookup map ─────────────────────────────────────────────
    ev_map: dict[UUID, Evidence] = {ev.id: ev for ev in state.evidence}

    # Separate citations into scorable (evidence found) and orphaned (not found)
    scorable: list[Citation] = []
    orphaned: list[Citation] = []
    for cit in state.citations:
        if cit.evidence_id in ev_map:
            scorable.append(cit)
        else:
            logger.warning(
                "critic: citation %s references missing evidence %s — treated as failing",
                cit.id,
                cit.evidence_id,
            )
            orphaned.append(cit)

    # ── NLI scoring (batch, one thread dispatch) ──────────────────────────────
    pairs = [
        (ev_map[cit.evidence_id].text, cit.claim_text)
        for cit in scorable
    ]

    logger.info(
        "critic: scoring %d citation(s) via NLI (threshold=%.2f)",
        len(pairs),
        settings.nli_entailment_threshold,
    )

    scores: list[float] = await asyncio.to_thread(
        nli.score_pairs,
        pairs,
        settings.nli_model,
    )

    # ── Build updated Citation objects ────────────────────────────────────────
    updated_scorable: list[Citation] = []
    for cit, score in zip(scorable, scores):
        updated_scorable.append(
            Citation(
                id=cit.id,
                evidence_id=cit.evidence_id,
                claim_text=cit.claim_text,
                is_verified=score >= settings.nli_entailment_threshold,
                nli_score=score,
            )
        )

    # Orphaned citations remain unverified with no NLI score
    updated_orphaned: list[Citation] = [
        Citation(
            id=cit.id,
            evidence_id=cit.evidence_id,
            claim_text=cit.claim_text,
            is_verified=False,
            nli_score=None,
        )
        for cit in orphaned
    ]

    all_updated = updated_scorable + updated_orphaned

    # ── Verdict derivation ────────────────────────────────────────────────────
    failing = [c for c in all_updated if not c.is_verified]

    if not failing:
        logger.info("critic: all %d citation(s) verified — PASS", len(all_updated))
        verdict = CriticVerdict.PASS
        failing_claims: list[FailingClaim] = []
        summary = f"All {len(all_updated)} citation(s) passed NLI entailment check."
    else:
        verdict = CriticVerdict.REVISE
        failing_claims = [
            FailingClaim(
                claim=c.claim_text,
                reason=(
                    f"NLI entailment score {c.nli_score:.3f} < threshold "
                    f"{settings.nli_entailment_threshold:.2f}"
                    if c.nli_score is not None
                    else "Evidence chunk not found in state"
                ),
            )
            for c in failing
        ]
        summary = (
            f"{len(failing)}/{len(all_updated)} citation(s) failed "
            f"NLI entailment check."
        )
        logger.info("critic: %d failing citation(s) — REVISE", len(failing))

    critique = CritiqueResult(
        verdict=verdict,
        failing_claims=failing_claims,
        summary=summary,
    )

    return {
        "citations": all_updated,
        "critique": critique,
        "iteration_count": state.iteration_count + 1,
    }
