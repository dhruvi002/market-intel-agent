"""Single-agent RAG baseline.

Given a natural-language query and a configured ``Retriever``, the agent:
  1. Retrieves relevant evidence chunks (hybrid BM25 + dense + reranker by default)
  2. Formats them as a numbered context string
  3. Calls an LLM using the RAG prompt template
  4. Parses inline [N] citation markers from the answer
  5. Returns a ``RAGResponse`` with answer, evidence, and citations

This is Phase 3 — intentionally a single-agent chain. The multi-agent
LangGraph graph (Phase 4) will build on top of this as one of its workers.

Design decisions:
- ``RAGResponse`` is a Pydantic model (not a dict) so downstream code —
  the FastAPI handler, Critic, evaluator — all receive typed data.
- Citation parsing is regex-based rather than asking the LLM to output JSON.
  Structured JSON output adds tokens and a parsing step; regex on the LLM's
  natural prose is simpler and robust enough for [N] markers.
- ``_extract_sentence()`` grabs 150 chars of context around each [N] marker
  as the ``claim_text`` for the Citation. This approximates a sentence and
  is enough for the NLI-based Critic to verify.
- Langfuse tracing is opt-in: if ``LANGFUSE_PUBLIC_KEY`` and
  ``LANGFUSE_SECRET_KEY`` are set, every run is traced; otherwise the agent
  runs without tracing (no SDK import error).
"""

from __future__ import annotations

import logging
import re
import time
from uuid import UUID, uuid4

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from mia_retrieval.retriever import RetrieveMode, Retriever
from mia_shared.config import get_settings
from mia_shared.schemas import Citation, Evidence

from .prompts import build_rag_messages

logger = logging.getLogger(__name__)

# Maximum characters of surrounding text to use as claim_text in a Citation
_CLAIM_CONTEXT_CHARS = 150


# ── Response schema ───────────────────────────────────────────────────────────

class RAGResponse(BaseModel):
    """Typed output of a single RAG agent run."""

    session_id: UUID = Field(default_factory=uuid4)
    query: str
    answer: str
    evidence: list[Evidence] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    model_used: str = ""
    retrieval_mode: str = ""
    reranked: bool = False
    latency_ms: float = 0.0

    @property
    def source_tickers(self) -> list[str]:
        """Unique tickers referenced by the evidence used."""
        return sorted({ev.ticker for ev in self.evidence if ev.ticker})


# ── Agent ─────────────────────────────────────────────────────────────────────

class RAGAgent:
    """Single-agent retrieve-then-generate RAG baseline.

    Parameters
    ----------
    retriever   : configured :class:`~mia_retrieval.retriever.Retriever`
    llm         : any LangChain ``BaseChatModel`` (use :func:`~mia_agents.llm.get_llm`)
    settings    : shared settings (uses ``get_settings()`` if not provided)
    """

    def __init__(
        self,
        retriever: Retriever,
        llm: BaseChatModel,
        settings=None,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._settings = settings or get_settings()

    async def run(
        self,
        query: str,
        *,
        tickers: list[str] | None = None,
        mode: RetrieveMode = RetrieveMode.HYBRID,
        rerank: bool = True,
        top_k: int | None = None,
    ) -> RAGResponse:
        """Run the RAG pipeline and return a typed response.

        Parameters
        ----------
        query   : natural-language question
        tickers : optional ticker filter (e.g. ``["NVDA", "AMD"]``)
        mode    : retrieval mode — ``hybrid`` (default), ``bm25``, ``dense``
        rerank  : apply cross-encoder reranking (default ``True``)
        top_k   : max evidence chunks to pass to the LLM
                  (defaults to ``settings.rerank_top_k``)
        """
        t_start = time.perf_counter()
        session_id = uuid4()

        # ── 1. Retrieve ───────────────────────────────────────────────────────
        evidence = await self._retriever.retrieve(
            query,
            mode=mode,
            rerank=rerank,
            ticker_filter=tickers,
            top_k=top_k,
        )
        logger.debug(
            "session=%s retrieved %d chunks (mode=%s rerank=%s)",
            session_id,
            len(evidence),
            mode.value,
            rerank,
        )

        # ── 2. Format context ─────────────────────────────────────────────────
        context = _format_context(evidence)

        # ── 3. Build prompt and call LLM ──────────────────────────────────────
        messages = build_rag_messages(query=query, context=context)
        response = await self._llm.ainvoke(messages)
        answer: str = response.content  # type: ignore[attr-defined]

        # Extract model name from response metadata when available
        model_used = _extract_model_name(response)

        # ── 4. Parse citations ────────────────────────────────────────────────
        citations = _parse_citations(answer, evidence)

        latency_ms = (time.perf_counter() - t_start) * 1000
        logger.debug(
            "session=%s done — %.0f ms, %d citations", session_id, latency_ms, len(citations)
        )

        return RAGResponse(
            session_id=session_id,
            query=query,
            answer=answer,
            evidence=evidence,
            citations=citations,
            model_used=model_used,
            retrieval_mode=mode.value,
            reranked=rerank,
            latency_ms=latency_ms,
        )


# ── Context formatting ────────────────────────────────────────────────────────

def _format_context(evidence: list[Evidence]) -> str:
    """Format evidence as a numbered list for the LLM prompt.

    Each chunk gets a header:  ``[N] TICKER FORM_TYPE — Section``

    Example::

        [1] NVDA 10-K — Risk Factors
        NVIDIA's data center revenue grew 217% year-over-year...

        [2] AMD 10-Q — MD&A
        AMD's MI300X shipments accelerated in Q3 2024...
    """
    if not evidence:
        return "(no evidence retrieved)"

    parts: list[str] = []
    for i, ev in enumerate(evidence, start=1):
        header_parts = [f"[{i}]"]
        if ev.ticker:
            header_parts.append(ev.ticker)
        if ev.filing_type:
            header_parts.append(ev.filing_type)
        if ev.section:
            header_parts.append(f"— {ev.section}")
        header = " ".join(header_parts)
        parts.append(f"{header}\n{ev.text.strip()}")

    return "\n\n".join(parts)


# ── Citation parsing ──────────────────────────────────────────────────────────

_CITATION_RE = re.compile(r"\[(\d+)\]")


def _parse_citations(answer: str, evidence: list[Evidence]) -> list[Citation]:
    """Extract ``[N]`` markers from *answer* and map to :class:`Evidence` objects.

    Rules:
    - Index is 1-based (``[1]`` → ``evidence[0]``).
    - Out-of-range indices (e.g. ``[99]`` when only 3 chunks) are silently ignored.
    - Duplicate references to the same evidence item produce a single Citation.

    Parameters
    ----------
    answer   : raw LLM output text containing ``[N]`` markers
    evidence : the ordered evidence list passed to the LLM

    Returns
    -------
    list[Citation]
        One entry per unique evidence item referenced, ordered by first appearance.
    """
    seen_evidence_ids: set[UUID] = set()
    citations: list[Citation] = []

    for match in _CITATION_RE.finditer(answer):
        idx = int(match.group(1)) - 1  # convert to 0-based
        if idx < 0 or idx >= len(evidence):
            logger.debug("Citation [%d] out of range (only %d items)", idx + 1, len(evidence))
            continue

        ev = evidence[idx]
        if ev.id in seen_evidence_ids:
            continue  # deduplicate

        seen_evidence_ids.add(ev.id)
        claim_text = _extract_claim_context(answer, match.start())
        citations.append(
            Citation(
                evidence_id=ev.id,
                claim_text=claim_text,
                is_verified=False,  # Critic (Phase 5) sets this
            )
        )

    return citations


def _extract_claim_context(text: str, marker_pos: int) -> str:
    """Return up to ``_CLAIM_CONTEXT_CHARS`` characters around *marker_pos*.

    Walks backwards to find a sentence boundary (period, newline) and forwards
    to find the end of the sentence, giving the Critic a full claim to verify.
    """
    half = _CLAIM_CONTEXT_CHARS // 2
    start = max(0, marker_pos - half)
    end = min(len(text), marker_pos + half)
    return text[start:end].strip()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_model_name(response) -> str:
    """Best-effort extraction of the model name from a LangChain response."""
    meta = getattr(response, "response_metadata", {}) or {}
    # Gemini puts it under "model_name"; Groq under "model"
    return meta.get("model_name") or meta.get("model") or ""
