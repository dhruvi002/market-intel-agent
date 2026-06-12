"""Tests for mia_agents.rag_agent — single-agent RAG baseline.

All external dependencies (Retriever, LLM) are mocked — no API calls, no
network, no vector DB required.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from mia_agents.rag_agent import (
    RAGAgent,
    RAGResponse,
    _extract_claim_context,
    _format_context,
    _parse_citations,
)
from mia_retrieval.retriever import RetrieveMode
from mia_shared.schemas import Citation, Evidence


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_evidence(
    ticker: str = "NVDA",
    filing_type: str = "10-K",
    section: str | None = "MD&A",
    text: str = "Revenue grew 217% year-over-year.",
    score: float = 0.9,
) -> Evidence:
    return Evidence(
        source_type="rag_chunk",
        ticker=ticker,
        filing_type=filing_type,
        section=section,
        text=text,
        relevance_score=score,
    )


def make_llm_response(content: str, model_name: str = "gemini-2.0-flash") -> MagicMock:
    resp = MagicMock()
    resp.content = content
    resp.response_metadata = {"model_name": model_name}
    return resp


def make_agent(
    evidence_list: list[Evidence] | None = None,
    llm_answer: str = "NVDA revenue grew [1].",
) -> RAGAgent:
    retriever = MagicMock()
    retriever.retrieve = AsyncMock(return_value=evidence_list or [])

    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=make_llm_response(llm_answer))

    settings = MagicMock()
    settings.rerank_top_k = 10

    return RAGAgent(retriever=retriever, llm=llm, settings=settings)


# ── _format_context ───────────────────────────────────────────────────────────

class TestFormatContext:
    def test_empty_returns_placeholder(self):
        assert _format_context([]) == "(no evidence retrieved)"

    def test_single_chunk_numbered(self):
        ev = make_evidence(ticker="NVDA", filing_type="10-K", section="MD&A", text="Some text.")
        result = _format_context([ev])
        assert result.startswith("[1] NVDA 10-K — MD&A")
        assert "Some text." in result

    def test_multiple_chunks_incrementing_numbers(self):
        evs = [make_evidence(ticker=t) for t in ["NVDA", "AMD", "INTC"]]
        result = _format_context(evs)
        assert "[1]" in result
        assert "[2]" in result
        assert "[3]" in result

    def test_missing_section_omits_dash(self):
        ev = make_evidence(ticker="AAPL", filing_type="10-Q", section=None)
        result = _format_context([ev])
        assert "— None" not in result
        assert "[1] AAPL 10-Q" in result

    def test_missing_ticker_omits_it(self):
        ev = make_evidence(ticker=None, filing_type="10-K", section="Risk Factors")
        result = _format_context([ev])
        assert "[1] 10-K — Risk Factors" in result

    def test_text_included(self):
        ev = make_evidence(text="Critical data point here.")
        result = _format_context([ev])
        assert "Critical data point here." in result

    def test_chunks_separated_by_blank_line(self):
        evs = [make_evidence(ticker="A"), make_evidence(ticker="B")]
        result = _format_context(evs)
        assert "\n\n" in result


# ── _parse_citations ──────────────────────────────────────────────────────────

class TestParseCitations:
    def test_no_markers_returns_empty(self):
        ev = make_evidence()
        result = _parse_citations("Revenue grew a lot.", [ev])
        assert result == []

    def test_single_marker_maps_to_evidence(self):
        ev = make_evidence()
        result = _parse_citations("Revenue grew [1].", [ev])
        assert len(result) == 1
        assert result[0].evidence_id == ev.id

    def test_multiple_different_markers(self):
        ev1 = make_evidence(ticker="NVDA")
        ev2 = make_evidence(ticker="AMD")
        result = _parse_citations("NVDA grew [1] while AMD [2] followed.", [ev1, ev2])
        assert len(result) == 2
        ids = {c.evidence_id for c in result}
        assert ev1.id in ids
        assert ev2.id in ids

    def test_duplicate_markers_produce_single_citation(self):
        ev = make_evidence()
        result = _parse_citations("Revenue [1] grew [1] again [1].", [ev])
        assert len(result) == 1

    def test_out_of_range_marker_ignored(self):
        ev = make_evidence()
        result = _parse_citations("See [99] for details.", [ev])
        assert result == []

    def test_zero_index_ignored(self):
        ev = make_evidence()
        result = _parse_citations("See [0] for details.", [ev])
        assert result == []

    def test_claim_text_populated(self):
        ev = make_evidence()
        result = _parse_citations("Revenue grew 217% year-over-year [1].", [ev])
        assert result[0].claim_text != ""

    def test_is_verified_false(self):
        ev = make_evidence()
        result = _parse_citations("Revenue grew [1].", [ev])
        assert result[0].is_verified is False

    def test_empty_evidence_list_returns_empty(self):
        result = _parse_citations("Revenue grew [1].", [])
        assert result == []


# ── RAGAgent.run ──────────────────────────────────────────────────────────────

class TestRAGAgentRun:
    def test_run_returns_rag_response(self):
        agent = make_agent()
        result = asyncio.run(agent.run("What is NVDA revenue?"))
        assert isinstance(result, RAGResponse)

    def test_run_query_preserved(self):
        agent = make_agent()
        result = asyncio.run(agent.run("NVDA data center?"))
        assert result.query == "NVDA data center?"

    def test_run_evidence_populated(self):
        evs = [make_evidence(ticker="NVDA"), make_evidence(ticker="AMD")]
        agent = make_agent(evidence_list=evs)
        result = asyncio.run(agent.run("Compare NVDA AMD"))
        assert len(result.evidence) == 2

    def test_run_calls_retriever_with_query(self):
        agent = make_agent()
        asyncio.run(agent.run("NVDA revenue?"))
        agent._retriever.retrieve.assert_awaited_once()
        call_args = agent._retriever.retrieve.call_args
        assert call_args.args[0] == "NVDA revenue?"

    def test_run_forwards_tickers(self):
        agent = make_agent()
        asyncio.run(agent.run("revenue?", tickers=["NVDA", "AMD"]))
        kwargs = agent._retriever.retrieve.call_args.kwargs
        assert kwargs["ticker_filter"] == ["NVDA", "AMD"]

    def test_run_forwards_mode(self):
        agent = make_agent()
        asyncio.run(agent.run("revenue?", mode=RetrieveMode.BM25))
        kwargs = agent._retriever.retrieve.call_args.kwargs
        assert kwargs["mode"] == RetrieveMode.BM25

    def test_run_citations_parsed(self):
        ev = make_evidence()
        agent = make_agent(evidence_list=[ev], llm_answer="Revenue grew [1].")
        result = asyncio.run(agent.run("revenue?"))
        assert len(result.citations) == 1
        assert result.citations[0].evidence_id == ev.id

    def test_run_empty_evidence_no_citations(self):
        agent = make_agent(evidence_list=[], llm_answer="I don't know [1].")
        result = asyncio.run(agent.run("revenue?"))
        assert result.citations == []

    def test_run_latency_positive(self):
        agent = make_agent()
        result = asyncio.run(agent.run("revenue?"))
        assert result.latency_ms > 0

    def test_run_session_ids_unique(self):
        agent = make_agent()
        r1 = asyncio.run(agent.run("a"))
        r2 = asyncio.run(agent.run("b"))
        assert r1.session_id != r2.session_id

    def test_run_model_used_extracted(self):
        ev = make_evidence()
        agent = make_agent(evidence_list=[ev], llm_answer="Answer [1].")
        result = asyncio.run(agent.run("q?"))
        assert result.model_used == "gemini-2.0-flash"

    def test_run_retrieval_mode_recorded(self):
        agent = make_agent()
        result = asyncio.run(agent.run("q?", mode=RetrieveMode.DENSE))
        assert result.retrieval_mode == "dense"

    def test_run_reranked_flag(self):
        agent = make_agent()
        result_on = asyncio.run(agent.run("q?", rerank=True))
        result_off = asyncio.run(agent.run("q?", rerank=False))
        assert result_on.reranked is True
        assert result_off.reranked is False


# ── RAGResponse.source_tickers ────────────────────────────────────────────────

class TestRAGResponseSourceTickers:
    def test_unique_tickers_sorted(self):
        evs = [
            make_evidence(ticker="NVDA"),
            make_evidence(ticker="AMD"),
            make_evidence(ticker="NVDA"),
        ]
        resp = RAGResponse(
            query="q",
            answer="a",
            evidence=evs,
        )
        assert resp.source_tickers == ["AMD", "NVDA"]

    def test_no_ticker_field_excluded(self):
        ev = make_evidence(ticker=None)
        resp = RAGResponse(query="q", answer="a", evidence=[ev])
        assert resp.source_tickers == []
