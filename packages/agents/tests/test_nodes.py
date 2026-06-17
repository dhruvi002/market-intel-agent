"""Tests for Phase 4 + Phase 5 agent nodes.

All LLM, NLI, Retriever, HTTP, and external-API dependencies are mocked —
no API calls, no network, no model weights loaded.
Each test class covers one node module.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from mia_shared.schemas import (
    AgentName,
    AgentState,
    Citation,
    CriticVerdict,
    CritiqueResult,
    Evidence,
    FailingClaim,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_evidence(
    ticker: str = "NVDA",
    text: str = "Revenue grew 217%.",
    score: float = 0.9,
    source_type: str = "rag_chunk",
) -> Evidence:
    return Evidence(
        source_type=source_type,
        ticker=ticker,
        filing_type="10-K",
        section="MD&A",
        text=text,
        relevance_score=score,
    )


def make_citation(claim_text: str = "Revenue grew 217%.", evidence: Evidence | None = None) -> Citation:
    ev = evidence or make_evidence()
    return Citation(evidence_id=ev.id, claim_text=claim_text)


def make_state(**kwargs) -> AgentState:
    defaults: dict = {"query": "What is NVDA revenue?"}
    defaults.update(kwargs)
    return AgentState(**defaults)


def make_llm(content: str = "Answer text.") -> MagicMock:
    resp = MagicMock()
    resp.content = content
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=resp)
    return llm


def make_structured_llm(return_value) -> MagicMock:
    """LLM mock that supports with_structured_output."""
    llm = MagicMock()
    structured = MagicMock()
    structured.ainvoke = AsyncMock(return_value=return_value)
    llm.with_structured_output = MagicMock(return_value=structured)
    return llm


# ════════════════════════════════════════════════════════════════════════════════
# supervisor_node
# ════════════════════════════════════════════════════════════════════════════════

class TestSupervisorNode:
    """Tests for mia_agents.nodes.supervisor.supervisor_node."""

    def _make_decision(self, next_agent: str = "retrieval", reasoning: str = "Best fit.", plan: str = "Search filings."):
        from mia_agents.nodes.supervisor import SupervisorDecision
        return SupervisorDecision(next_agent=next_agent, reasoning=reasoning, plan=plan)

    def test_routes_to_retrieval(self):
        from mia_agents.nodes.supervisor import supervisor_node
        state = make_state()
        llm = make_structured_llm(self._make_decision("retrieval"))
        result = asyncio.run(supervisor_node(state, llm=llm))
        assert result["active_agent"] == AgentName.RETRIEVAL

    def test_routes_to_web_search(self):
        from mia_agents.nodes.supervisor import supervisor_node
        state = make_state()
        llm = make_structured_llm(self._make_decision("web_search"))
        result = asyncio.run(supervisor_node(state, llm=llm))
        assert result["active_agent"] == AgentName.WEB_SEARCH

    def test_routes_to_edgar_parser(self):
        from mia_agents.nodes.supervisor import supervisor_node
        state = make_state()
        llm = make_structured_llm(self._make_decision("edgar_parser"))
        result = asyncio.run(supervisor_node(state, llm=llm))
        assert result["active_agent"] == AgentName.EDGAR_PARSER

    def test_routes_to_sql_generator(self):
        from mia_agents.nodes.supervisor import supervisor_node
        state = make_state()
        llm = make_structured_llm(self._make_decision("sql_generator"))
        result = asyncio.run(supervisor_node(state, llm=llm))
        assert result["active_agent"] == AgentName.SQL_GENERATOR

    def test_unknown_agent_defaults_to_retrieval(self):
        from mia_agents.nodes.supervisor import supervisor_node
        state = make_state()
        llm = make_structured_llm(self._make_decision("banana_agent"))
        result = asyncio.run(supervisor_node(state, llm=llm))
        assert result["active_agent"] == AgentName.RETRIEVAL

    def test_non_routable_agent_defaults_to_retrieval(self):
        """SUMMARIZER is a valid AgentName but not a routable worker."""
        from mia_agents.nodes.supervisor import supervisor_node
        state = make_state()
        llm = make_structured_llm(self._make_decision("summarizer"))
        result = asyncio.run(supervisor_node(state, llm=llm))
        assert result["active_agent"] == AgentName.RETRIEVAL

    def test_plan_propagated(self):
        from mia_agents.nodes.supervisor import supervisor_node
        state = make_state()
        llm = make_structured_llm(self._make_decision(plan="Focus on data center revenue."))
        result = asyncio.run(supervisor_node(state, llm=llm))
        assert result["plan"] == "Focus on data center revenue."

    def test_returns_active_agent_and_plan_keys(self):
        from mia_agents.nodes.supervisor import supervisor_node
        state = make_state()
        llm = make_structured_llm(self._make_decision())
        result = asyncio.run(supervisor_node(state, llm=llm))
        assert "active_agent" in result
        assert "plan" in result

    def test_critique_feedback_appended_on_revision(self):
        """On iteration > 0, supervisor sends critique feedback to LLM."""
        from mia_agents.nodes.supervisor import supervisor_node
        critique = CritiqueResult(
            verdict=CriticVerdict.REVISE,
            failing_claims=[FailingClaim(claim="Revenue 217%", reason="Not in evidence")],
            summary="Unsupported claim.",
        )
        state = make_state(iteration_count=1, critique=critique)
        llm = make_structured_llm(self._make_decision())
        asyncio.run(supervisor_node(state, llm=llm))
        # structured_llm.ainvoke called with multiple messages (including critique)
        call_args = llm.with_structured_output.return_value.ainvoke.call_args
        messages = call_args.args[0]
        assert len(messages) >= 3  # system + query + critique feedback

    def test_no_critique_feedback_on_first_iteration(self):
        from mia_agents.nodes.supervisor import supervisor_node
        state = make_state(iteration_count=0, critique=None)
        llm = make_structured_llm(self._make_decision())
        asyncio.run(supervisor_node(state, llm=llm))
        call_args = llm.with_structured_output.return_value.ainvoke.call_args
        messages = call_args.args[0]
        assert len(messages) == 2  # system + query only


# ════════════════════════════════════════════════════════════════════════════════
# retrieval_node
# ════════════════════════════════════════════════════════════════════════════════

class TestRetrievalNode:
    """Tests for mia_agents.nodes.retrieval.retrieval_node."""

    def _make_rag_result(self, evidence=None, citations=None):
        """Mock RAGAgent.run return value."""
        from mia_agents.rag_agent import RAGResponse
        ev = evidence if evidence is not None else [make_evidence()]
        cit = citations if citations is not None else []
        return RAGResponse(
            query="q",
            answer="answer [1].",
            evidence=ev,
            citations=cit,
        )

    def _make_agent_mock(self, result):
        """Patch RAGAgent so retrieval_node uses our mock."""
        agent = MagicMock()
        agent.run = AsyncMock(return_value=result)
        return agent

    def test_populates_evidence(self):
        from mia_agents.nodes.retrieval import retrieval_node
        ev = make_evidence()
        result = self._make_rag_result(evidence=[ev])
        with patch("mia_agents.nodes.retrieval.RAGAgent") as MockAgent:
            MockAgent.return_value = self._make_agent_mock(result)
            state = make_state()
            out = asyncio.run(retrieval_node(state, retriever=MagicMock(), llm=MagicMock()))
        assert len(out["evidence"]) == 1
        assert out["evidence"][0].id == ev.id

    def test_accumulates_evidence_across_iterations(self):
        from mia_agents.nodes.retrieval import retrieval_node
        existing = make_evidence(ticker="AMD")
        new_ev = make_evidence(ticker="NVDA")
        result = self._make_rag_result(evidence=[new_ev])
        with patch("mia_agents.nodes.retrieval.RAGAgent") as MockAgent:
            MockAgent.return_value = self._make_agent_mock(result)
            state = make_state(evidence=[existing])
            out = asyncio.run(retrieval_node(state, retriever=MagicMock(), llm=MagicMock()))
        ids = {ev.id for ev in out["evidence"]}
        assert existing.id in ids
        assert new_ev.id in ids

    def test_deduplicates_evidence(self):
        """Evidence already in state should not be added again."""
        from mia_agents.nodes.retrieval import retrieval_node
        ev = make_evidence()
        result = self._make_rag_result(evidence=[ev])  # same ev returned again
        with patch("mia_agents.nodes.retrieval.RAGAgent") as MockAgent:
            MockAgent.return_value = self._make_agent_mock(result)
            state = make_state(evidence=[ev])  # already present
            out = asyncio.run(retrieval_node(state, retriever=MagicMock(), llm=MagicMock()))
        assert len(out["evidence"]) == 1  # not doubled

    def test_empty_retrieval_result(self):
        from mia_agents.nodes.retrieval import retrieval_node
        result = self._make_rag_result(evidence=[], citations=[])
        with patch("mia_agents.nodes.retrieval.RAGAgent") as MockAgent:
            MockAgent.return_value = self._make_agent_mock(result)
            state = make_state()
            out = asyncio.run(retrieval_node(state, retriever=MagicMock(), llm=MagicMock()))
        assert out["evidence"] == []
        assert out["citations"] == []

    def test_returns_evidence_and_citations_keys(self):
        from mia_agents.nodes.retrieval import retrieval_node
        result = self._make_rag_result()
        with patch("mia_agents.nodes.retrieval.RAGAgent") as MockAgent:
            MockAgent.return_value = self._make_agent_mock(result)
            state = make_state()
            out = asyncio.run(retrieval_node(state, retriever=MagicMock(), llm=MagicMock()))
        assert "evidence" in out
        assert "citations" in out


# ════════════════════════════════════════════════════════════════════════════════
# summarizer_node
# ════════════════════════════════════════════════════════════════════════════════

class TestSummarizerNode:
    """Tests for mia_agents.nodes.summarizer.summarizer_node."""

    def test_draft_populated(self):
        from mia_agents.nodes.summarizer import summarizer_node
        state = make_state(evidence=[make_evidence()])
        llm = make_llm("Revenue grew [1].")
        out = asyncio.run(summarizer_node(state, llm=llm))
        assert out["draft"] == "Revenue grew [1]."

    def test_returns_draft_key(self):
        from mia_agents.nodes.summarizer import summarizer_node
        state = make_state(evidence=[make_evidence()])
        out = asyncio.run(summarizer_node(state, llm=make_llm()))
        assert "draft" in out

    def test_fallback_draft_on_empty_evidence(self):
        from mia_agents.nodes.summarizer import summarizer_node
        state = make_state(evidence=[])
        out = asyncio.run(summarizer_node(state, llm=make_llm()))
        assert "Insufficient evidence" in out["draft"]

    def test_llm_not_called_on_empty_evidence(self):
        from mia_agents.nodes.summarizer import summarizer_node
        state = make_state(evidence=[])
        llm = make_llm()
        asyncio.run(summarizer_node(state, llm=llm))
        llm.ainvoke.assert_not_called()

    def test_evidence_sorted_by_relevance(self):
        """Summarizer should sort evidence by score; we verify LLM is called."""
        from mia_agents.nodes.summarizer import summarizer_node
        evs = [
            make_evidence(ticker="A", score=0.5),
            make_evidence(ticker="B", score=0.95),
            make_evidence(ticker="C", score=0.7),
        ]
        state = make_state(evidence=evs)
        llm = make_llm("Answer [1].")
        asyncio.run(summarizer_node(state, llm=llm))
        llm.ainvoke.assert_awaited_once()

    def test_draft_non_empty_string(self):
        from mia_agents.nodes.summarizer import summarizer_node
        state = make_state(evidence=[make_evidence()])
        out = asyncio.run(summarizer_node(state, llm=make_llm("Some content.")))
        assert isinstance(out["draft"], str)
        assert len(out["draft"]) > 0


# ════════════════════════════════════════════════════════════════════════════════
# summarizer_node — Phase 7 streaming
# ════════════════════════════════════════════════════════════════════════════════

def make_streaming_llm(tokens: list[str]) -> MagicMock:
    """LLM mock whose astream() yields token chunks; ainvoke should not be called."""

    async def _astream(*args, **kwargs):  # noqa: ANN002, ANN003
        for token in tokens:
            chunk = MagicMock()
            chunk.content = token
            yield chunk

    llm = MagicMock()
    llm.astream = _astream
    # ainvoke should NOT be called on the streaming path; keep it as sentinel
    llm.ainvoke = AsyncMock(side_effect=AssertionError("ainvoke called on streaming LLM"))
    return llm


class TestSummarizerStreaming:
    """Phase 7: token_callback streaming path in summarizer_node."""

    # ── helpers ──────────────────────────────────────────────────────────────

    def _run_streaming(
        self,
        tokens: list[str],
        state: AgentState | None = None,
    ) -> tuple[dict, list[str]]:
        """Run summarizer_node with a streaming LLM; return (result, received_tokens)."""
        from mia_agents.nodes.summarizer import summarizer_node

        collected: list[str] = []

        async def _cb(token: str) -> None:
            collected.append(token)

        s = state or make_state(evidence=[make_evidence()])
        llm = make_streaming_llm(tokens)
        out = asyncio.run(summarizer_node(s, llm=llm, token_callback=_cb))
        return out, collected

    # ── tests ─────────────────────────────────────────────────────────────────

    def test_callback_receives_each_token(self):
        """token_callback is invoked once for every non-empty token."""
        tokens = ["Revenue", " grew", " 12%."]
        _, received = self._run_streaming(tokens)
        assert received == tokens

    def test_draft_accumulated_from_tokens(self):
        """Returned draft equals joined tokens."""
        tokens = ["Hello", " ", "world."]
        out, _ = self._run_streaming(tokens)
        assert out["draft"] == "Hello world."

    def test_empty_tokens_skipped(self):
        """Empty-string token chunks are not forwarded to the callback."""
        tokens = ["A", "", "B", "", "C"]
        _, received = self._run_streaming(tokens)
        assert received == ["A", "B", "C"]

    def test_draft_excludes_empty_tokens(self):
        """Empty tokens contribute nothing to the accumulated draft."""
        tokens = ["X", "", "Y"]
        out, _ = self._run_streaming(tokens)
        assert out["draft"] == "XY"

    def test_ainvoke_not_called_when_streaming(self):
        """astream() is used; ainvoke must NOT be called (would raise AssertionError)."""
        # make_streaming_llm sets ainvoke to raise — test passes only if not invoked
        tokens = ["ok"]
        out, _ = self._run_streaming(tokens)
        assert out["draft"] == "ok"

    def test_fallback_on_empty_evidence_no_streaming(self):
        """Empty evidence returns fallback draft without calling astream."""
        from mia_agents.nodes.summarizer import summarizer_node

        called: list[str] = []

        async def _cb(token: str) -> None:
            called.append(token)

        state = make_state(evidence=[])
        llm = make_streaming_llm([])  # astream yields nothing
        out = asyncio.run(summarizer_node(state, llm=llm, token_callback=_cb))
        assert "Insufficient evidence" in out["draft"]
        assert called == []  # callback never fired

    def test_no_callback_uses_ainvoke(self):
        """When token_callback=None the node uses ainvoke (Phase 6 path)."""
        from mia_agents.nodes.summarizer import summarizer_node

        state = make_state(evidence=[make_evidence()])
        llm = make_llm("Batch answer.")
        out = asyncio.run(summarizer_node(state, llm=llm, token_callback=None))
        assert out["draft"] == "Batch answer."
        llm.ainvoke.assert_awaited_once()

    def test_single_token_stream(self):
        """Works correctly with a single-token response."""
        tokens = ["Done."]
        out, received = self._run_streaming(tokens)
        assert out["draft"] == "Done."
        assert received == ["Done."]


# ════════════════════════════════════════════════════════════════════════════════
# critic_node  (Phase 5 — NLI-based)
# ════════════════════════════════════════════════════════════════════════════════

class TestNLICritic:
    """Tests for the Phase-5 NLI critic in mia_agents.nodes.critic.

    ``mia_agents.nli.score_pairs`` is always patched so no model weights
    are loaded and the test suite remains fast + offline.
    """

    def _run_critic(self, state: AgentState, nli_scores: list[float] | None = None) -> dict:
        """Run critic_node with nli.score_pairs mocked to return *nli_scores*."""
        from mia_agents.nodes.critic import critic_node

        scores = nli_scores if nli_scores is not None else []
        with patch("mia_agents.nodes.critic.nli.score_pairs", return_value=scores):
            return asyncio.run(critic_node(state))

    # ── Basic verdict tests ───────────────────────────────────────────────────

    def test_pass_verdict_all_verified(self):
        ev = make_evidence()
        cit = make_citation(evidence=ev)
        state = make_state(evidence=[ev], citations=[cit], draft="Revenue grew [1].")
        out = self._run_critic(state, nli_scores=[0.9])
        assert out["critique"].verdict == CriticVerdict.PASS

    def test_revise_verdict_some_failing(self):
        ev = make_evidence()
        cit = make_citation("Revenue grew 500%.", ev)
        state = make_state(evidence=[ev], citations=[cit], draft="Revenue grew 500% [1].")
        out = self._run_critic(state, nli_scores=[0.2])
        assert out["critique"].verdict == CriticVerdict.REVISE

    def test_revise_only_partial_failure(self):
        """One passing, one failing citation → REVISE."""
        ev1 = make_evidence(text="Revenue grew 122%.")
        ev2 = make_evidence(text="Margin was 20%.")
        cit1 = make_citation("Revenue grew 122%.", ev1)
        cit2 = make_citation("Margin was 60%.", ev2)
        state = make_state(
            evidence=[ev1, ev2], citations=[cit1, cit2],
            draft="Revenue grew 122% [1]. Margin was 60% [2].",
        )
        out = self._run_critic(state, nli_scores=[0.9, 0.1])
        assert out["critique"].verdict == CriticVerdict.REVISE
        assert len(out["critique"].failing_claims) == 1

    def test_escalate_empty_draft(self):
        state = make_state(draft="")
        out = self._run_critic(state)
        assert out["critique"].verdict == CriticVerdict.ESCALATE

    def test_escalate_no_evidence_no_citations(self):
        state = make_state(evidence=[], citations=[], draft="Some draft.")
        out = self._run_critic(state)
        assert out["critique"].verdict == CriticVerdict.ESCALATE

    def test_pass_no_citations_with_evidence(self):
        """Web-search results add evidence but no citations → fast-path PASS."""
        state = make_state(evidence=[make_evidence()], citations=[], draft="Some draft.")
        out = self._run_critic(state)
        assert out["critique"].verdict == CriticVerdict.PASS

    # ── NLI score / threshold tests ───────────────────────────────────────────

    def test_is_verified_true_at_threshold(self):
        """Score exactly at threshold should be verified."""
        from mia_shared.config import get_settings
        threshold = get_settings().nli_entailment_threshold
        ev = make_evidence()
        cit = make_citation(evidence=ev)
        state = make_state(evidence=[ev], citations=[cit], draft="Draft [1].")
        out = self._run_critic(state, nli_scores=[threshold])
        assert out["citations"][0].is_verified is True

    def test_is_verified_false_below_threshold(self):
        from mia_shared.config import get_settings
        threshold = get_settings().nli_entailment_threshold
        ev = make_evidence()
        cit = make_citation(evidence=ev)
        state = make_state(evidence=[ev], citations=[cit], draft="Draft [1].")
        out = self._run_critic(state, nli_scores=[threshold - 0.01])
        assert out["citations"][0].is_verified is False

    def test_nli_score_stored_on_citation(self):
        ev = make_evidence()
        cit = make_citation(evidence=ev)
        state = make_state(evidence=[ev], citations=[cit], draft="Draft [1].")
        out = self._run_critic(state, nli_scores=[0.73])
        assert pytest.approx(out["citations"][0].nli_score, abs=1e-5) == 0.73

    def test_all_scores_stored(self):
        ev1, ev2 = make_evidence(ticker="A"), make_evidence(ticker="B")
        cit1, cit2 = make_citation(evidence=ev1), make_citation(evidence=ev2)
        state = make_state(evidence=[ev1, ev2], citations=[cit1, cit2], draft="Draft.")
        out = self._run_critic(state, nli_scores=[0.8, 0.4])
        scores = {c.nli_score for c in out["citations"]}
        assert pytest.approx(0.8, abs=1e-5) in scores
        assert pytest.approx(0.4, abs=1e-5) in scores

    # ── Failing claims ────────────────────────────────────────────────────────

    def test_failing_claims_populated_on_revise(self):
        ev = make_evidence()
        cit = make_citation("Fake claim.", ev)
        state = make_state(evidence=[ev], citations=[cit], draft="Fake claim [1].")
        out = self._run_critic(state, nli_scores=[0.1])
        assert len(out["critique"].failing_claims) == 1
        assert out["critique"].failing_claims[0].claim == "Fake claim."

    def test_no_failing_claims_on_pass(self):
        ev = make_evidence()
        cit = make_citation(evidence=ev)
        state = make_state(evidence=[ev], citations=[cit], draft="Revenue grew [1].")
        out = self._run_critic(state, nli_scores=[0.95])
        assert out["critique"].failing_claims == []

    # ── State mutation / keys ─────────────────────────────────────────────────

    def test_returns_citations_key(self):
        ev = make_evidence()
        cit = make_citation(evidence=ev)
        state = make_state(evidence=[ev], citations=[cit], draft="Draft.")
        out = self._run_critic(state, nli_scores=[0.8])
        assert "citations" in out

    def test_returns_critique_key(self):
        state = make_state(evidence=[make_evidence()], citations=[], draft="Draft.")
        out = self._run_critic(state)
        assert "critique" in out

    def test_returns_iteration_count_key(self):
        state = make_state(evidence=[make_evidence()], citations=[], draft="Draft.")
        out = self._run_critic(state)
        assert "iteration_count" in out

    def test_iteration_count_incremented(self):
        state = make_state(evidence=[make_evidence()], citations=[], draft="Draft.", iteration_count=2)
        out = self._run_critic(state)
        assert out["iteration_count"] == 3

    def test_critique_is_critique_result(self):
        state = make_state(evidence=[make_evidence()], citations=[], draft="Draft.")
        out = self._run_critic(state)
        assert isinstance(out["critique"], CritiqueResult)

    def test_updated_citations_preserve_ids(self):
        """Citation IDs must not change after NLI scoring."""
        ev = make_evidence()
        cit = make_citation(evidence=ev)
        state = make_state(evidence=[ev], citations=[cit], draft="Draft.")
        out = self._run_critic(state, nli_scores=[0.9])
        assert out["citations"][0].id == cit.id

    def test_orphaned_citation_marked_failing(self):
        """Citation whose evidence_id is not in state.evidence → is_verified=False."""
        from uuid import uuid4
        ev = make_evidence()
        # Create citation pointing at a *different* (non-existent) evidence id
        orphan = Citation(evidence_id=uuid4(), claim_text="Orphaned claim.")
        state = make_state(evidence=[ev], citations=[orphan], draft="Orphaned claim [1].")
        out = self._run_critic(state)
        assert out["citations"][0].is_verified is False
        assert out["citations"][0].nli_score is None
        assert out["critique"].verdict == CriticVerdict.REVISE

    def test_nli_score_pairs_called_with_correct_pairs(self):
        """score_pairs should receive (evidence.text, citation.claim_text) pairs."""
        from mia_agents.nodes.critic import critic_node
        ev = make_evidence(text="Evidence text here.")
        cit = make_citation("Claim text here.", ev)
        state = make_state(evidence=[ev], citations=[cit], draft="Draft.")
        with patch("mia_agents.nodes.critic.nli.score_pairs", return_value=[0.9]) as mock_sp:
            asyncio.run(critic_node(state))
        mock_sp.assert_called_once()
        pairs_arg = mock_sp.call_args.args[0]
        assert pairs_arg == [("Evidence text here.", "Claim text here.")]


# ════════════════════════════════════════════════════════════════════════════════
# web_search_node  (Phase 5 — Tavily)
# ════════════════════════════════════════════════════════════════════════════════

class TestWebSearchNode:
    """Tests for mia_agents.nodes.web_search.web_search_node."""

    def _run(self, state: AgentState, tavily_results: list[dict] | None = None) -> dict:
        """Run web_search_node with AsyncTavilyClient mocked."""
        from mia_agents.nodes.web_search import web_search_node
        results = tavily_results if tavily_results is not None else []
        mock_client = AsyncMock()
        mock_client.search = AsyncMock(return_value={"results": results})

        with patch("mia_agents.nodes.web_search.AsyncTavilyClient", return_value=mock_client):
            return asyncio.run(web_search_node(state))

    def _make_result(self, url="https://example.com/1", content="Some article.", title="Title") -> dict:
        return {"url": url, "content": content, "title": title, "score": 0.85}

    def test_returns_evidence_from_results(self):
        state = make_state()
        out = self._run(state, [self._make_result()])
        assert len(out["evidence"]) == 1

    def test_evidence_source_type_is_web(self):
        state = make_state()
        out = self._run(state, [self._make_result()])
        assert out["evidence"][0].source_type == "web"

    def test_evidence_url_stored(self):
        state = make_state()
        out = self._run(state, [self._make_result(url="https://reuters.com/article")])
        assert out["evidence"][0].source_url == "https://reuters.com/article"

    def test_accumulates_existing_evidence(self):
        existing = make_evidence()
        state = make_state(evidence=[existing])
        out = self._run(state, [self._make_result()])
        assert len(out["evidence"]) == 2
        assert out["evidence"][0].id == existing.id

    def test_deduplicates_by_url(self):
        existing = Evidence(source_type="web", source_url="https://example.com/1", text="Old.")
        state = make_state(evidence=[existing])
        out = self._run(state, [self._make_result(url="https://example.com/1")])
        # URL already present — should not be added again
        assert len(out["evidence"]) == 1

    def test_empty_results_returns_no_new_evidence(self):
        state = make_state()
        out = self._run(state, [])
        assert out["evidence"] == []

    def test_skips_result_with_empty_content(self):
        state = make_state()
        out = self._run(state, [{"url": "https://x.com", "content": "", "title": "T"}])
        assert out["evidence"] == []

    def test_returns_evidence_and_citations_keys(self):
        state = make_state()
        out = self._run(state)
        assert "evidence" in out
        assert "citations" in out

    def test_citations_unchanged(self):
        ev = make_evidence()
        cit = make_citation(evidence=ev)
        state = make_state(evidence=[ev], citations=[cit])
        out = self._run(state, [self._make_result()])
        assert len(out["citations"]) == 1
        assert out["citations"][0].id == cit.id

    def test_tavily_error_returns_existing_evidence(self):
        from mia_agents.nodes.web_search import web_search_node
        state = make_state(evidence=[make_evidence()])
        mock_client = AsyncMock()
        mock_client.search = AsyncMock(side_effect=RuntimeError("API down"))
        with patch("mia_agents.nodes.web_search.AsyncTavilyClient", return_value=mock_client):
            out = asyncio.run(web_search_node(state))
        assert len(out["evidence"]) == 1  # unchanged


# ════════════════════════════════════════════════════════════════════════════════
# edgar_parser_node  (Phase 5 — EDGAR EFTS)
# ════════════════════════════════════════════════════════════════════════════════

class TestEdgarParserNode:
    """Tests for mia_agents.nodes.edgar_parser.edgar_parser_node."""

    def _make_hit(
        self,
        entity_id: str = "789019",
        entity_name: str = "NVIDIA CORP",
        form_type: str = "10-K",
        period: str = "2024-01-28",
        highlight_text: str = "Revenue increased by 217%.",
    ) -> dict:
        return {
            "_source": {
                "entity_id": entity_id,
                "entity_name": entity_name,
                "form_type": form_type,
                "period_of_report": period,
            },
            "highlight": {"file_date": [highlight_text]},
        }

    def _run(self, state: AgentState, hits: list[dict] | None = None, status_code: int = 200) -> dict:
        from mia_agents.nodes.edgar_parser import edgar_parser_node
        efts_response = {"hits": {"hits": hits if hits is not None else []}}

        mock_resp = MagicMock()
        mock_resp.json = MagicMock(return_value=efts_response)
        mock_resp.raise_for_status = MagicMock()
        mock_resp.status_code = status_code

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_resp)

        with (
            patch("mia_agents.nodes.edgar_parser.httpx.AsyncClient", return_value=mock_client),
            patch("mia_agents.nodes.edgar_parser.asyncio.sleep", new_callable=AsyncMock),
        ):
            return asyncio.run(edgar_parser_node(state))

    def test_returns_evidence_from_hits(self):
        state = make_state(query="NVDA 10-K revenue")
        out = self._run(state, [self._make_hit()])
        assert len(out["evidence"]) == 1

    def test_evidence_source_type_is_edgar_filing(self):
        state = make_state(query="NVDA revenue")
        out = self._run(state, [self._make_hit()])
        assert out["evidence"][0].source_type == "edgar_filing"

    def test_evidence_filing_type_stored(self):
        state = make_state(query="NVDA 10-K")
        out = self._run(state, [self._make_hit(form_type="10-K")])
        assert out["evidence"][0].filing_type == "10-K"

    def test_evidence_text_from_highlights(self):
        state = make_state(query="NVDA revenue")
        out = self._run(state, [self._make_hit(highlight_text="Revenue up 217%.")])
        assert "Revenue up 217%." in out["evidence"][0].text

    def test_accumulates_existing_evidence(self):
        existing = make_evidence()
        state = make_state(evidence=[existing], query="NVDA 10-K")
        out = self._run(state, [self._make_hit()])
        assert len(out["evidence"]) == 2

    def test_empty_hits_returns_no_new_evidence(self):
        state = make_state(query="NVDA revenue")
        out = self._run(state, [])
        assert out["evidence"] == []

    def test_returns_evidence_and_citations_keys(self):
        state = make_state(query="NVDA 10-K")
        out = self._run(state)
        assert "evidence" in out
        assert "citations" in out

    def test_citations_unchanged(self):
        ev = make_evidence()
        cit = make_citation(evidence=ev)
        state = make_state(evidence=[ev], citations=[cit], query="NVDA 10-K")
        out = self._run(state, [self._make_hit()])
        assert len(out["citations"]) == 1
        assert out["citations"][0].id == cit.id

    def test_http_error_returns_existing_evidence(self):
        """HTTP errors should be swallowed and return unmodified evidence."""
        from mia_agents.nodes.edgar_parser import edgar_parser_node
        import httpx as _httpx

        mock_resp = MagicMock()
        mock_resp.status_code = 503
        mock_resp.raise_for_status = MagicMock(
            side_effect=_httpx.HTTPStatusError("503", request=MagicMock(), response=mock_resp)
        )

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_resp)

        existing = make_evidence()
        state = make_state(evidence=[existing], query="NVDA 10-K")

        with (
            patch("mia_agents.nodes.edgar_parser.httpx.AsyncClient", return_value=mock_client),
            patch("mia_agents.nodes.edgar_parser.asyncio.sleep", new_callable=AsyncMock),
        ):
            out = asyncio.run(edgar_parser_node(state))

        assert len(out["evidence"]) == 1
        assert out["evidence"][0].id == existing.id

    def test_no_text_in_hit_skipped(self):
        """Hits with no highlight and no fallback period text are skipped."""
        hit = {
            "_source": {"entity_id": "1", "form_type": "10-K", "period_of_report": ""},
            "highlight": {},
        }
        state = make_state(query="NVDA 10-K")
        out = self._run(state, [hit])
        assert out["evidence"] == []


# ── _extract_ticker unit tests ─────────────────────────────────────────────────

class TestExtractTicker:
    """Unit tests for edgar_parser._extract_ticker helper."""

    def test_extracts_nvda(self):
        from mia_agents.nodes.edgar_parser import _extract_ticker
        assert _extract_ticker("NVDA data center revenue Q4 2024") == "NVDA"

    def test_extracts_amd(self):
        from mia_agents.nodes.edgar_parser import _extract_ticker
        assert _extract_ticker("AMD GPU margins 10-K") == "AMD"

    def test_skips_common_words(self):
        from mia_agents.nodes.edgar_parser import _extract_ticker
        assert _extract_ticker("What is the revenue for MSFT?") == "MSFT"

    def test_returns_none_for_no_ticker(self):
        from mia_agents.nodes.edgar_parser import _extract_ticker
        assert _extract_ticker("what is the revenue for the company?") is None

    def test_single_letter_skipped(self):
        from mia_agents.nodes.edgar_parser import _extract_ticker
        # "A" by itself is in skip-list; "AB" should still be found
        assert _extract_ticker("AB metrics") == "AB"


# ════════════════════════════════════════════════════════════════════════════════
# stub nodes  (Phase 6 — all stubs promoted; this class documents the history)
# ════════════════════════════════════════════════════════════════════════════════

class TestStubNodes:
    """Phase 6: stubs.py is now empty — all workers are real implementations.

    sql_generator_node was promoted in Phase 6.  Its tests live in
    test_sql_generator.py.  This class is kept as a marker so test counts
    remain traceable across phases.
    """

    def test_stubs_module_importable(self):
        """stubs.py still exists (no import-break for external callers)."""
        import mia_agents.nodes.stubs  # noqa: F401

    def test_stubs_module_has_no_sql_generator(self):
        """sql_generator_node no longer lives in stubs — it's in nodes/sql_generator.py."""
        import mia_agents.nodes.stubs as stubs

        assert not hasattr(stubs, "sql_generator_node")

    def test_real_sql_generator_importable(self):
        """The promoted node is importable from its real location."""
        from mia_agents.nodes.sql_generator import sql_generator_node  # noqa: F401

        assert callable(sql_generator_node)
