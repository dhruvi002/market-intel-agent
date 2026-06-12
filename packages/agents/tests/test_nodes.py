"""Tests for Phase 4 agent nodes.

All LLM and Retriever dependencies are mocked — no API calls, no network.
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
    CriticVerdict,
    CritiqueResult,
    Evidence,
    FailingClaim,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_evidence(ticker: str = "NVDA", text: str = "Revenue grew 217%.", score: float = 0.9) -> Evidence:
    return Evidence(
        source_type="rag_chunk",
        ticker=ticker,
        filing_type="10-K",
        section="MD&A",
        text=text,
        relevance_score=score,
    )


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
        ev = evidence or [make_evidence()]
        cit = citations or []
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
# critic_node
# ════════════════════════════════════════════════════════════════════════════════

class TestCriticNode:
    """Tests for mia_agents.nodes.critic.critic_node."""

    def _make_critique_out(self, verdict: CriticVerdict = CriticVerdict.PASS, summary: str = "All good."):
        from mia_agents.nodes.critic import _CritiqueOut
        return _CritiqueOut(verdict=verdict, summary=summary)

    def test_pass_verdict(self):
        from mia_agents.nodes.critic import critic_node
        state = make_state(evidence=[make_evidence()], draft="Revenue grew [1].")
        llm = make_structured_llm(self._make_critique_out(CriticVerdict.PASS))
        out = asyncio.run(critic_node(state, llm=llm))
        assert out["critique"].verdict == CriticVerdict.PASS

    def test_revise_verdict(self):
        from mia_agents.nodes.critic import critic_node
        state = make_state(evidence=[make_evidence()], draft="Revenue grew 500% [1].")
        llm = make_structured_llm(self._make_critique_out(CriticVerdict.REVISE, "Unsupported figure."))
        out = asyncio.run(critic_node(state, llm=llm))
        assert out["critique"].verdict == CriticVerdict.REVISE

    def test_escalate_verdict(self):
        from mia_agents.nodes.critic import critic_node
        state = make_state(evidence=[], draft="No evidence available.")
        llm = make_structured_llm(self._make_critique_out(CriticVerdict.ESCALATE))
        out = asyncio.run(critic_node(state, llm=llm))
        assert out["critique"].verdict == CriticVerdict.ESCALATE

    def test_iteration_count_incremented(self):
        from mia_agents.nodes.critic import critic_node
        state = make_state(evidence=[make_evidence()], draft="Draft.", iteration_count=1)
        llm = make_structured_llm(self._make_critique_out())
        out = asyncio.run(critic_node(state, llm=llm))
        assert out["iteration_count"] == 2

    def test_iteration_count_starts_at_zero(self):
        from mia_agents.nodes.critic import critic_node
        state = make_state(evidence=[make_evidence()], draft="Draft.", iteration_count=0)
        llm = make_structured_llm(self._make_critique_out())
        out = asyncio.run(critic_node(state, llm=llm))
        assert out["iteration_count"] == 1

    def test_empty_draft_triggers_escalate(self):
        """Critic should issue ESCALATE without calling LLM when draft is empty."""
        from mia_agents.nodes.critic import critic_node
        state = make_state(evidence=[], draft="")
        llm = make_structured_llm(self._make_critique_out())
        out = asyncio.run(critic_node(state, llm=llm))
        assert out["critique"].verdict == CriticVerdict.ESCALATE
        llm.with_structured_output.return_value.ainvoke.assert_not_called()

    def test_returns_critique_and_iteration_count_keys(self):
        from mia_agents.nodes.critic import critic_node
        state = make_state(evidence=[make_evidence()], draft="Draft.")
        llm = make_structured_llm(self._make_critique_out())
        out = asyncio.run(critic_node(state, llm=llm))
        assert "critique" in out
        assert "iteration_count" in out

    def test_critique_is_critique_result(self):
        from mia_agents.nodes.critic import critic_node
        state = make_state(evidence=[make_evidence()], draft="Draft.")
        llm = make_structured_llm(self._make_critique_out())
        out = asyncio.run(critic_node(state, llm=llm))
        assert isinstance(out["critique"], CritiqueResult)

    def test_structured_output_parse_failure_defaults_to_revise(self):
        """If LLM returns malformed JSON, critic should default to REVISE."""
        from mia_agents.nodes.critic import critic_node
        state = make_state(evidence=[make_evidence()], draft="Draft.")
        broken_llm = MagicMock()
        structured = MagicMock()
        structured.ainvoke = AsyncMock(side_effect=ValueError("bad json"))
        broken_llm.with_structured_output = MagicMock(return_value=structured)
        out = asyncio.run(critic_node(state, llm=broken_llm))
        assert out["critique"].verdict == CriticVerdict.REVISE

    def test_failing_claims_propagated(self):
        from mia_agents.nodes.critic import _CritiqueOut, critic_node
        from mia_shared.schemas import FailingClaim
        out_model = _CritiqueOut(
            verdict=CriticVerdict.REVISE,
            failing_claims=[FailingClaim(claim="Revenue X", reason="Not found")],
            summary="Claim unsupported.",
        )
        state = make_state(evidence=[make_evidence()], draft="Revenue X [1].")
        llm = make_structured_llm(out_model)
        out = asyncio.run(critic_node(state, llm=llm))
        assert len(out["critique"].failing_claims) == 1
        assert out["critique"].failing_claims[0].claim == "Revenue X"


# ════════════════════════════════════════════════════════════════════════════════
# stub nodes
# ════════════════════════════════════════════════════════════════════════════════

class TestStubNodes:
    """Tests for mia_agents.nodes.stubs — web_search, edgar_parser, sql_generator."""

    @pytest.mark.parametrize("node_name,agent_name", [
        ("web_search_node", AgentName.WEB_SEARCH),
        ("edgar_parser_node", AgentName.EDGAR_PARSER),
        ("sql_generator_node", AgentName.SQL_GENERATOR),
    ])
    def test_stub_returns_messages(self, node_name, agent_name):
        import mia_agents.nodes.stubs as stubs
        node_fn = getattr(stubs, node_name)
        state = make_state()
        out = asyncio.run(node_fn(state))
        assert "messages" in out
        assert len(out["messages"]) == 1
        assert agent_name.value in out["messages"][0]["content"]

    @pytest.mark.parametrize("node_name", [
        "web_search_node", "edgar_parser_node", "sql_generator_node"
    ])
    def test_stub_returns_active_agent_none(self, node_name):
        import mia_agents.nodes.stubs as stubs
        node_fn = getattr(stubs, node_name)
        state = make_state()
        out = asyncio.run(node_fn(state))
        assert out.get("active_agent") is None

    @pytest.mark.parametrize("node_name", [
        "web_search_node", "edgar_parser_node", "sql_generator_node"
    ])
    def test_stub_is_async(self, node_name):
        import asyncio as _asyncio
        import mia_agents.nodes.stubs as stubs
        node_fn = getattr(stubs, node_name)
        state = make_state()
        coro = node_fn(state)
        assert _asyncio.iscoroutine(coro)
        asyncio.run(coro)
