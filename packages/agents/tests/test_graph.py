"""Tests for the Phase-4 LangGraph graph — routing logic and graph structure.

These tests focus on the routing functions and graph topology rather than
end-to-end execution (which would require real LLM + Retriever).

Integration tests that call ``build_graph`` directly mock out all node
functions to verify the graph wiring.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mia_shared.schemas import (
    AgentName,
    AgentState,
    CriticVerdict,
    CritiqueResult,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_state(**kwargs) -> AgentState:
    defaults: dict = {"query": "What is NVDA revenue?"}
    defaults.update(kwargs)
    return AgentState(**defaults)


# ════════════════════════════════════════════════════════════════════════════════
# _route_after_supervisor
# ════════════════════════════════════════════════════════════════════════════════

class TestRouteAfterSupervisor:
    """Unit tests for the supervisor → worker routing function."""

    @pytest.mark.parametrize("agent,expected_node", [
        (AgentName.RETRIEVAL, "retrieval"),
        (AgentName.WEB_SEARCH, "web_search"),
        (AgentName.EDGAR_PARSER, "edgar_parser"),
        (AgentName.SQL_GENERATOR, "sql_generator"),
    ])
    def test_routes_known_agents(self, agent, expected_node):
        from mia_agents.graph import _route_after_supervisor
        state = make_state(active_agent=agent)
        assert _route_after_supervisor(state) == expected_node

    def test_none_active_agent_defaults_to_retrieval(self):
        from mia_agents.graph import _route_after_supervisor
        state = make_state(active_agent=None)
        assert _route_after_supervisor(state) == "retrieval"

    def test_non_routable_agent_defaults_to_retrieval(self):
        """SUMMARIZER is not a routable worker node."""
        from mia_agents.graph import _route_after_supervisor
        state = make_state(active_agent=AgentName.SUMMARIZER)
        assert _route_after_supervisor(state) == "retrieval"

    def test_critic_agent_defaults_to_retrieval(self):
        from mia_agents.graph import _route_after_supervisor
        state = make_state(active_agent=AgentName.CRITIC)
        assert _route_after_supervisor(state) == "retrieval"


# ════════════════════════════════════════════════════════════════════════════════
# _route_after_critic
# ════════════════════════════════════════════════════════════════════════════════

class TestRouteAfterCritic:
    """Unit tests for the critic → END/supervisor routing function."""

    def test_pass_verdict_goes_to_end(self):
        from langgraph.graph import END
        from mia_agents.graph import _route_after_critic
        state = make_state(
            iteration_count=1,
            critique=CritiqueResult(verdict=CriticVerdict.PASS, summary="Good."),
        )
        assert _route_after_critic(state) == END

    def test_escalate_verdict_goes_to_end(self):
        from langgraph.graph import END
        from mia_agents.graph import _route_after_critic
        state = make_state(
            iteration_count=1,
            critique=CritiqueResult(verdict=CriticVerdict.ESCALATE, summary="Need different approach."),
        )
        assert _route_after_critic(state) == END

    def test_revise_verdict_goes_to_supervisor(self):
        from mia_agents.graph import _route_after_critic
        state = make_state(
            iteration_count=1,
            critique=CritiqueResult(verdict=CriticVerdict.REVISE, summary="Fix claims."),
        )
        assert _route_after_critic(state) == "supervisor"

    def test_iteration_cap_terminates_revise(self):
        """Even a REVISE verdict should go to END when iteration_count >= max."""
        from langgraph.graph import END
        from mia_agents.graph import _route_after_critic
        from mia_shared.config import get_settings
        max_iter = get_settings().max_iterations
        state = make_state(
            iteration_count=max_iter,
            critique=CritiqueResult(verdict=CriticVerdict.REVISE, summary="Still failing."),
        )
        assert _route_after_critic(state) == END

    def test_no_critique_goes_to_end(self):
        """If critique is None (shouldn't happen normally), route to END safely."""
        from langgraph.graph import END
        from mia_agents.graph import _route_after_critic
        state = make_state(iteration_count=1, critique=None)
        assert _route_after_critic(state) == END

    def test_zero_iterations_revise_goes_to_supervisor(self):
        """First revision (iteration_count=1, below cap) should loop back."""
        from mia_agents.graph import _route_after_critic
        state = make_state(
            iteration_count=1,
            critique=CritiqueResult(verdict=CriticVerdict.REVISE, summary="Revise."),
        )
        assert _route_after_critic(state) == "supervisor"

    def test_iteration_cap_minus_one_still_allows_revision(self):
        from mia_agents.graph import _route_after_critic
        from mia_shared.config import get_settings
        max_iter = get_settings().max_iterations
        state = make_state(
            iteration_count=max_iter - 1,
            critique=CritiqueResult(verdict=CriticVerdict.REVISE, summary="Revise."),
        )
        assert _route_after_critic(state) == "supervisor"


# ════════════════════════════════════════════════════════════════════════════════
# build_graph — structural tests
# ════════════════════════════════════════════════════════════════════════════════

class TestBuildGraph:
    """Smoke tests that build_graph returns a compiled graph with the right nodes."""

    def _make_retriever_mock(self):
        return MagicMock()

    def _make_llm_mock(self):
        llm = MagicMock()
        # with_structured_output used by supervisor and critic
        structured = MagicMock()
        structured.ainvoke = AsyncMock(return_value=MagicMock())
        llm.with_structured_output = MagicMock(return_value=structured)
        llm.ainvoke = AsyncMock(return_value=MagicMock(content="draft text"))
        return llm

    def test_build_graph_returns_compiled(self):
        from mia_agents.graph import build_graph
        graph = build_graph(retriever=self._make_retriever_mock(), llm=self._make_llm_mock())
        # CompiledStateGraph has get_graph / ainvoke methods
        assert hasattr(graph, "ainvoke")
        assert hasattr(graph, "invoke")

    def test_graph_has_expected_nodes(self):
        from mia_agents.graph import build_graph
        graph = build_graph(retriever=self._make_retriever_mock(), llm=self._make_llm_mock())
        node_names = set(graph.get_graph().nodes.keys())
        for expected in ("supervisor", "retrieval", "web_search", "edgar_parser",
                          "sql_generator", "summarizer", "critic"):
            assert expected in node_names, f"Missing node: {expected}"

    def test_build_graph_uses_default_llm_when_none(self):
        """build_graph(retriever=...) with no llm arg should call get_llm().

        Patches mia_agents.llm.get_llm — the lazy import inside build_graph
        does ``from mia_agents.llm import get_llm`` at call time, so patching
        the source attribute is the correct target.
        """
        from mia_agents.graph import build_graph
        with patch("mia_agents.llm.get_llm") as mock_get_llm:
            mock_get_llm.return_value = self._make_llm_mock()
            build_graph(retriever=self._make_retriever_mock())
            mock_get_llm.assert_called_once()

    def test_build_graph_uses_provided_llm(self):
        """Explicit llm arg should NOT call get_llm()."""
        from mia_agents.graph import build_graph
        with patch("mia_agents.llm.get_llm") as mock_get_llm:
            build_graph(retriever=self._make_retriever_mock(), llm=self._make_llm_mock())
            mock_get_llm.assert_not_called()


# ════════════════════════════════════════════════════════════════════════════════
# End-to-end graph invocation (fully mocked nodes)
# ════════════════════════════════════════════════════════════════════════════════

class TestGraphEndToEnd:
    """Test full graph runs with mocked node functions.

    We patch the node coroutines so the graph wiring is exercised without
    any real LLM or retriever calls.
    """

    def _build_mocked_graph(
        self,
        supervisor_result=None,
        retrieval_result=None,
        summarizer_result=None,
        critic_result=None,
    ):
        """Build a graph where all node functions are replaced with mocks."""
        from mia_shared.schemas import CritiqueResult

        sup_out = supervisor_result or {"active_agent": AgentName.RETRIEVAL, "plan": "Search."}
        ret_out = retrieval_result or {"evidence": [], "citations": []}
        sum_out = summarizer_result or {"draft": "Revenue grew [1]."}
        crit_out = critic_result or {
            "critique": CritiqueResult(verdict=CriticVerdict.PASS, summary="Good."),
            "iteration_count": 1,
        }

        async def _supervisor(state):
            return sup_out

        async def _retrieval(state):
            return ret_out

        async def _summarizer(state):
            return sum_out

        async def _critic(state):
            return crit_out

        async def _stub(state):
            return {"messages": [], "active_agent": None}

        from langgraph.graph import END, START, StateGraph
        from mia_agents.graph import _route_after_critic, _route_after_supervisor

        graph = StateGraph(AgentState)
        graph.add_node("supervisor", _supervisor)
        graph.add_node("retrieval", _retrieval)
        graph.add_node("web_search", _stub)
        graph.add_node("edgar_parser", _stub)
        graph.add_node("sql_generator", _stub)
        graph.add_node("summarizer", _summarizer)
        graph.add_node("critic", _critic)

        graph.add_edge(START, "supervisor")
        graph.add_conditional_edges(
            "supervisor",
            _route_after_supervisor,
            {"retrieval": "retrieval", "web_search": "web_search",
             "edgar_parser": "edgar_parser", "sql_generator": "sql_generator"},
        )
        for worker in ("retrieval", "web_search", "edgar_parser", "sql_generator"):
            graph.add_edge(worker, "summarizer")
        graph.add_edge("summarizer", "critic")
        graph.add_conditional_edges(
            "critic",
            _route_after_critic,
            {END: END, "supervisor": "supervisor"},
        )

        return graph.compile()

    def test_happy_path_produces_draft(self):
        graph = self._build_mocked_graph()
        result = asyncio.run(graph.ainvoke({"query": "NVDA revenue?"}))
        assert result["draft"] == "Revenue grew [1]."

    def test_happy_path_critique_is_pass(self):
        graph = self._build_mocked_graph()
        result = asyncio.run(graph.ainvoke({"query": "NVDA revenue?"}))
        assert result["critique"].verdict == CriticVerdict.PASS

    def test_query_preserved_in_final_state(self):
        graph = self._build_mocked_graph()
        result = asyncio.run(graph.ainvoke({"query": "AMD margins?"}))
        assert result["query"] == "AMD margins?"

    def test_graph_terminates_on_pass(self):
        """Graph should not loop when critic says PASS."""
        call_count = 0

        async def counting_supervisor(state):
            nonlocal call_count
            call_count += 1
            return {"active_agent": AgentName.RETRIEVAL, "plan": "Search."}

        from langgraph.graph import END, START, StateGraph
        from mia_agents.graph import _route_after_critic, _route_after_supervisor
        from mia_shared.schemas import CritiqueResult

        graph = StateGraph(AgentState)
        graph.add_node("supervisor", counting_supervisor)
        graph.add_node("retrieval", AsyncMock(return_value={"evidence": [], "citations": []}))
        graph.add_node("web_search", AsyncMock(return_value={"messages": [], "active_agent": None}))
        graph.add_node("edgar_parser", AsyncMock(return_value={"messages": [], "active_agent": None}))
        graph.add_node("sql_generator", AsyncMock(return_value={"messages": [], "active_agent": None}))
        graph.add_node("summarizer", AsyncMock(return_value={"draft": "Good draft."}))
        graph.add_node("critic", AsyncMock(return_value={
            "critique": CritiqueResult(verdict=CriticVerdict.PASS, summary="OK"),
            "iteration_count": 1,
        }))

        graph.add_edge(START, "supervisor")
        graph.add_conditional_edges(
            "supervisor", _route_after_supervisor,
            {"retrieval": "retrieval", "web_search": "web_search",
             "edgar_parser": "edgar_parser", "sql_generator": "sql_generator"},
        )
        for w in ("retrieval", "web_search", "edgar_parser", "sql_generator"):
            graph.add_edge(w, "summarizer")
        graph.add_edge("summarizer", "critic")
        graph.add_conditional_edges(
            "critic", _route_after_critic, {END: END, "supervisor": "supervisor"},
        )

        compiled = graph.compile()
        asyncio.run(compiled.ainvoke({"query": "test?"}))
        assert call_count == 1  # supervisor called exactly once

    def test_graph_routes_to_web_search_stub(self):
        """When supervisor picks web_search, graph reaches summarizer via stub."""
        graph = self._build_mocked_graph(
            supervisor_result={"active_agent": AgentName.WEB_SEARCH, "plan": "Search web."}
        )
        result = asyncio.run(graph.ainvoke({"query": "Recent NVDA news?"}))
        # stub doesn't add evidence but graph should still produce a draft
        assert "draft" in result
