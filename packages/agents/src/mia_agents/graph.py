"""Phase 4 LangGraph multi-agent StateGraph.

Topology
--------
::

    START
      │
      ▼
  supervisor ──────────────────────────────────────────────────────┐
      │                                                             │ (REVISE +
      │ active_agent routing                                        │  iteration < max)
      ▼                                                             │
  ┌──────────────────────────────────────────────────┐             │
  │  retrieval │ web_search │ edgar_parser │ sql_gen  │             │
  └──────────────────────────────────────────────────┘             │
      │ (all workers converge here)                                 │
      ▼                                                             │
  summarizer                                                        │
      │                                                             │
      ▼                                                             │
   critic ──────────────────────────────────────────────────────────┘
      │
      │ PASS or ESCALATE or iteration >= max_iterations
      ▼
    END

Key design decisions
--------------------
- ``AgentState`` (Pydantic BaseModel from mia_shared) is used directly as the
  LangGraph state schema.  LangGraph 1.x supports Pydantic models natively.
- Node dependencies (``retriever``, ``llm``) are injected via closures so
  each node function stays pure and independently testable.
- Routing functions are synchronous — LangGraph requires conditional-edge
  functions to be sync.
- The ``max_iterations`` cap is enforced in ``_route_after_critic``; the Critic
  increments ``iteration_count`` so the cap fires even on ESCALATE.
- All stub workers (web_search, edgar_parser, sql_generator) share a single
  converging edge to ``summarizer`` — adding a real implementation later only
  requires changing the stub function body, not the graph topology.

Usage
-----
::

    from mia_agents.graph import build_graph
    from mia_retrieval.retriever import Retriever

    retriever = Retriever(...)
    graph = build_graph(retriever=retriever)
    result = await graph.ainvoke({"query": "What is NVDA data center revenue?"})
    # result is an AgentState dict with draft, evidence, citations, critique
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Any

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from mia_retrieval.retriever import Retriever
from mia_shared.config import get_settings
from mia_shared.schemas import AgentName, AgentState, CriticVerdict

logger = logging.getLogger(__name__)

# Node name constants — single source of truth to avoid typo-bugs in edge wiring
_SUPERVISOR = "supervisor"
_RETRIEVAL = "retrieval"
_WEB_SEARCH = "web_search"
_EDGAR_PARSER = "edgar_parser"
_SQL_GENERATOR = "sql_generator"
_SUMMARIZER = "summarizer"
_CRITIC = "critic"

# Map AgentName enum values → graph node names
_AGENT_TO_NODE: dict[AgentName, str] = {
    AgentName.RETRIEVAL: _RETRIEVAL,
    AgentName.WEB_SEARCH: _WEB_SEARCH,
    AgentName.EDGAR_PARSER: _EDGAR_PARSER,
    AgentName.SQL_GENERATOR: _SQL_GENERATOR,
}


# ── Routing functions (must be sync for LangGraph conditional edges) ──────────

def _route_after_supervisor(state: AgentState) -> str:
    """Return the worker node name based on supervisor's ``active_agent`` choice."""
    agent = state.active_agent
    if agent is None or agent not in _AGENT_TO_NODE:
        logger.warning(
            "graph: supervisor set active_agent=%r — defaulting to retrieval", agent
        )
        return _RETRIEVAL
    return _AGENT_TO_NODE[agent]


def _route_after_critic(state: AgentState) -> str:
    """Return END or supervisor for another retrieval pass."""
    settings = get_settings()

    if state.iteration_count >= settings.max_iterations:
        logger.info(
            "graph: iteration cap reached (%d/%d) — terminating",
            state.iteration_count,
            settings.max_iterations,
        )
        return END

    if state.critique is None:
        return END

    verdict = state.critique.verdict
    if verdict in (CriticVerdict.PASS, CriticVerdict.ESCALATE):
        logger.info("graph: critic verdict=%s — terminating", verdict.value)
        return END

    # REVISE → send back to supervisor for another worker pass
    logger.info(
        "graph: critic verdict=revise — looping back (iteration %d)",
        state.iteration_count,
    )
    return _SUPERVISOR


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph(
    retriever: Retriever,
    llm: BaseChatModel | None = None,
) -> Any:  # CompiledStateGraph — typed as Any to avoid circular imports
    """Build and compile the multi-agent LangGraph StateGraph.

    Parameters
    ----------
    retriever : configured ``Retriever`` (hybrid BM25 + dense by default)
    llm       : LangChain ``BaseChatModel``; defaults to the full Gemini→Groq→
                Cerebras fallback chain from :func:`~mia_agents.llm.get_llm`

    Returns
    -------
    CompiledStateGraph
        Ready to invoke with ``graph.invoke({"query": "..."})`` or
        ``await graph.ainvoke({"query": "..."})``.
    """
    if llm is None:
        from mia_agents.llm import get_llm  # noqa: PLC0415

        llm = get_llm()

    # Import nodes lazily — avoids torch/transformers at module import time
    from mia_agents.nodes.critic import critic_node  # noqa: PLC0415
    from mia_agents.nodes.retrieval import retrieval_node  # noqa: PLC0415
    from mia_agents.nodes.stubs import (  # noqa: PLC0415
        edgar_parser_node,
        sql_generator_node,
        web_search_node,
    )
    from mia_agents.nodes.summarizer import summarizer_node  # noqa: PLC0415
    from mia_agents.nodes.supervisor import supervisor_node  # noqa: PLC0415

    # Bind dependencies via async closures — keeps node functions pure/testable
    async def _supervisor(state: AgentState) -> dict:
        return await supervisor_node(state, llm=llm)

    async def _retrieval(state: AgentState) -> dict:
        return await retrieval_node(state, retriever=retriever, llm=llm)

    async def _summarizer(state: AgentState) -> dict:
        return await summarizer_node(state, llm=llm)

    async def _critic(state: AgentState) -> dict:
        return await critic_node(state, llm=llm)

    # ── Graph construction ────────────────────────────────────────────────────
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node(_SUPERVISOR, _supervisor)
    graph.add_node(_RETRIEVAL, _retrieval)
    graph.add_node(_WEB_SEARCH, web_search_node)
    graph.add_node(_EDGAR_PARSER, edgar_parser_node)
    graph.add_node(_SQL_GENERATOR, sql_generator_node)
    graph.add_node(_SUMMARIZER, _summarizer)
    graph.add_node(_CRITIC, _critic)

    # Entry point
    graph.add_edge(START, _SUPERVISOR)

    # Supervisor → worker (conditional on active_agent)
    graph.add_conditional_edges(
        _SUPERVISOR,
        _route_after_supervisor,
        {
            _RETRIEVAL: _RETRIEVAL,
            _WEB_SEARCH: _WEB_SEARCH,
            _EDGAR_PARSER: _EDGAR_PARSER,
            _SQL_GENERATOR: _SQL_GENERATOR,
        },
    )

    # All workers converge on summarizer
    for worker in (_RETRIEVAL, _WEB_SEARCH, _EDGAR_PARSER, _SQL_GENERATOR):
        graph.add_edge(worker, _SUMMARIZER)

    # Summarizer → critic (always)
    graph.add_edge(_SUMMARIZER, _CRITIC)

    # Critic → END or back to supervisor (conditional on verdict)
    graph.add_conditional_edges(
        _CRITIC,
        _route_after_critic,
        {
            END: END,
            _SUPERVISOR: _SUPERVISOR,
        },
    )

    return graph.compile()
