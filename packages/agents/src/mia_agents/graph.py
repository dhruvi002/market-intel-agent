"""Phase 6 LangGraph multi-agent StateGraph.

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

Phase 6 changes
---------------
- ``sql_generator_node`` promoted from stub to real NL→SQL worker
  (imported from ``nodes.sql_generator`` instead of ``nodes.stubs``).
- ``build_graph`` now accepts an optional ``event_callback`` parameter.
  When provided, every node wrapper emits ``AGENT_START`` / ``AGENT_END``
  events, plus sub-events for evidence, draft chunks, and critique.
  When ``event_callback=None`` (default), behaviour is identical to Phase 5.

Key design decisions (unchanged from Phase 5)
--------------------------------------------
- ``AgentState`` (Pydantic BaseModel from mia_shared) is used directly as the
  LangGraph state schema.  LangGraph 1.x supports Pydantic models natively.
- Node dependencies (``retriever``, ``llm``) are injected via closures so
  each node function stays pure and independently testable.
- Routing functions are synchronous — LangGraph requires conditional-edge
  functions to be sync.
- The ``max_iterations`` cap is enforced in ``_route_after_critic``.

Usage
-----
::

    from mia_agents.graph import build_graph
    from mia_retrieval.retriever import Retriever

    # Without streaming
    graph = build_graph(retriever=retriever)
    result = await graph.ainvoke({"query": "What is NVDA data center revenue?"})

    # With streaming (Phase 6)
    async def my_emitter(event: AgentEvent) -> None:
        await redis.publish(f"events:{event.session_id}", event.model_dump_json())

    graph = build_graph(retriever=retriever, event_callback=my_emitter)
    result = await graph.ainvoke({"query": "...", "session_id": session_id})
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from mia_retrieval.retriever import Retriever
from mia_shared.config import get_settings
from mia_shared.schemas import (
    AgentEvent,
    AgentName,
    AgentState,
    CriticVerdict,
    EventType,
)

logger = logging.getLogger(__name__)

# ── Type alias ────────────────────────────────────────────────────────────────

# Async callable that receives an AgentEvent and persists / forwards it
EventCallback = Callable[[AgentEvent], Awaitable[None]]

# ── Node name constants ───────────────────────────────────────────────────────

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


# ── Event-emission helpers ────────────────────────────────────────────────────

async def _emit(
    cb: EventCallback | None,
    event_type: EventType,
    session_id: Any,
    agent: AgentName | None = None,
    payload: dict[str, Any] | None = None,
) -> None:
    """Emit a single event via *cb* if provided; silently no-op otherwise."""
    if cb is None:
        return
    try:
        await cb(
            AgentEvent(
                session_id=session_id,
                event_type=event_type,
                agent=agent,
                payload=payload or {},
            )
        )
    except Exception as exc:  # noqa: BLE001
        # Never let an emitter failure crash the graph
        logger.warning("graph: event emitter raised %s — ignoring", exc)


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph(
    retriever: Retriever,
    llm: BaseChatModel | None = None,
    event_callback: EventCallback | None = None,
) -> Any:  # CompiledStateGraph — typed as Any to avoid circular imports
    """Build and compile the multi-agent LangGraph StateGraph.

    Parameters
    ----------
    retriever       : configured ``Retriever`` (hybrid BM25 + dense by default)
    llm             : LangChain ``BaseChatModel``; defaults to the full
                      Gemini→Groq→Cerebras fallback chain from
                      :func:`~mia_agents.llm.get_llm`
    event_callback  : optional async callable ``(AgentEvent) -> None``.
                      When provided, each node wrapper emits AGENT_START +
                      AGENT_END events, plus EVIDENCE_ADDED / DRAFT_CHUNK /
                      CRITIQUE sub-events.  When ``None``, the graph runs
                      identically to Phase 5.

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
    from mia_agents.nodes.edgar_parser import edgar_parser_node  # noqa: PLC0415
    from mia_agents.nodes.retrieval import retrieval_node  # noqa: PLC0415
    from mia_agents.nodes.sql_generator import sql_generator_node  # noqa: PLC0415
    from mia_agents.nodes.summarizer import summarizer_node  # noqa: PLC0415
    from mia_agents.nodes.supervisor import supervisor_node  # noqa: PLC0415
    from mia_agents.nodes.web_search import web_search_node  # noqa: PLC0415

    cb = event_callback  # local alias for brevity in closures

    # ── Instrumented node closures ─────────────────────────────────────────────
    # Each closure: emit AGENT_START → run real node → emit AGENT_END +
    # domain-specific sub-events.  When cb=None, the _emit calls are no-ops
    # and the closures behave identically to the plain wrappers in Phase 5.

    async def _supervisor(state: AgentState) -> dict:
        await _emit(cb, EventType.AGENT_START, state.session_id, AgentName.SUPERVISOR)
        result = await supervisor_node(state, llm=llm)
        await _emit(
            cb,
            EventType.AGENT_END,
            state.session_id,
            AgentName.SUPERVISOR,
            {"plan": result.get("plan", "")},
        )
        return result

    async def _retrieval(state: AgentState) -> dict:
        await _emit(cb, EventType.AGENT_START, state.session_id, AgentName.RETRIEVAL)
        result = await retrieval_node(state, retriever=retriever, llm=llm)
        await _emit(cb, EventType.AGENT_END, state.session_id, AgentName.RETRIEVAL)
        # Emit each new evidence chunk
        new_ev = result.get("evidence", [])[len(state.evidence):]
        for ev in new_ev:
            await _emit(
                cb,
                EventType.EVIDENCE_ADDED,
                state.session_id,
                AgentName.RETRIEVAL,
                {"evidence": ev.model_dump(mode="json")},
            )
        return result

    async def _web_search(state: AgentState) -> dict:
        await _emit(cb, EventType.AGENT_START, state.session_id, AgentName.WEB_SEARCH)
        result = await web_search_node(state)
        await _emit(cb, EventType.AGENT_END, state.session_id, AgentName.WEB_SEARCH)
        new_ev = result.get("evidence", [])[len(state.evidence):]
        for ev in new_ev:
            await _emit(
                cb,
                EventType.EVIDENCE_ADDED,
                state.session_id,
                AgentName.WEB_SEARCH,
                {"evidence": ev.model_dump(mode="json")},
            )
        return result

    async def _edgar_parser(state: AgentState) -> dict:
        await _emit(cb, EventType.AGENT_START, state.session_id, AgentName.EDGAR_PARSER)
        result = await edgar_parser_node(state)
        await _emit(cb, EventType.AGENT_END, state.session_id, AgentName.EDGAR_PARSER)
        new_ev = result.get("evidence", [])[len(state.evidence):]
        for ev in new_ev:
            await _emit(
                cb,
                EventType.EVIDENCE_ADDED,
                state.session_id,
                AgentName.EDGAR_PARSER,
                {"evidence": ev.model_dump(mode="json")},
            )
        return result

    async def _sql_generator(state: AgentState) -> dict:
        await _emit(cb, EventType.AGENT_START, state.session_id, AgentName.SQL_GENERATOR)
        result = await sql_generator_node(state, llm=llm)
        await _emit(cb, EventType.AGENT_END, state.session_id, AgentName.SQL_GENERATOR)
        new_ev = result.get("evidence", [])[len(state.evidence):]
        for ev in new_ev:
            await _emit(
                cb,
                EventType.EVIDENCE_ADDED,
                state.session_id,
                AgentName.SQL_GENERATOR,
                {"evidence": ev.model_dump(mode="json")},
            )
        return result

    async def _summarizer(state: AgentState) -> dict:
        await _emit(cb, EventType.AGENT_START, state.session_id, AgentName.SUMMARIZER)

        # Phase 7: when a callback is wired, stream tokens individually so the
        # browser receives live DRAFT_CHUNK events.  Each token callback emits
        # one DRAFT_CHUNK with {"chunk": token}.  When cb=None the
        # token_callback is None and summarizer_node falls back to ainvoke.
        async def _token_cb(token: str) -> None:
            await _emit(
                cb,
                EventType.DRAFT_CHUNK,
                state.session_id,
                AgentName.SUMMARIZER,
                {"chunk": token},
            )

        result = await summarizer_node(
            state,
            llm=llm,
            token_callback=_token_cb if cb is not None else None,
        )
        await _emit(cb, EventType.AGENT_END, state.session_id, AgentName.SUMMARIZER)
        # No full-draft DRAFT_CHUNK here — tokens were already emitted above
        # (or cb=None and no events are needed).
        return result

    async def _critic(state: AgentState) -> dict:
        await _emit(cb, EventType.AGENT_START, state.session_id, AgentName.CRITIC)
        result = await critic_node(state)
        await _emit(cb, EventType.AGENT_END, state.session_id, AgentName.CRITIC)
        if critique := result.get("critique"):
            await _emit(
                cb,
                EventType.CRITIQUE,
                state.session_id,
                AgentName.CRITIC,
                {"result": critique.model_dump(mode="json")},
            )
        return result

    # ── Graph construction ────────────────────────────────────────────────────
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node(_SUPERVISOR, _supervisor)
    graph.add_node(_RETRIEVAL, _retrieval)
    graph.add_node(_WEB_SEARCH, _web_search)
    graph.add_node(_EDGAR_PARSER, _edgar_parser)
    graph.add_node(_SQL_GENERATOR, _sql_generator)
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
