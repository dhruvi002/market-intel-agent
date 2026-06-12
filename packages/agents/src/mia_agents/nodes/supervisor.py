"""Supervisor node — routes queries to the most appropriate worker agent.

The supervisor uses ``with_structured_output`` to get a typed routing decision
from the LLM rather than parsing free-text. This is more robust to output
variation across providers (Gemini, Groq, Cerebras).

Design decisions:
- Structured output via ``SupervisorDecision`` Pydantic model ensures the
  routing key is always a valid ``AgentName``.
- On revision iterations the critic's failing claims are fed back as an
  additional HumanMessage so the supervisor can try a different worker or
  refine the plan.
- If the LLM returns an unknown or non-routable agent name, the supervisor
  falls back to RETRIEVAL rather than crashing.
- ``active_agent`` is set in state so the graph's conditional edge can read it
  without additional LLM calls.
"""

from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from mia_shared.schemas import AgentName, AgentState

logger = logging.getLogger(__name__)

# Agents the supervisor is allowed to route to (stubs are valid routing targets)
_ROUTABLE: frozenset[AgentName] = frozenset(
    {
        AgentName.RETRIEVAL,
        AgentName.WEB_SEARCH,
        AgentName.EDGAR_PARSER,
        AgentName.SQL_GENERATOR,
    }
)


class SupervisorDecision(BaseModel):
    """Structured output contract for the Supervisor LLM call."""

    next_agent: str = Field(
        description="One of: retrieval, web_search, edgar_parser, sql_generator"
    )
    reasoning: str = Field(description="One sentence explaining the choice")
    plan: str = Field(description="Brief instruction for the chosen agent")


async def supervisor_node(state: AgentState, *, llm: BaseChatModel) -> dict:
    """Analyse the query and route to the most appropriate worker.

    Parameters
    ----------
    state : current AgentState — reads ``query``, ``iteration_count``, ``critique``
    llm   : bound LangChain chat model

    Returns
    -------
    dict
        Updates for ``active_agent`` and ``plan``.
    """
    from mia_agents.prompts import SUPERVISOR_SYSTEM  # noqa: PLC0415

    logger.info(
        "supervisor: routing query=%r  iteration=%d",
        state.query[:80],
        state.iteration_count,
    )

    structured_llm = llm.with_structured_output(SupervisorDecision)

    messages: list = [
        SystemMessage(content=SUPERVISOR_SYSTEM),
        HumanMessage(content=f"Query: {state.query}"),
    ]

    # On revision loops, feed critic feedback back so supervisor can adapt
    if state.critique is not None and state.iteration_count > 0:
        failing_lines = "\n".join(
            f"  • {fc.claim}: {fc.reason}" for fc in state.critique.failing_claims
        )
        messages.append(
            HumanMessage(
                content=(
                    f"Critic verdict after iteration {state.iteration_count}: "
                    f"{state.critique.verdict.value}\n"
                    f"Summary: {state.critique.summary}\n"
                    f"Failing claims:\n{failing_lines or '  (none listed)'}\n\n"
                    "Please re-route to gather better or additional evidence."
                )
            )
        )

    decision: SupervisorDecision = await structured_llm.ainvoke(messages)

    # Validate and normalise — guard against hallucinated agent names
    try:
        next_agent = AgentName(decision.next_agent)
    except ValueError:
        logger.warning(
            "supervisor: LLM returned unknown agent %r — defaulting to retrieval",
            decision.next_agent,
        )
        next_agent = AgentName.RETRIEVAL

    if next_agent not in _ROUTABLE:
        logger.warning(
            "supervisor: non-routable agent %r — defaulting to retrieval",
            next_agent,
        )
        next_agent = AgentName.RETRIEVAL

    logger.info(
        "supervisor: → %s | %s", next_agent.value, decision.reasoning
    )

    return {
        "active_agent": next_agent,
        "plan": decision.plan,
    }
