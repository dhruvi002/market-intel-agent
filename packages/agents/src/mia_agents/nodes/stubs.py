"""Stub worker nodes — placeholders for Phase 5+ workers.

These nodes add a message to the state indicating they are not yet implemented,
then pass control forward to the Summarizer.  Stubs are valid LangGraph nodes:
they participate in routing, logging, and state updates — they simply do not
yet fetch external data.

Implementation order (planned):
- Phase 5: web_search_node — Tavily search + snippet retrieval
- Phase 5: edgar_parser_node — direct EDGAR accession-number fetch
- Phase 6: sql_generator_node — NL→SQL against the Postgres metrics schema
"""

from __future__ import annotations

import logging

from mia_shared.schemas import AgentName, AgentState

logger = logging.getLogger(__name__)

_NOT_IMPLEMENTED_MSG = (
    "This worker is not yet implemented (Phase 4 stub). "
    "Evidence will be sourced from the retrieval worker instead."
)


def _stub_message(agent: AgentName) -> dict:
    """Return a state-update dict that logs a stub message for *agent*."""
    logger.info("stub: %s called — not yet implemented", agent.value)
    return {
        "messages": [
            {
                "role": "system",
                "content": f"[{agent.value}] {_NOT_IMPLEMENTED_MSG}",
                "agent": agent.value,
            }
        ],
        # Reset active_agent so routing falls through correctly
        "active_agent": None,
    }


async def web_search_node(state: AgentState) -> dict:
    """Stub: live web search via Tavily (Phase 5).

    Will retrieve recent news, analyst reports, and post-filing data.
    """
    return _stub_message(AgentName.WEB_SEARCH)


async def edgar_parser_node(state: AgentState) -> dict:
    """Stub: direct EDGAR filing fetch by accession number (Phase 5).

    Will support "show me the actual 10-K for NVDA Q4 2024" queries.
    """
    return _stub_message(AgentName.EDGAR_PARSER)


async def sql_generator_node(state: AgentState) -> dict:
    """Stub: NL→SQL query against the structured metrics database (Phase 6).

    Will handle precise numerical comparisons and ranking queries.
    """
    return _stub_message(AgentName.SQL_GENERATOR)
