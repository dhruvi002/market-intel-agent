"""Phase 6 agent nodes — lazy-loaded to avoid import-time model loading.

Public surface:

    from mia_agents.nodes import (
        supervisor_node,
        retrieval_node,
        summarizer_node,
        critic_node,
        web_search_node,
        edgar_parser_node,
        sql_generator_node,
    )

Phase 6: ``sql_generator_node`` promoted from stub to real NL→SQL worker
in ``nodes/sql_generator.py``.  ``web_search_node`` and ``edgar_parser_node``
were promoted in Phase 5.  ``stubs.py`` is now empty.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mia_agents.nodes.critic import critic_node
    from mia_agents.nodes.edgar_parser import edgar_parser_node
    from mia_agents.nodes.retrieval import retrieval_node
    from mia_agents.nodes.sql_generator import sql_generator_node
    from mia_agents.nodes.summarizer import summarizer_node
    from mia_agents.nodes.supervisor import supervisor_node
    from mia_agents.nodes.web_search import web_search_node

__all__ = [
    "supervisor_node",
    "retrieval_node",
    "summarizer_node",
    "critic_node",
    "web_search_node",
    "edgar_parser_node",
    "sql_generator_node",
]


def __getattr__(name: str):  # noqa: ANN001
    """Lazy attribute loader — defers heavy imports to first access."""
    if name == "supervisor_node":
        from mia_agents.nodes.supervisor import supervisor_node  # noqa: PLC0415

        return supervisor_node
    if name == "retrieval_node":
        from mia_agents.nodes.retrieval import retrieval_node  # noqa: PLC0415

        return retrieval_node
    if name == "summarizer_node":
        from mia_agents.nodes.summarizer import summarizer_node  # noqa: PLC0415

        return summarizer_node
    if name == "critic_node":
        from mia_agents.nodes.critic import critic_node  # noqa: PLC0415

        return critic_node
    if name == "web_search_node":
        from mia_agents.nodes.web_search import web_search_node  # noqa: PLC0415

        return web_search_node
    if name == "edgar_parser_node":
        from mia_agents.nodes.edgar_parser import edgar_parser_node  # noqa: PLC0415

        return edgar_parser_node
    if name == "sql_generator_node":
        from mia_agents.nodes.sql_generator import sql_generator_node  # noqa: PLC0415

        return sql_generator_node
    raise AttributeError(f"module 'mia_agents.nodes' has no attribute {name!r}")
