"""Phase 4 agent nodes — lazy-loaded to avoid import-time model loading.

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
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mia_agents.nodes.critic import critic_node
    from mia_agents.nodes.retrieval import retrieval_node
    from mia_agents.nodes.stubs import edgar_parser_node, sql_generator_node, web_search_node
    from mia_agents.nodes.summarizer import summarizer_node
    from mia_agents.nodes.supervisor import supervisor_node

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
    if name in {"web_search_node", "edgar_parser_node", "sql_generator_node"}:
        from mia_agents.nodes import stubs as _stubs  # noqa: PLC0415

        return getattr(_stubs, name)
    raise AttributeError(f"module 'mia_agents.nodes' has no attribute {name!r}")
