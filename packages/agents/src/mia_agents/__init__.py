"""mia_agents — LangGraph multi-agent orchestration package.

Public API (lazy-loaded to avoid importing torch/transformers on package import):

Phase 3 — RAG baseline:
    from mia_agents import RAGAgent, RAGResponse, get_llm, LLMProvider

Phase 4 — Multi-agent graph:
    from mia_agents import build_graph

Heavy dependencies (sentence-transformers, torch) live in mia_retrieval and are
not imported until the Retriever is constructed.  LLM SDKs (langchain_google_genai,
langchain_groq) are loaded lazily inside llm.py._build_single().
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mia_agents.graph import build_graph
    from mia_agents.llm import LLMProvider, get_llm
    from mia_agents.prompts import (
        RAG_PROMPT,
        SUMMARIZER_PROMPT,
        build_critic_messages,
        build_rag_messages,
        build_summarizer_messages,
    )
    from mia_agents.rag_agent import RAGAgent, RAGResponse

__all__ = [
    # Phase 3
    "RAGAgent",
    "RAGResponse",
    "get_llm",
    "LLMProvider",
    "RAG_PROMPT",
    "build_rag_messages",
    # Phase 4
    "build_graph",
    "SUMMARIZER_PROMPT",
    "build_summarizer_messages",
    "build_critic_messages",
]


def __getattr__(name: str):  # noqa: ANN001
    """Lazy attribute loader — defers expensive imports to first access."""
    if name in {"get_llm", "LLMProvider"}:
        from mia_agents import llm as _llm  # noqa: PLC0415

        return getattr(_llm, name)
    if name in {
        "RAG_PROMPT",
        "SUMMARIZER_PROMPT",
        "build_rag_messages",
        "build_summarizer_messages",
        "build_critic_messages",
    }:
        from mia_agents import prompts as _prompts  # noqa: PLC0415

        return getattr(_prompts, name)
    if name in {"RAGAgent", "RAGResponse"}:
        from mia_agents import rag_agent as _ra  # noqa: PLC0415

        return getattr(_ra, name)
    if name == "build_graph":
        from mia_agents.graph import build_graph  # noqa: PLC0415

        return build_graph
    raise AttributeError(f"module 'mia_agents' has no attribute {name!r}")
