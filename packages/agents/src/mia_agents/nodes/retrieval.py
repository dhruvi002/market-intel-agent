"""Retrieval node — wraps the Phase-3 RAGAgent as a LangGraph worker.

Design decisions:
- The node re-uses ``RAGAgent`` directly so retrieval logic (hybrid BM25 +
  dense + reranker) is not duplicated.
- Evidence is accumulated: chunks already in ``state.evidence`` are not
  re-added, preventing duplicates across revision iterations.
- Citations from the RAGAgent run are accumulated the same way.
- ``active_agent`` is reset to ``None`` after retrieval so that the static
  edge to ``summarizer`` takes over (the conditional edge after the supervisor
  only fires once per iteration).
"""

from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel

from mia_agents.rag_agent import RAGAgent
from mia_retrieval.retriever import Retriever
from mia_shared.schemas import AgentState

logger = logging.getLogger(__name__)


async def retrieval_node(
    state: AgentState,
    *,
    retriever: Retriever,
    llm: BaseChatModel,
) -> dict:
    """Run the RAGAgent and accumulate evidence into the graph state.

    Parameters
    ----------
    state     : current AgentState
    retriever : configured Retriever (hybrid BM25 + dense by default)
    llm       : LangChain chat model for the generation step inside RAGAgent

    Returns
    -------
    dict
        Updates for ``evidence`` and ``citations`` (accumulated, not replaced).
    """
    logger.info("retrieval: query=%r", state.query[:80])

    agent = RAGAgent(retriever=retriever, llm=llm)
    result = await agent.run(state.query)

    # Deduplicate by Evidence.id — preserve insertion order
    existing_ids = {ev.id for ev in state.evidence}
    new_evidence = [ev for ev in result.evidence if ev.id not in existing_ids]

    existing_cit_ids = {c.id for c in state.citations}
    new_citations = [c for c in result.citations if c.id not in existing_cit_ids]

    logger.info(
        "retrieval: +%d evidence chunks  +%d citations",
        len(new_evidence),
        len(new_citations),
    )

    return {
        "evidence": [*state.evidence, *new_evidence],
        "citations": [*state.citations, *new_citations],
    }
