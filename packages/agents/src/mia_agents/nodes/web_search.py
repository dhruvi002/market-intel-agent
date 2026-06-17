"""Web-search worker node — live web retrieval via Tavily.

Fetches up to ``_MAX_RESULTS`` search results for the current query and
converts each result into an ``Evidence`` object with
``source_type="web"``.  Results are deduplicated against URLs already
present in ``state.evidence`` so that a second call on the same query
(e.g. during a REVISE loop) does not add duplicates.

Design decisions
----------------
- ``AsyncTavilyClient`` is instantiated fresh per call; Tavily's client is
  lightweight and stateless, so there is no benefit to a singleton here.
- ``search_depth="basic"`` minimises latency and API quota consumption.
  Switch to ``"advanced"`` for richer snippets if quota allows.
- Empty ``content`` results (Tavily occasionally returns metadata-only
  hits) are silently skipped.
- Any exception from the Tavily API is caught and logged; the node returns
  the unmodified evidence list rather than crashing the graph.
"""

from __future__ import annotations

import logging

from mia_shared.config import get_settings
from mia_shared.schemas import AgentState, Evidence

logger = logging.getLogger(__name__)

_MAX_RESULTS: int = 5


async def web_search_node(state: AgentState) -> dict:
    """Search the live web for the current query and accumulate results.

    Parameters
    ----------
    state : current AgentState — reads ``query`` and ``evidence``

    Returns
    -------
    dict
        Updates for ``evidence`` (accumulated) and ``citations`` (unchanged).
    """
    from tavily import AsyncTavilyClient  # noqa: PLC0415

    settings = get_settings()
    logger.info("web_search: querying Tavily for %r", state.query[:80])

    try:
        client = AsyncTavilyClient(
            api_key=settings.tavily_api_key.get_secret_value()
        )
        response: dict = await client.search(
            state.query,
            max_results=_MAX_RESULTS,
            search_depth="basic",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("web_search: Tavily request failed (%s) — returning empty results", exc)
        return {
            "evidence": state.evidence,
            "citations": state.citations,
        }

    results: list[dict] = response.get("results", [])
    logger.info("web_search: received %d result(s)", len(results))

    # Deduplication: skip URLs already in state.evidence
    existing_urls: set[str] = {
        ev.source_url for ev in state.evidence if ev.source_url
    }

    new_evidence: list[Evidence] = []
    for result in results:
        url: str = result.get("url", "")
        content: str = result.get("content", "")
        if not content:
            continue
        if url and url in existing_urls:
            logger.debug("web_search: skipping duplicate URL %s", url)
            continue
        new_evidence.append(
            Evidence(
                source_type="web",
                source_url=url or None,
                text=content,
                relevance_score=result.get("score"),
                metadata={"title": result.get("title", "")},
            )
        )
        if url:
            existing_urls.add(url)

    logger.info("web_search: +%d new evidence chunk(s)", len(new_evidence))
    return {
        "evidence": [*state.evidence, *new_evidence],
        "citations": state.citations,
    }
