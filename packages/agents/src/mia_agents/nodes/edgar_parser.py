"""EDGAR-parser worker node — direct EDGAR filing retrieval.

Uses the SEC's free EFTS (full-text search) API to locate filings that
match the query, then returns highlighted snippets as ``Evidence`` objects
with ``source_type="edgar_filing"``.

EDGAR fair-access policy
------------------------
All requests include the ``User-Agent`` header required by SEC fair-access
rules (``settings.edgar_user_agent``), and a minimum inter-request delay
(``settings.edgar_request_delay_s``, default 0.11 s) is applied before
every call to stay well under the 10 req/s rate limit.

Ticker extraction
-----------------
The node attempts to extract a ticker symbol (1–5 uppercase letters) from
the query.  Common English words that happen to match the pattern (A, I,
THE, …) are excluded via a skip-list.  If no ticker is found the raw query
is used as the EFTS search term.

EFTS API
--------
Endpoint: ``https://efts.sec.gov/LATEST/search-index``
Relevant query parameters:

- ``q``     : quoted search term
- ``forms`` : comma-separated form types (10-K, 10-Q, 8-K)

Response shape::

    {
      "hits": {
        "hits": [
          {
            "_source": {
              "entity_id": "...",
              "entity_name": "...",
              "form_type": "10-K",
              "period_of_report": "2024-01-31"
            },
            "highlight": {
              "<field>": ["...snippet..."]
            }
          }
        ]
      }
    }

Design decisions
----------------
- At most ``_MAX_HITS`` filing hits are converted to Evidence, capping
  context growth per call.
- ``highlight`` fields supply the actual text snippets; if none are
  present the period-of-report string is used as a minimal fallback.
- Any HTTP or JSON error is caught and logged; the node returns the
  unmodified evidence list so the graph can continue.
"""

from __future__ import annotations

import asyncio
import logging
import re
from urllib.parse import quote_plus

import httpx

from mia_shared.config import get_settings
from mia_shared.schemas import AgentState, Evidence

logger = logging.getLogger(__name__)

_EFTS_URL = "https://efts.sec.gov/LATEST/search-index"
_MAX_HITS: int = 3
_FORMS = "10-K,10-Q,8-K"

# Uppercase words to skip when looking for tickers
_SKIP_WORDS: frozenset[str] = frozenset({
    "A", "AN", "THE", "AND", "OR", "IN", "OF", "FOR", "IS", "ARE", "WAS",
    "TO", "AT", "BY", "IT", "ITS", "BE", "AS", "IF", "ON", "DO", "NO",
    "US", "ME", "MY", "HE", "SHE", "WE", "RE", "AM",
})

_TICKER_RE = re.compile(r"\b([A-Z]{1,5})\b")


def _extract_ticker(query: str) -> str | None:
    """Return the first token that looks like a ticker symbol."""
    for match in _TICKER_RE.finditer(query):
        token = match.group(1)
        if token not in _SKIP_WORDS and 2 <= len(token) <= 5:
            return token
    return None


async def edgar_parser_node(state: AgentState) -> dict:
    """Fetch EDGAR filings matching the query and accumulate as evidence.

    Parameters
    ----------
    state : current AgentState — reads ``query`` and ``evidence``

    Returns
    -------
    dict
        Updates for ``evidence`` (accumulated) and ``citations`` (unchanged).
    """
    settings = get_settings()

    # Respect EDGAR's fair-access rate limit before every call
    await asyncio.sleep(settings.edgar_request_delay_s)

    ticker = _extract_ticker(state.query)
    search_term = ticker if ticker else state.query[:100]
    logger.info(
        "edgar_parser: searching EFTS for %r (from query %r)",
        search_term,
        state.query[:60],
    )

    params = {
        "q": f'"{search_term}"',
        "forms": _FORMS,
    }
    headers = {
        "User-Agent": settings.edgar_user_agent,
        "Accept": "application/json",
    }

    try:
        async with httpx.AsyncClient(headers=headers, timeout=15.0) as client:
            resp = await client.get(_EFTS_URL, params=params)
            resp.raise_for_status()
            data: dict = resp.json()
    except httpx.HTTPStatusError as exc:
        logger.warning(
            "edgar_parser: EDGAR returned HTTP %d — returning empty results",
            exc.response.status_code,
        )
        return {"evidence": state.evidence, "citations": state.citations}
    except Exception as exc:  # noqa: BLE001
        logger.warning("edgar_parser: request failed (%s) — returning empty results", exc)
        return {"evidence": state.evidence, "citations": state.citations}

    hits: list[dict] = data.get("hits", {}).get("hits", [])[:_MAX_HITS]
    logger.info("edgar_parser: received %d hit(s)", len(hits))

    existing_urls: set[str] = {
        ev.source_url for ev in state.evidence if ev.source_url
    }

    new_evidence: list[Evidence] = []
    for hit in hits:
        src: dict = hit.get("_source", {})
        entity_id: str = src.get("entity_id", "")

        # Build a canonical URL for this filing's EDGAR page
        source_url = (
            f"https://www.sec.gov/cgi-bin/browse-edgar"
            f"?action=getcompany&CIK={entity_id}&type={src.get('form_type', '')}"
            if entity_id
            else None
        )

        if source_url and source_url in existing_urls:
            logger.debug("edgar_parser: skipping duplicate filing URL %s", source_url)
            continue

        # Collect highlighted text snippets from all highlight fields
        highlights: dict = hit.get("highlight", {})
        snippets: list[str] = []
        for field_hits in highlights.values():
            if isinstance(field_hits, list):
                snippets.extend(field_hits)

        text = " … ".join(snippets) if snippets else src.get("period_of_report", "")
        if not text:
            continue

        new_evidence.append(
            Evidence(
                source_type="edgar_filing",
                source_url=source_url,
                ticker=ticker,
                filing_type=src.get("form_type"),
                section=src.get("period_of_report"),
                text=text,
                metadata={
                    "entity_name": src.get("entity_name", ""),
                    "entity_id": entity_id,
                },
            )
        )
        if source_url:
            existing_urls.add(source_url)

    logger.info("edgar_parser: +%d new evidence chunk(s)", len(new_evidence))
    return {
        "evidence": [*state.evidence, *new_evidence],
        "citations": state.citations,
    }
