"""Async EDGAR filing downloader.

Covers:
  - Ticker → CIK resolution via the EDGAR company tickers JSON
  - Recent filings list via the EDGAR submissions API
  - Primary document download (10-K/10-Q/8-K HTML/PDF)
  - XBRL companyfacts JSON fetch

Rate limited to ≤8 req/sec to stay comfortably under EDGAR's 10 req/sec policy.
Retries on transient HTTP errors via tenacity.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from mia_shared.config import get_settings

logger = logging.getLogger(__name__)

EDGAR_BASE = "https://data.sec.gov"
SEC_BASE = "https://www.sec.gov"

# Module-level headers — built lazily so Settings isn't evaluated at import time
_HEADERS: dict[str, str] | None = None


def _edgar_headers() -> dict[str, str]:
    global _HEADERS
    if _HEADERS is None:
        settings = get_settings()
        _HEADERS = {
            "User-Agent": settings.edgar_user_agent,
            "Accept-Encoding": "gzip, deflate",
        }
    return _HEADERS


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class FilingMeta:
    """Lightweight metadata for one SEC filing."""

    ticker: str
    cik: str
    filing_type: str          # "10-K" | "10-Q" | "8-K"
    accession_number: str     # e.g. "0001045810-23-000017"
    filed_date: Optional[date] = None
    period_of_report: Optional[date] = None
    primary_doc: Optional[str] = None   # filename of primary document


# ── Rate limiter ──────────────────────────────────────────────────────────────

class _RateLimiter:
    """Simple async token-bucket rate limiter."""

    def __init__(self, requests_per_second: float = 8.0) -> None:
        self._min_interval = 1.0 / requests_per_second
        self._last_call: float = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            gap = self._min_interval - (now - self._last_call)
            if gap > 0:
                await asyncio.sleep(gap)
            self._last_call = time.monotonic()


_rate_limiter = _RateLimiter()


# ── Main downloader ───────────────────────────────────────────────────────────

class EDGARDownloader:
    """Async EDGAR client.  Use as an async context manager."""

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            headers=_edgar_headers(),
            timeout=60.0,
            follow_redirects=True,
            # Don't inherit system proxy settings — EDGAR calls must go direct
            trust_env=False,
        )

    async def __aenter__(self) -> "EDGARDownloader":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self._client.aclose()

    # ── Internal HTTP helper ──────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    async def _get(self, url: str, base_headers: dict | None = None) -> httpx.Response:
        await _rate_limiter.acquire()
        headers = base_headers or {}
        resp = await self._client.get(url, headers=headers)
        resp.raise_for_status()
        return resp

    # ── Public API ────────────────────────────────────────────────────────────

    async def get_cik(self, ticker: str) -> str:
        """Resolve a ticker to a zero-padded 10-digit CIK string.

        Uses EDGAR's company_tickers.json mapping (updated nightly by SEC).
        """
        url = f"{SEC_BASE}/files/company_tickers.json"
        resp = await self._get(url)
        data: dict = resp.json()
        ticker_upper = ticker.upper()
        for entry in data.values():
            if entry.get("ticker", "").upper() == ticker_upper:
                cik = str(entry["cik_str"]).zfill(10)
                logger.debug("Resolved %s → CIK %s", ticker, cik)
                return cik
        raise ValueError(
            f"Could not resolve ticker {ticker!r} to a CIK — "
            "check that it is a valid US-exchange ticker listed in SEC EDGAR."
        )

    async def get_recent_filings(
        self,
        cik: str,
        form_type: str,
        limit: int = 5,
    ) -> list[FilingMeta]:
        """Return metadata for the most recent filings of the given form type.

        Calls the EDGAR submissions endpoint which returns the last 1000 filings.
        """
        url = f"{EDGAR_BASE}/submissions/CIK{cik}.json"
        resp = await self._get(url)
        data: dict = resp.json()

        recent: dict = data.get("filings", {}).get("recent", {})
        forms: list[str] = recent.get("form", [])
        accessions: list[str] = recent.get("accessionNumber", [])
        filed_dates: list[str] = recent.get("filingDate", [])
        periods: list[str] = recent.get("reportDate", [])
        primary_docs: list[str] = recent.get("primaryDocument", [])
        tickers: list[str] = data.get("tickers", [])
        ticker = tickers[0] if tickers else ""

        results: list[FilingMeta] = []
        for i, form in enumerate(forms):
            if form == form_type and len(results) < limit:
                results.append(
                    FilingMeta(
                        ticker=ticker,
                        cik=cik,
                        filing_type=form,
                        accession_number=accessions[i] if i < len(accessions) else "",
                        filed_date=_parse_date(filed_dates[i]) if i < len(filed_dates) else None,
                        period_of_report=_parse_date(periods[i]) if i < len(periods) else None,
                        primary_doc=primary_docs[i] if i < len(primary_docs) else None,
                    )
                )
        logger.debug(
            "Found %d %s filings for CIK %s (limit %d)", len(results), form_type, cik, limit
        )
        return results

    async def download_filing_document(
        self,
        cik: str,
        accession_number: str,
        primary_doc: str,
        download_dir: Path,
    ) -> Path:
        """Download the primary document of a filing.

        Returns the local path to the downloaded file.

        EDGAR filing URLs follow the pattern:
          https://www.sec.gov/Archives/edgar/data/{cik}/{clean_accession}/{filename}
        """
        clean_acc = accession_number.replace("-", "")
        cik_int = int(cik)  # strip leading zeros for the URL path
        url = f"{SEC_BASE}/Archives/edgar/data/{cik_int}/{clean_acc}/{primary_doc}"

        # SEC's Archives endpoint uses a different Host header
        resp = await self._get(
            url,
            base_headers={"Host": "www.sec.gov"},
        )
        download_dir.mkdir(parents=True, exist_ok=True)
        local_path = download_dir / primary_doc
        local_path.write_bytes(resp.content)
        logger.info(
            "Downloaded %s/%s → %s (%d bytes)",
            accession_number,
            primary_doc,
            local_path,
            len(resp.content),
        )
        return local_path

    async def get_xbrl_facts(self, cik: str) -> dict:
        """Fetch the full XBRL companyfacts JSON for a company.

        This single call returns all historically reported XBRL facts across
        every filed form — typically 10-100k entries depending on company history.
        """
        url = f"{EDGAR_BASE}/api/xbrl/companyfacts/CIK{cik}.json"
        resp = await self._get(url)
        return resp.json()


# ── Utility ───────────────────────────────────────────────────────────────────

def _parse_date(s: str | None) -> Optional[date]:
    if not s:
        return None
    try:
        return date.fromisoformat(s)
    except ValueError:
        return None
