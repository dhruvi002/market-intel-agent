"""ARQ task definitions for the ingestion pipeline.

Tasks registered here can be enqueued from the API via:
    await arq_pool.enqueue_job("ingest_ticker", ticker="NVDA")
    await arq_pool.enqueue_job("ingest_filing", ticker="NVDA", accession_number="...")
"""

from __future__ import annotations

import logging

from mia_ingestion.pipeline import IngestionPipeline

logger = logging.getLogger(__name__)


async def ingest_ticker(
    ctx: dict,
    ticker: str,
    form_types: list[str] | None = None,
) -> dict:
    """Ingest all recent filings + XBRL facts for a ticker.

    Args:
        ctx:        ARQ context dict (contains redis pool, settings, etc.)
        ticker:     US exchange ticker symbol, e.g. "NVDA"
        form_types: Subset of ["10-K", "10-Q", "8-K"] to fetch.
                    Defaults to all three.

    Returns:
        {"ticker": str, "filing_count": int, "filing_ids": list[str]}
    """
    logger.info("ingest_ticker started: ticker=%s form_types=%s", ticker, form_types)
    pipeline = IngestionPipeline()
    filing_ids = await pipeline.ingest_ticker(ticker, form_types=form_types)
    result = {
        "ticker": ticker,
        "filing_count": len(filing_ids),
        "filing_ids": [str(fid) for fid in filing_ids],
    }
    logger.info("ingest_ticker done: %s", result)
    return result


async def ingest_filing(
    ctx: dict,
    ticker: str,
    accession_number: str,
) -> dict:
    """Ingest a single filing identified by its EDGAR accession number.

    Args:
        ctx:               ARQ context dict
        ticker:            US exchange ticker symbol
        accession_number:  EDGAR accession number, e.g. "0001045810-23-000017"

    Returns:
        {"ticker": str, "accession_number": str, "filing_id": str | None}
    """
    logger.info(
        "ingest_filing started: ticker=%s accession=%s", ticker, accession_number
    )
    pipeline = IngestionPipeline()
    filing_id = await pipeline.ingest_filing(ticker, accession_number)
    result = {
        "ticker": ticker,
        "accession_number": accession_number,
        "filing_id": str(filing_id) if filing_id else None,
    }
    logger.info("ingest_filing done: %s", result)
    return result
