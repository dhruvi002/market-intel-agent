#!/usr/bin/env python3
"""CLI script to trigger ingestion for one or more tickers without running the ARQ worker.

Useful for:
  - Manual ingestion during development
  - One-off data loading
  - Debugging a specific ticker

Usage (from project root):
    uv run python scripts/ingest_ticker.py NVDA
    uv run python scripts/ingest_ticker.py NVDA AAPL MSFT
    uv run python scripts/ingest_ticker.py NVDA --forms 10-K
    uv run python scripts/ingest_ticker.py NVDA --forms 10-K 10-Q
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)


async def run(tickers: list[str], form_types: list[str] | None) -> None:
    from mia_ingestion.pipeline import IngestionPipeline

    pipeline = IngestionPipeline()
    for ticker in tickers:
        print(f"\n{'─'*60}")
        print(f"Ingesting {ticker.upper()}")
        print(f"{'─'*60}")
        filing_ids = await pipeline.ingest_ticker(ticker.upper(), form_types=form_types)
        print(f"✓ {ticker.upper()}: {len(filing_ids)} filing(s) ingested")
        for fid in filing_ids:
            print(f"  {fid}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest SEC filings for one or more tickers")
    parser.add_argument("tickers", nargs="+", help="Ticker symbols, e.g. NVDA AAPL")
    parser.add_argument(
        "--forms",
        nargs="+",
        metavar="FORM",
        choices=["10-K", "10-Q", "8-K"],
        default=None,
        help="Form types to ingest (default: 10-K 10-Q 8-K)",
    )
    args = parser.parse_args()
    asyncio.run(run(args.tickers, args.forms))


if __name__ == "__main__":
    main()
