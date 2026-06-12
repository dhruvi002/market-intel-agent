#!/usr/bin/env python
"""CLI: index one or more tickers into Qdrant + BM25.

Usage
-----
    python scripts/index_ticker.py NVDA AAPL MSFT
    python scripts/index_ticker.py NVDA --forms 10-K 10-Q
    python scripts/index_ticker.py NVDA --force

Reads filings from ``mia.filings`` (Postgres) where status='indexed' and
raw_text is not NULL, then runs the IndexingPipeline to upsert chunks into
Qdrant and rebuild the BM25 index.

Prerequisites
-------------
    make up-infra   # Postgres + Qdrant must be running
    make migrate    # mia.filings table must exist
    make ingest ticker=NVDA  # filings must be ingested first (Phase 1)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

_BM25_PATH = Path("data/bm25_index.pkl")


async def _run(tickers: list[str], forms: list[str], force: bool) -> None:
    from sqlalchemy import select

    from mia_ingestion.db import get_db_session
    from mia_ingestion.models import Filing
    from mia_retrieval.bm25_index import BM25Index
    from mia_retrieval.embedder import get_embedder
    from mia_retrieval.indexer import FilingRecord, IndexingPipeline
    from mia_retrieval.qdrant_store import QdrantStore
    from mia_shared.config import get_settings

    cfg = get_settings()

    # Load (or start fresh) BM25 index
    bm25 = BM25Index()
    if _BM25_PATH.exists():
        bm25 = BM25Index.load(_BM25_PATH)
        logger.info("Loaded existing BM25 index (%d chunks)", bm25.size)
    else:
        logger.info("No existing BM25 index — starting fresh")

    qdrant = QdrantStore(url=cfg.qdrant_url, collection=cfg.qdrant_collection)
    embedder = get_embedder(model_name=cfg.embedding_model)
    pipeline = IndexingPipeline(
        qdrant=qdrant,
        embedder=embedder,
        bm25=bm25,
        bm25_path=_BM25_PATH,
    )

    await pipeline.ensure_ready()

    for ticker in tickers:
        ticker = ticker.upper()
        logger.info("Loading filings for %s from Postgres …", ticker)

        async with get_db_session() as session:
            stmt = (
                select(Filing)
                .where(
                    Filing.ticker == ticker,
                    Filing.status == "indexed",
                    Filing.raw_text.isnot(None),  # type: ignore[arg-type]
                )
            )
            if forms:
                stmt = stmt.where(Filing.filing_type.in_(forms))  # type: ignore[attr-defined]

            result = await session.execute(stmt)
            filings = result.scalars().all()

        logger.info("Found %d filings for %s", len(filings), ticker)
        if not filings:
            logger.warning("No indexed filings found for %s — run `make ingest ticker=%s` first", ticker, ticker)
            continue

        records = [
            FilingRecord(
                id=str(f.id),
                ticker=f.ticker,
                filing_type=f.filing_type,
                accession_number=f.accession_number,
                raw_text=f.raw_text,
            )
            for f in filings
        ]

        stats = await pipeline.index_ticker(ticker=ticker, filings=records, force=force)

        logger.info(
            "%s done: %d processed, %d skipped, %d empty, %d chunks indexed, %d errors",
            ticker,
            stats.filings_processed,
            stats.filings_skipped,
            stats.filings_empty,
            stats.chunks_upserted,
            len(stats.errors),
        )
        for err in stats.errors:
            logger.error("  %s", err)


def main() -> None:
    parser = argparse.ArgumentParser(description="Index SEC filings into Qdrant + BM25")
    parser.add_argument("tickers", nargs="+", help="Ticker symbols to index (e.g. NVDA AAPL)")
    parser.add_argument(
        "--forms",
        nargs="*",
        default=["10-K", "10-Q", "8-K"],
        help="Filing form types to include (default: 10-K 10-Q 8-K)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-index even if filings are already in Qdrant",
    )
    args = parser.parse_args()

    try:
        asyncio.run(_run(args.tickers, args.forms, args.force))
    except KeyboardInterrupt:
        logger.info("Interrupted")
        sys.exit(0)


if __name__ == "__main__":
    main()
