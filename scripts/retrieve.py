#!/usr/bin/env python
"""CLI: test retrieval against the indexed corpus.

Usage
-----
    python scripts/retrieve.py "How is NVDA's data center revenue growing?"
    python scripts/retrieve.py "AMD competitive risk factors" --mode bm25 --no-rerank
    python scripts/retrieve.py "NVDA cash position" --tickers NVDA --top-k 5

Prints results as formatted text with source metadata.

Prerequisites
-------------
    make up-infra
    make ingest ticker=NVDA
    make index ticker=NVDA
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

_BM25_PATH = Path("data/bm25_index.pkl")


async def _run(
    query: str,
    mode: str,
    top_k: int,
    rerank: bool,
    tickers: list[str],
) -> None:
    from mia_retrieval.bm25_index import BM25Index
    from mia_retrieval.embedder import get_embedder
    from mia_retrieval.qdrant_store import QdrantStore
    from mia_retrieval.reranker import get_reranker
    from mia_retrieval.retriever import RetrieveMode, Retriever
    from mia_shared.config import get_settings

    cfg = get_settings()

    bm25 = BM25Index()
    if _BM25_PATH.exists():
        bm25 = BM25Index.load(_BM25_PATH)
    elif mode in ("bm25", "hybrid"):
        logger.warning("No BM25 index found at %s — BM25 results will be empty", _BM25_PATH)

    retriever = Retriever(
        qdrant=QdrantStore(url=cfg.qdrant_url, collection=cfg.qdrant_collection),
        embedder=get_embedder(model_name=cfg.embedding_model),
        bm25=bm25,
        reranker=get_reranker(model_name=cfg.reranker_model),
        settings=cfg,
    )

    try:
        retrieve_mode = RetrieveMode(mode)
    except ValueError:
        logger.error("Invalid mode %r. Choose: bm25, dense, hybrid", mode)
        sys.exit(1)

    print(f"\nQuery : {query}")
    print(f"Mode  : {mode}  |  Rerank: {rerank}  |  Top-k: {top_k}")
    if tickers:
        print(f"Filter: tickers={tickers}")
    print("─" * 70)

    results = await retriever.retrieve(
        query,
        top_k=top_k,
        mode=retrieve_mode,
        rerank=rerank,
        ticker_filter=tickers or None,
    )

    if not results:
        print("No results found.")
        return

    for i, ev in enumerate(results, 1):
        score_str = f"{ev.relevance_score:.4f}" if ev.relevance_score is not None else "n/a"
        print(f"\n[{i}] score={score_str}  ticker={ev.ticker}  form={ev.filing_type}", end="")
        if ev.section:
            print(f"  section={ev.section}", end="")
        print()
        print(f"    accession: {ev.metadata.get('accession_number', '?')}")
        # Print first 300 chars of text
        preview = ev.text.replace("\n", " ")[:300]
        if len(ev.text) > 300:
            preview += " …"
        print(f"    {preview}")

    print(f"\n{len(results)} result(s) returned.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test retrieval from the indexed corpus")
    parser.add_argument("query", help="Natural-language query")
    parser.add_argument(
        "--mode",
        choices=["bm25", "dense", "hybrid"],
        default="hybrid",
        help="Retrieval mode (default: hybrid)",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of results (default: 5)")
    parser.add_argument(
        "--no-rerank",
        dest="rerank",
        action="store_false",
        help="Disable cross-encoder reranking",
    )
    parser.add_argument(
        "--tickers",
        nargs="*",
        default=[],
        help="Restrict results to these tickers",
    )
    args = parser.parse_args()

    try:
        asyncio.run(_run(args.query, args.mode, args.top_k, args.rerank, args.tickers))
    except KeyboardInterrupt:
        logger.info("Interrupted")
        sys.exit(0)


if __name__ == "__main__":
    main()
