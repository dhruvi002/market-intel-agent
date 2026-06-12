"""Tests for mia_retrieval.retriever — mocked dependencies."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from mia_retrieval.bm25_index import BM25Index
from mia_retrieval.chunker import Chunk
from mia_retrieval.retriever import RetrieveMode, Retriever, _chunk_to_evidence


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_chunk(chunk_id: str, ticker: str = "NVDA", text: str = "test text") -> Chunk:
    return Chunk(
        id=chunk_id,
        filing_id="filing-001",
        ticker=ticker,
        filing_type="10-K",
        accession_number="0001-23",
        section=None,
        text=text,
        chunk_index=0,
        total_chunks=1,
    )


def make_scored_point(chunk_id: str, score: float, ticker: str = "NVDA") -> MagicMock:
    p = MagicMock()
    p.id = chunk_id
    p.score = score
    p.payload = {
        "filing_id": "filing-001",
        "ticker": ticker,
        "filing_type": "10-K",
        "accession_number": "0001-23",
        "section": None,
        "text": "dense result text",
        "chunk_index": 0,
        "total_chunks": 1,
    }
    return p


def make_settings(
    bm25_top_k: int = 50,
    dense_top_k: int = 50,
    rerank_top_k: int = 10,
) -> MagicMock:
    s = MagicMock()
    s.bm25_top_k = bm25_top_k
    s.dense_top_k = dense_top_k
    s.rerank_top_k = rerank_top_k
    return s


def build_retriever(
    bm25_results: list[tuple[Chunk, float]] | None = None,
    dense_results: list[Any] | None = None,
    reranker_output: list[tuple[Chunk, float]] | None = None,
    bm25_index: BM25Index | None = None,
) -> Retriever:
    """Build a Retriever with mocked components."""
    bm25 = bm25_index or MagicMock(spec=BM25Index)
    if bm25_results is not None:
        bm25.search.return_value = bm25_results

    embedder = MagicMock()
    embedder.embed_query.return_value = np.zeros(1024, dtype=np.float32)

    qdrant = MagicMock()
    qdrant.search = AsyncMock(return_value=dense_results or [])

    reranker = MagicMock()
    if reranker_output is not None:
        reranker.rerank.return_value = reranker_output
    else:
        reranker.rerank.side_effect = lambda q, candidates, top_k: [
            (c, 0.5) for c in candidates[:top_k]
        ]

    return Retriever(
        qdrant=qdrant,
        embedder=embedder,
        bm25=bm25,
        reranker=reranker,
        settings=make_settings(),
    )


# ── _chunk_to_evidence ────────────────────────────────────────────────────────

def test_chunk_to_evidence_fields() -> None:
    chunk = make_chunk("c1", ticker="AAPL", text="Apple revenue beat estimates")
    evidence = _chunk_to_evidence(chunk, 0.87)
    assert evidence.source_type == "rag_chunk"
    assert evidence.ticker == "AAPL"
    assert evidence.filing_type == "10-K"
    assert evidence.text == "Apple revenue beat estimates"
    assert abs(evidence.relevance_score - 0.87) < 1e-9
    assert evidence.metadata["chunk_index"] == 0


# ── BM25 mode ─────────────────────────────────────────────────────────────────

async def test_bm25_mode_returns_evidence() -> None:
    chunks = [make_chunk(f"c{i}") for i in range(3)]
    bm25_results = [(c, float(3 - i)) for i, c in enumerate(chunks)]
    retriever = build_retriever(bm25_results=bm25_results)

    results = await retriever.retrieve(
        "NVIDIA revenue", mode=RetrieveMode.BM25, rerank=False, top_k=3
    )
    assert len(results) == 3
    for ev in results:
        assert ev.source_type == "rag_chunk"


async def test_bm25_mode_ticker_filter() -> None:
    nvda_chunk = make_chunk("c_nvda", ticker="NVDA")
    amd_chunk = make_chunk("c_amd", ticker="AMD")
    bm25_results = [(nvda_chunk, 3.0), (amd_chunk, 2.0)]

    retriever = build_retriever(bm25_results=bm25_results)
    results = await retriever.retrieve(
        "revenue", mode=RetrieveMode.BM25, rerank=False, ticker_filter=["NVDA"]
    )
    tickers = {ev.ticker for ev in results}
    assert "AMD" not in tickers
    assert "NVDA" in tickers


# ── Dense mode ────────────────────────────────────────────────────────────────

async def test_dense_mode_calls_qdrant() -> None:
    dense_results = [make_scored_point(f"c{i}", float(1.0 - i * 0.1)) for i in range(3)]
    retriever = build_retriever(dense_results=dense_results)

    results = await retriever.retrieve(
        "data center growth", mode=RetrieveMode.DENSE, rerank=False, top_k=3
    )
    retriever._qdrant.search.assert_awaited_once()
    assert len(results) == 3


# ── Hybrid mode ───────────────────────────────────────────────────────────────

async def test_hybrid_mode_calls_both() -> None:
    chunks = [make_chunk("c1"), make_chunk("c2")]
    bm25_results = [(chunks[0], 5.0), (chunks[1], 3.0)]
    dense_results = [make_scored_point("c1", 0.95), make_scored_point("c3", 0.85)]

    retriever = build_retriever(
        bm25_results=bm25_results,
        dense_results=dense_results,
    )
    results = await retriever.retrieve(
        "NVIDIA GPU revenue", mode=RetrieveMode.HYBRID, rerank=False, top_k=5
    )
    retriever._qdrant.search.assert_awaited_once()
    retriever._bm25.search.assert_called_once()
    assert len(results) > 0


# ── Reranking ─────────────────────────────────────────────────────────────────

async def test_rerank_is_called_when_enabled() -> None:
    chunks = [make_chunk(f"c{i}") for i in range(5)]
    bm25_results = [(c, float(5 - i)) for i, c in enumerate(chunks)]
    reranker_output = [(chunks[0], 0.99), (chunks[2], 0.88)]

    retriever = build_retriever(
        bm25_results=bm25_results, reranker_output=reranker_output
    )
    results = await retriever.retrieve(
        "revenue", mode=RetrieveMode.BM25, rerank=True, top_k=2
    )
    retriever._reranker.rerank.assert_called_once()
    assert len(results) == 2


async def test_rerank_not_called_when_disabled() -> None:
    chunks = [make_chunk("c1")]
    bm25_results = [(chunks[0], 3.0)]
    retriever = build_retriever(bm25_results=bm25_results)
    await retriever.retrieve("query", mode=RetrieveMode.BM25, rerank=False, top_k=1)
    retriever._reranker.rerank.assert_not_called()


# ── Empty results ─────────────────────────────────────────────────────────────

async def test_no_candidates_returns_empty() -> None:
    retriever = build_retriever(bm25_results=[], dense_results=[])
    results = await retriever.retrieve("anything", mode=RetrieveMode.HYBRID, rerank=False)
    assert results == []
