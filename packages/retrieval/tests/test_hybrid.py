"""Tests for mia_retrieval.hybrid — Reciprocal Rank Fusion."""

from unittest.mock import MagicMock

from mia_retrieval.chunker import Chunk
from mia_retrieval.hybrid import RRF_K, reciprocal_rank_fusion


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_chunk(chunk_id: str, text: str = "dummy text") -> Chunk:
    return Chunk(
        id=chunk_id,
        filing_id="filing-001",
        ticker="NVDA",
        filing_type="10-K",
        accession_number="0001-23",
        section=None,
        text=text,
        chunk_index=0,
        total_chunks=1,
    )


def make_scored_point(chunk_id: str, score: float, text: str = "dummy text") -> MagicMock:
    """Mock a qdrant_client ScoredPoint with payload."""
    point = MagicMock()
    point.id = chunk_id
    point.score = score
    point.payload = {
        "filing_id": "filing-001",
        "ticker": "NVDA",
        "filing_type": "10-K",
        "accession_number": "0001-23",
        "section": None,
        "text": text,
        "chunk_index": 0,
        "total_chunks": 1,
    }
    return point


# ── RRF correctness ───────────────────────────────────────────────────────────

def test_empty_inputs_return_empty() -> None:
    results = reciprocal_rank_fusion([], [])
    assert results == []


def test_bm25_only_results_returned() -> None:
    bm25 = [(make_chunk("c1"), 3.5), (make_chunk("c2"), 2.1)]
    results = reciprocal_rank_fusion(bm25, [])
    ids = [c.id for c, _ in results]
    assert "c1" in ids
    assert "c2" in ids


def test_dense_only_results_returned() -> None:
    dense = [make_scored_point("c3", 0.9), make_scored_point("c4", 0.7)]
    results = reciprocal_rank_fusion([], dense)
    ids = [c.id for c, _ in results]
    assert "c3" in ids
    assert "c4" in ids


def test_results_sorted_descending_by_rrf_score() -> None:
    bm25 = [(make_chunk("c1"), 5.0), (make_chunk("c2"), 3.0)]
    dense = [make_scored_point("c1", 0.95), make_scored_point("c3", 0.85)]
    results = reciprocal_rank_fusion(bm25, dense)
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True), "Results not sorted descending"


def test_chunk_in_both_lists_gets_higher_score() -> None:
    """A chunk ranked #1 in both BM25 and dense should outscore a chunk in only one."""
    shared_chunk = make_chunk("shared")
    only_bm25 = make_chunk("only_bm25")

    bm25 = [(shared_chunk, 10.0), (only_bm25, 8.0)]
    dense = [make_scored_point("shared", 0.99)]

    results = reciprocal_rank_fusion(bm25, dense)
    id_to_score = {c.id: s for c, s in results}

    assert id_to_score["shared"] > id_to_score["only_bm25"], (
        "Chunk in both lists should score higher than chunk in only BM25"
    )


def test_rrf_score_formula() -> None:
    """Verify RRF formula: score = sum(1 / (k + rank + 1)), rank is 0-based."""
    bm25 = [(make_chunk("c1"), 1.0)]   # rank 0 in BM25
    dense = [make_scored_point("c1", 1.0)]  # rank 0 in dense

    results = reciprocal_rank_fusion(bm25, dense, k=RRF_K)
    _, fused_score = results[0]

    expected = (1.0 / (RRF_K + 0 + 1)) + (1.0 / (RRF_K + 0 + 1))
    assert abs(fused_score - expected) < 1e-9


def test_no_duplicates_in_output() -> None:
    """A chunk appearing in both lists should produce exactly one output entry."""
    shared = make_chunk("c_shared")
    bm25 = [(shared, 3.0), (make_chunk("c_bm25"), 2.0)]
    dense = [
        make_scored_point("c_shared", 0.9),
        make_scored_point("c_dense", 0.8),
    ]

    results = reciprocal_rank_fusion(bm25, dense)
    ids = [c.id for c, _ in results]
    assert len(ids) == len(set(ids)), "Duplicate chunk IDs in RRF output"
    assert len(ids) == 3  # shared, c_bm25, c_dense


def test_chunk_metadata_preserved() -> None:
    """Chunks reconstructed from ScoredPoint payloads should carry full metadata."""
    dense = [make_scored_point("c99", 0.88, text="revenue data center growth")]
    results = reciprocal_rank_fusion([], dense)
    chunk, _ = results[0]
    assert chunk.id == "c99"
    assert chunk.ticker == "NVDA"
    assert chunk.text == "revenue data center growth"
