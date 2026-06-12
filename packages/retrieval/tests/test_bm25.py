"""Tests for mia_retrieval.bm25_index."""

import pickle
import tempfile
from pathlib import Path

import pytest

from mia_retrieval.bm25_index import BM25Index, _tokenize
from mia_retrieval.chunker import Chunk


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_chunk(text: str, idx: int = 0, ticker: str = "NVDA") -> Chunk:
    return Chunk(
        id=f"chunk-{ticker}-{idx}",
        filing_id="filing-001",
        ticker=ticker,
        filing_type="10-K",
        accession_number="0001-23",
        section=None,
        text=text,
        chunk_index=idx,
        total_chunks=1,
    )


CORPUS = [
    make_chunk("NVIDIA reported record revenue of 44 billion dollars in fiscal 2024", 0),
    make_chunk("Data center segment grew 217 percent year over year", 1),
    make_chunk("AMD announced strong growth in the AI accelerator market", 2),
    make_chunk("Risk factors include supply chain disruptions and geopolitical risks", 3),
    make_chunk("Cash and cash equivalents totaled 7.6 billion at quarter end", 4),
]


# ── Tokenizer ─────────────────────────────────────────────────────────────────

def test_tokenize_lowercases() -> None:
    assert _tokenize("NVIDIA Revenue") == ["nvidia", "revenue"]


def test_tokenize_splits_on_whitespace() -> None:
    tokens = _tokenize("hello  world\tfoo")
    assert "hello" in tokens
    assert "world" in tokens
    assert "foo" in tokens


# ── Build ─────────────────────────────────────────────────────────────────────

def test_build_empty_corpus() -> None:
    idx = BM25Index()
    idx.build([])
    assert idx.is_empty
    assert idx.size == 0


def test_build_sets_size() -> None:
    idx = BM25Index()
    idx.build(CORPUS)
    assert idx.size == len(CORPUS)


def test_rebuild_replaces_state() -> None:
    idx = BM25Index()
    idx.build(CORPUS)
    idx.build(CORPUS[:2])
    assert idx.size == 2


# ── Search ────────────────────────────────────────────────────────────────────

def test_search_empty_index_returns_empty() -> None:
    idx = BM25Index()
    assert idx.search("revenue", top_k=5) == []


def test_search_returns_relevant_result() -> None:
    idx = BM25Index()
    idx.build(CORPUS)
    results = idx.search("NVIDIA revenue fiscal 2024", top_k=3)
    assert len(results) > 0
    top_text = results[0][0].text
    assert "revenue" in top_text.lower() or "NVIDIA" in top_text


def test_search_top_k_respected() -> None:
    idx = BM25Index()
    idx.build(CORPUS)
    results = idx.search("revenue growth data center", top_k=2)
    assert len(results) <= 2


def test_search_scores_positive() -> None:
    idx = BM25Index()
    idx.build(CORPUS)
    results = idx.search("NVIDIA revenue", top_k=5)
    for _, score in results:
        assert score > 0


def test_search_sorted_descending() -> None:
    idx = BM25Index()
    idx.build(CORPUS)
    results = idx.search("revenue data center growth", top_k=5)
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


def test_search_oov_query_returns_empty() -> None:
    """A query with no terms in the corpus should return no results."""
    idx = BM25Index()
    idx.build(CORPUS)
    results = idx.search("zxqwerty xyzabcdef", top_k=5)
    assert results == []


# ── Add ───────────────────────────────────────────────────────────────────────

def test_add_increases_size() -> None:
    idx = BM25Index()
    idx.build(CORPUS)
    new_chunk = make_chunk("Intel posted declining margins in the PC segment", idx=99)
    idx.add([new_chunk])
    assert idx.size == len(CORPUS) + 1


def test_add_makes_new_chunk_searchable() -> None:
    idx = BM25Index()
    idx.build(CORPUS)
    new_chunk = make_chunk("Quantum computing revenue breakthrough announced", idx=99)
    idx.add([new_chunk])
    results = idx.search("quantum computing revenue", top_k=3)
    texts = [c.text for c, _ in results]
    assert any("Quantum" in t for t in texts)


# ── Persist / load ────────────────────────────────────────────────────────────

def test_save_and_load_roundtrip() -> None:
    idx = BM25Index()
    idx.build(CORPUS)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "bm25_test.pkl"
        idx.save(path)
        assert path.exists()

        loaded = BM25Index.load(path)
        assert loaded.size == idx.size

        # Search results should be identical after roundtrip
        orig_results = idx.search("NVIDIA revenue", top_k=3)
        load_results = loaded.search("NVIDIA revenue", top_k=3)
        orig_ids = [c.id for c, _ in orig_results]
        load_ids = [c.id for c, _ in load_results]
        assert orig_ids == load_ids


def test_save_creates_parent_directory() -> None:
    idx = BM25Index()
    idx.build(CORPUS[:2])
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "nested" / "subdir" / "bm25.pkl"
        idx.save(path)
        assert path.exists()
