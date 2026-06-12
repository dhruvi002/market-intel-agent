"""Tests for mia_retrieval.chunker."""

import pytest

from mia_retrieval.chunker import Chunk, Chunker, _make_chunk_id


# ── Fixtures ──────────────────────────────────────────────────────────────────

FILING_ID = "11111111-0000-0000-0000-000000000001"
TICKER = "NVDA"
FORM = "10-K"
ACCESSION = "0001045810-23-000017"


def make_chunker(**kwargs: object) -> Chunker:
    defaults = {"chunk_size": 10, "overlap": 2, "min_words": 3}
    defaults.update(kwargs)  # type: ignore[arg-type]
    return Chunker(**defaults)  # type: ignore[arg-type]


def long_text(words: int = 50) -> str:
    return " ".join(f"word{i}" for i in range(words))


# ── Chunk ID ──────────────────────────────────────────────────────────────────

def test_chunk_id_is_deterministic() -> None:
    id1 = _make_chunk_id(FILING_ID, 0)
    id2 = _make_chunk_id(FILING_ID, 0)
    assert id1 == id2


def test_chunk_ids_differ_by_index() -> None:
    id0 = _make_chunk_id(FILING_ID, 0)
    id1 = _make_chunk_id(FILING_ID, 1)
    assert id0 != id1


def test_chunk_ids_differ_by_filing() -> None:
    other_id = "22222222-0000-0000-0000-000000000002"
    assert _make_chunk_id(FILING_ID, 0) != _make_chunk_id(other_id, 0)


# ── Basic chunking ────────────────────────────────────────────────────────────

def test_empty_text_returns_empty() -> None:
    chunker = make_chunker()
    assert chunker.chunk("", FILING_ID, TICKER, FORM, ACCESSION) == []


def test_whitespace_only_returns_empty() -> None:
    chunker = make_chunker()
    assert chunker.chunk("   \n\n   ", FILING_ID, TICKER, FORM, ACCESSION) == []


def test_too_short_text_returns_empty() -> None:
    chunker = make_chunker(min_words=10)
    result = chunker.chunk("only five words here", FILING_ID, TICKER, FORM, ACCESSION)
    assert result == []


def test_short_text_fits_in_one_chunk() -> None:
    chunker = make_chunker(chunk_size=100)
    text = long_text(30)
    chunks = chunker.chunk(text, FILING_ID, TICKER, FORM, ACCESSION)
    assert len(chunks) == 1


def test_long_text_produces_multiple_chunks() -> None:
    chunker = make_chunker(chunk_size=10, overlap=2)
    text = long_text(50)
    chunks = chunker.chunk(text, FILING_ID, TICKER, FORM, ACCESSION)
    assert len(chunks) > 1


def test_chunk_size_respected() -> None:
    chunker = make_chunker(chunk_size=10, overlap=0)
    text = long_text(35)
    chunks = chunker.chunk(text, FILING_ID, TICKER, FORM, ACCESSION)
    for chunk in chunks[:-1]:  # last chunk may be smaller
        assert len(chunk.text.split()) <= 10


def test_chunk_metadata_filled() -> None:
    chunker = make_chunker(chunk_size=20, overlap=2)
    text = long_text(50)
    chunks = chunker.chunk(text, FILING_ID, TICKER, FORM, ACCESSION, section="MD&A")
    for chunk in chunks:
        assert chunk.filing_id == FILING_ID
        assert chunk.ticker == TICKER
        assert chunk.filing_type == FORM
        assert chunk.accession_number == ACCESSION
        assert chunk.section == "MD&A"


def test_chunk_indices_sequential() -> None:
    chunker = make_chunker(chunk_size=10, overlap=2)
    text = long_text(50)
    chunks = chunker.chunk(text, FILING_ID, TICKER, FORM, ACCESSION)
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_index == i


def test_total_chunks_matches_list_length() -> None:
    chunker = make_chunker(chunk_size=10, overlap=2)
    text = long_text(50)
    chunks = chunker.chunk(text, FILING_ID, TICKER, FORM, ACCESSION)
    total = len(chunks)
    for chunk in chunks:
        assert chunk.total_chunks == total


def test_overlap_carries_words_forward() -> None:
    """The first words of chunk[1] should appear at the end of chunk[0]."""
    chunker = make_chunker(chunk_size=10, overlap=3)
    text = long_text(30)
    chunks = chunker.chunk(text, FILING_ID, TICKER, FORM, ACCESSION)
    assert len(chunks) >= 2

    tail = chunks[0].text.split()[-3:]
    head = chunks[1].text.split()[:3]
    assert tail == head, f"Expected overlap: tail={tail}, head={head}"


# ── with_total ────────────────────────────────────────────────────────────────

def test_with_total_returns_updated_copy() -> None:
    chunker = make_chunker(chunk_size=100)
    text = long_text(20)
    chunks = chunker.chunk(text, FILING_ID, TICKER, FORM, ACCESSION)
    chunk = chunks[0]
    updated = chunk.with_total(42)
    assert updated.total_chunks == 42
    assert updated.id == chunk.id  # identity preserved


# ── Constructor validation ────────────────────────────────────────────────────

def test_overlap_must_be_less_than_chunk_size() -> None:
    with pytest.raises(ValueError, match="overlap"):
        Chunker(chunk_size=5, overlap=5)
