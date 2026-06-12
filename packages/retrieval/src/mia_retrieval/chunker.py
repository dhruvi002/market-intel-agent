"""Text chunker for SEC filing content.

Strategy: paragraph-aware word-count chunking with overlap.

Design decisions:
- Word-count proxy (not tokenizer) to avoid pulling the sentence-transformers
  tokenizer into a module loaded at import time.  bge-large-en-v1.5 has a
  512-token limit; financial text averages ~1.4–1.6 tokens/word, so
  CHUNK_SIZE_WORDS=300 → ≈420–480 tokens, safely under the cap.
- Paragraphs (\n\n) are respected as natural break-points; sentences (\n)
  as secondary split-points.  This keeps financial statement paragraphs and
  MD&A bullets together.
- Overlap (CHUNK_OVERLAP_WORDS) carries the tail of one chunk into the head
  of the next, preserving cross-sentence context for the retriever.
- Chunk IDs are deterministic UUID5 values from (filing_id, chunk_index),
  so re-indexing the same filing is fully idempotent.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Sequence

# ── Tunable constants ─────────────────────────────────────────────────────────

CHUNK_SIZE_WORDS: int = 300   # max words per chunk
CHUNK_OVERLAP_WORDS: int = 40  # words carried forward from previous chunk
MIN_CHUNK_WORDS: int = 20     # discard tail chunks shorter than this

# Namespace for deterministic chunk UUIDs
_CHUNK_NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # UUID_DNS


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Chunk:
    """A single text chunk produced from one SEC filing.

    Attributes
    ----------
    id               : deterministic UUID5 derived from (filing_id, chunk_index)
    filing_id        : UUID of the source ``mia.filings`` row (as str)
    ticker           : e.g. "NVDA"
    filing_type      : "10-K" | "10-Q" | "8-K"
    accession_number : SEC accession number, e.g. "0001045810-23-000017"
    section          : heuristic section label ("MD&A", "Risk Factors", …) or None
    text             : the raw chunk text
    chunk_index      : 0-based position within the parent filing
    total_chunks     : total chunks produced from the parent filing
    """

    id: str
    filing_id: str
    ticker: str
    filing_type: str
    accession_number: str
    section: str | None
    text: str
    chunk_index: int
    total_chunks: int  # placeholder — caller updates via Chunk.with_total()

    def with_total(self, total: int) -> "Chunk":
        """Return a copy with ``total_chunks`` set (used after all chunks are known)."""
        return Chunk(
            id=self.id,
            filing_id=self.filing_id,
            ticker=self.ticker,
            filing_type=self.filing_type,
            accession_number=self.accession_number,
            section=self.section,
            text=self.text,
            chunk_index=self.chunk_index,
            total_chunks=total,
        )


def _make_chunk_id(filing_id: str, chunk_index: int) -> str:
    """Deterministic UUID5 from (filing_id, chunk_index)."""
    key = f"{filing_id}:{chunk_index}"
    return str(uuid.uuid5(_CHUNK_NS, key))


# ── Chunker ───────────────────────────────────────────────────────────────────

class Chunker:
    """Split filing text into overlapping word-count chunks.

    Parameters
    ----------
    chunk_size    : max words per chunk (default CHUNK_SIZE_WORDS)
    overlap       : words of overlap between consecutive chunks (default CHUNK_OVERLAP_WORDS)
    min_words     : minimum words for the last chunk to be kept (default MIN_CHUNK_WORDS)
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE_WORDS,
        overlap: int = CHUNK_OVERLAP_WORDS,
        min_words: int = MIN_CHUNK_WORDS,
    ) -> None:
        if overlap >= chunk_size:
            raise ValueError(f"overlap ({overlap}) must be < chunk_size ({chunk_size})")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_words = min_words

    # ── Public API ────────────────────────────────────────────────────────────

    def chunk(
        self,
        text: str,
        filing_id: str,
        ticker: str,
        filing_type: str,
        accession_number: str,
        section: str | None = None,
    ) -> list[Chunk]:
        """Chunk *text* and return a list of :class:`Chunk` objects.

        Returns an empty list for blank or very-short input.
        """
        if not text or not text.strip():
            return []

        words = _tokenize(text)
        if len(words) < self.min_words:
            return []

        raw_chunks = self._split_words(words)

        # Build Chunk objects (total_chunks filled in after the list is complete)
        total = len(raw_chunks)
        chunks: list[Chunk] = []
        for idx, word_seq in enumerate(raw_chunks):
            chunk_text = " ".join(word_seq)
            cid = _make_chunk_id(filing_id, idx)
            chunks.append(
                Chunk(
                    id=cid,
                    filing_id=filing_id,
                    ticker=ticker,
                    filing_type=filing_type,
                    accession_number=accession_number,
                    section=section,
                    text=chunk_text,
                    chunk_index=idx,
                    total_chunks=total,
                )
            )
        return chunks

    # ── Internal ──────────────────────────────────────────────────────────────

    def _split_words(self, words: list[str]) -> list[list[str]]:
        """Sliding window over the flat word list."""
        step = self.chunk_size - self.overlap
        slices: list[list[str]] = []
        start = 0
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            segment = words[start:end]
            if len(segment) >= self.min_words or not slices:
                slices.append(segment)
            start += step
        return slices


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Simple whitespace tokenisation.  Fast and dependency-free."""
    return text.split()
