"""BM25 sparse retrieval index over filing chunks.

Design decisions:
- rank_bm25.BM25Okapi: the most widely used BM25 variant; k1=1.5, b=0.75 are
  the library defaults and empirically solid for IR tasks.
- Rebuild-on-add: BM25Okapi doesn't support incremental updates (the IDF is
  computed over the full corpus at build time).  Adding chunks requires
  rebuilding the whole index.  For a Phase 2 / capstone corpus this is fast
  (<1s for 20k chunks); a production system would shard or switch to Elastic.
- Pickle persistence: simple and zero-dependency.  The index lives in
  data/bm25_index.pkl (gitignored).  It is rebuilt by ``make index`` and
  reloaded at worker startup.
- Tokenisation: lowercase + whitespace split.  Stopword removal and stemming
  improve recall marginally at the cost of interpretability; omitted here.
- Scores ≤ 0 are filtered out: BM25 returns 0 for terms with no IDF signal
  (very common words or OOV terms).  Including zero-score results in RRF
  degrades fusion quality.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rank_bm25 import BM25Okapi

from mia_retrieval.chunker import Chunk

logger = logging.getLogger(__name__)

_DEFAULT_BM25_PATH = Path("data/bm25_index.pkl")


class BM25Index:
    """BM25Okapi wrapper with persist / load support.

    Usage
    -----
    >>> idx = BM25Index()
    >>> idx.build(chunks)          # initial build
    >>> idx.add(new_chunks)        # incremental (triggers full rebuild)
    >>> results = idx.search(query, top_k=50)
    >>> idx.save(Path("data/bm25_index.pkl"))

    >>> idx2 = BM25Index.load(Path("data/bm25_index.pkl"))
    """

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._chunks: list[Chunk] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def build(self, chunks: list[Chunk]) -> None:
        """Build (or rebuild) the index from *chunks*.

        Replaces any existing index state.
        """
        if not chunks:
            self._bm25 = None
            self._chunks = []
            return

        from rank_bm25 import BM25Okapi  # noqa: PLC0415

        tokenized = [_tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(tokenized)
        self._chunks = list(chunks)
        logger.info("BM25 index built: %d chunks", len(self._chunks))

    def add(self, chunks: list[Chunk]) -> None:
        """Add *chunks* to the index (full rebuild under the hood).

        Call :meth:`save` afterwards to persist.
        """
        combined = self._chunks + list(chunks)
        self.build(combined)

    def search(self, query: str, top_k: int) -> list[tuple[Chunk, float]]:
        """Return up to *top_k* (Chunk, score) pairs, score-descending.

        Chunks with score ≤ 0 are excluded (no BM25 signal for this query).
        """
        if self._bm25 is None or not self._chunks:
            return []

        tokens = _tokenize(query)
        if not tokens:
            return []

        scores: np.ndarray = self._bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: list[tuple[Chunk, float]] = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0.0:
                break  # indices are sorted descending; no need to continue
            results.append((self._chunks[idx], score))
        return results

    def save(self, path: Path = _DEFAULT_BM25_PATH) -> None:
        """Pickle the index to *path* (parent directories created if needed)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"bm25": self._bm25, "chunks": self._chunks}
        with open(path, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("BM25 index saved to %s (%d chunks)", path, len(self._chunks))

    @classmethod
    def load(cls, path: Path = _DEFAULT_BM25_PATH) -> "BM25Index":
        """Load a previously saved index from *path*."""
        with open(path, "rb") as fh:
            payload = pickle.load(fh)  # noqa: S301
        inst = cls()
        inst._bm25 = payload["bm25"]
        inst._chunks = payload["chunks"]
        logger.info("BM25 index loaded from %s (%d chunks)", path, len(inst._chunks))
        return inst

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        """Number of chunks in the index."""
        return len(self._chunks)

    @property
    def is_empty(self) -> bool:
        return self._bm25 is None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Lowercase whitespace tokenisation — matches Chunker._tokenize."""
    return text.lower().split()
