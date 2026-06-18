"""Cross-encoder reranker using BAAI/bge-reranker-v2-m3.

Design decisions:
- bge-reranker-v2-m3 over Cohere Rerank: free, local, no network call, and
  outperforms Cohere rerank-v2 on several BEIR benchmarks.  The "-v2-m3"
  variant uses a multilingual base (mdeberta) which also handles the numeric-
  heavy language in SEC filings better than English-only rerankers.
- CrossEncoder (not BiEncoder): bi-encoder scores query and document
  independently.  Cross-encoders concatenate the pair as one input, allowing
  full self-attention between query and passage tokens — much higher accuracy
  at the cost of O(n) forward passes.  For top-k=50 candidates after
  BM25/dense retrieval, this is fast enough on CPU (~200ms).
- Lazy loading: the ~550MB model is loaded on first :meth:`rerank` call, not
  at import time, keeping tests and lightweight imports fast.
- show_progress_bar=False: suppresses tqdm output in async workers and tests.
- Singleton pattern (module-level ``get_reranker``): same rationale as the
  Embedder singleton — one model load per process.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

from mia_retrieval.chunker import Chunk

logger = logging.getLogger(__name__)

DEFAULT_RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"


class Reranker:
    """Cross-encoder reranker with lazy model loading.

    Parameters
    ----------
    model_name : HuggingFace model identifier
    device     : e.g. "cpu", "cuda".  ``None`` → auto-detect.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        device: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._model: CrossEncoder | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    def rerank(
        self,
        query: str,
        candidates: list[Chunk],
        top_k: int,
    ) -> list[tuple[Chunk, float]]:
        """Re-score *candidates* with the cross-encoder and return the top *top_k*.

        Parameters
        ----------
        query      : the user's natural-language query
        candidates : candidate chunks from BM25/dense/hybrid retrieval
        top_k      : number of top results to return after reranking

        Returns
        -------
        list[tuple[Chunk, float]]
            Sorted by reranker score descending, at most *top_k* entries.
        """
        if not candidates:
            return []

        model = self._load()
        pairs = [(query, c.text) for c in candidates]
        raw_scores = model.predict(pairs, show_progress_bar=False)

        # raw_scores is a numpy array or list of floats
        ranked = sorted(
            zip(candidates, (float(s) for s in raw_scores)),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:top_k]

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load(self) -> CrossEncoder:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info("Loading reranker model %s …", self._model_name)
            # Default to CPU: embedder already uses MPS on Apple Silicon and
            # running both models on MPS simultaneously exhausts shared memory.
            # Cross-encoder on k=50 candidates is ~200ms on CPU — fast enough.
            device = self._device or "cpu"
            self._model = CrossEncoder(self._model_name, device=device)
            logger.info("Reranker model loaded")
        return self._model


# ── Module-level singleton ────────────────────────────────────────────────────

_reranker: Reranker | None = None


def get_reranker(
    model_name: str = DEFAULT_RERANKER_MODEL,
    device: str | None = None,
) -> Reranker:
    """Return the process-wide :class:`Reranker` singleton."""
    global _reranker
    if _reranker is None:
        _reranker = Reranker(model_name=model_name, device=device)
    return _reranker
