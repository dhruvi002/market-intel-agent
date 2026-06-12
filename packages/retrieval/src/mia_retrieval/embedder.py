"""Dense embedding via BAAI/bge-large-en-v1.5.

Design decisions:
- Singleton pattern (module-level ``_embedder``): loading a 1.3GB sentence-
  transformer model is expensive (~2–3s on CPU).  A singleton ensures the
  model is loaded exactly once per process regardless of how many callers
  invoke ``get_embedder()``.
- ``normalize_embeddings=True``: with cosine-distance collections in Qdrant,
  normalised vectors let Qdrant use an inner-product HNSW optimisation
  (cosine ≡ dot product on unit vectors), giving a 10–20% speed-up on large
  collections.
- ``show_progress_bar=False`` in encode(): suppresses tqdm output in async
  worker processes and tests.
- The embedding dimension is hard-coded as EMBED_DIM = 1024 so callers can
  create Qdrant collections before loading the model.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

EMBED_DIM: int = 1024   # bge-large-en-v1.5 output dimension
DEFAULT_MODEL: str = "BAAI/bge-large-en-v1.5"
DEFAULT_BATCH_SIZE: int = 32


class Embedder:
    """Wraps a SentenceTransformer model with lazy loading and batched encoding.

    Parameters
    ----------
    model_name  : HuggingFace model identifier (default: bge-large-en-v1.5)
    batch_size  : number of texts encoded per forward pass
    device      : e.g. "cpu", "cuda", "mps".  ``None`` → auto-detect.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._device = device
        self._model: SentenceTransformer | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    def embed(self, texts: list[str]) -> np.ndarray:
        """Encode *texts* and return a (N, EMBED_DIM) float32 array.

        Embeddings are L2-normalised so cosine similarity equals dot product.
        """
        if not texts:
            return np.empty((0, EMBED_DIM), dtype=np.float32)
        model = self._load()
        result = model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(result, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Convenience wrapper: encode a single query string → shape (EMBED_DIM,)."""
        return self.embed([query])[0]

    @property
    def dim(self) -> int:
        """Embedding dimension (fixed at 1024 for bge-large-en-v1.5)."""
        return EMBED_DIM

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load(self) -> "SentenceTransformer":
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415

            logger.info("Loading embedding model %s …", self._model_name)
            kwargs: dict[str, object] = {}
            if self._device is not None:
                kwargs["device"] = self._device
            self._model = SentenceTransformer(self._model_name, **kwargs)
            logger.info("Embedding model loaded (dim=%d)", EMBED_DIM)
        return self._model


# ── Module-level singleton ────────────────────────────────────────────────────

_embedder: Embedder | None = None


def get_embedder(
    model_name: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str | None = None,
) -> Embedder:
    """Return the process-wide :class:`Embedder` singleton.

    The first call constructs and returns a new instance; subsequent calls
    return the same object regardless of arguments.  This ensures the ~1.3GB
    model is loaded only once per worker process.
    """
    global _embedder  # noqa: PLW0603
    if _embedder is None:
        _embedder = Embedder(model_name=model_name, batch_size=batch_size, device=device)
    return _embedder
