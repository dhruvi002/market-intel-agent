"""NLI entailment scoring via a local cross-encoder model.

The model is loaded lazily on first call and cached in-process via
``functools.lru_cache``.  Keeping the loader behind a function means
importing this module is cheap — torch / transformers are not pulled in
at module-import time.

Labels for ``cross-encoder/nli-deberta-v3-base`` (and most HuggingFace
NLI cross-encoders) follow the order::

    [contradiction, neutral, entailment]   →   indices [0, 1, 2]

``score_pairs`` returns the softmax probability of the entailment class,
so callers receive a float in ``[0, 1]`` where 1.0 = perfect entailment.

Usage
-----
::

    from mia_agents import nli

    # Single-pair convenience
    p = nli.score_pairs([("Evidence text here.", "Claim to verify.")],
                        model_name="cross-encoder/nli-deberta-v3-base")
    print(p[0])   # e.g. 0.87

This module is intentionally free of async code — callers that need async
behaviour should dispatch via ``asyncio.to_thread``.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np

logger = logging.getLogger(__name__)

# Index of the "entailment" class in the three-way NLI output
_ENTAILMENT_IDX: int = 2


@lru_cache(maxsize=1)
def _get_nli_model(model_name: str):
    """Load and cache the NLI CrossEncoder model (singleton per model name).

    The first call downloads / loads the model weights; subsequent calls
    return the cached instance immediately.
    """
    from sentence_transformers import CrossEncoder  # noqa: PLC0415

    logger.info("nli: loading model '%s' — first call may take a moment", model_name)
    model = CrossEncoder(model_name)
    logger.info("nli: model '%s' ready", model_name)
    return model


def score_pairs(
    pairs: list[tuple[str, str]],
    model_name: str = "cross-encoder/nli-deberta-v3-base",
) -> list[float]:
    """Return entailment probabilities for a batch of (premise, hypothesis) pairs.

    Parameters
    ----------
    pairs      : list of ``(premise, hypothesis)`` tuples.
                 ``premise`` is the source evidence text;
                 ``hypothesis`` is the claim to verify.
    model_name : HuggingFace model identifier or local path.

    Returns
    -------
    list[float]
        Entailment probability in ``[0, 1]`` for each pair, in input order.
        Returns an empty list when ``pairs`` is empty.

    Notes
    -----
    - This function is *synchronous* and CPU-bound.  Call it from async
      code via ``await asyncio.to_thread(score_pairs, pairs, model_name)``.
    - ``apply_softmax=True`` is passed to ``CrossEncoder.predict`` so the
      raw logits are converted to proper probabilities before extraction.
    """
    if not pairs:
        return []

    model = _get_nli_model(model_name)
    probs: np.ndarray = model.predict(pairs, apply_softmax=True)  # shape (N, 3)

    # probs may be 1-D if N == 1 — normalise to always be 2-D
    if probs.ndim == 1:
        probs = probs[np.newaxis, :]

    return [float(row[_ENTAILMENT_IDX]) for row in probs]
