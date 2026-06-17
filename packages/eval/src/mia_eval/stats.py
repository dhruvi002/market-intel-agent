"""Bootstrap confidence intervals for eval metrics.

The capstone Definition-of-Done requires the ablation table to report the
retrieval-precision lift "with 95% confidence intervals".  With ~50 golden
questions the per-query metric distribution is non-normal and bounded in
[0, 1], so a parametric (t-based) interval is the wrong tool.  A
**percentile bootstrap** makes no distributional assumption: resample the
per-query scores with replacement many times, recompute the mean each time,
and read the 2.5th / 97.5th percentiles of the resampled means.

Pure NumPy — no SciPy dependency at import — so it stays cheap to test and
deterministic under a fixed seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(slots=True, frozen=True)
class MeanCI:
    """A point estimate (mean) with a bootstrap confidence interval."""

    mean: float
    low: float
    high: float
    confidence: float
    n: int

    @property
    def margin(self) -> float:
        """Half-width of the interval (max distance from the mean to a bound)."""
        return max(self.high - self.mean, self.mean - self.low)

    def __str__(self) -> str:  # e.g. "0.742 [0.681, 0.803]"
        return f"{self.mean:.3f} [{self.low:.3f}, {self.high:.3f}]"


def bootstrap_ci(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int | None = 42,
) -> MeanCI:
    """Percentile-bootstrap CI for the mean of *values*.

    Parameters
    ----------
    values      : per-query metric scores (e.g. nDCG for each golden question)
    confidence  : interval coverage (default 0.95)
    n_resamples : number of bootstrap resamples (default 10,000)
    seed        : RNG seed for reproducibility (default 42; pass ``None`` for
                  nondeterministic behaviour)

    Returns
    -------
    MeanCI

    Notes
    -----
    - Empty input → all-zero interval (avoids crashing an eval run that had no
      data for a category).
    - Single value → degenerate interval ``[v, v]``.
    """
    arr = np.asarray(list(values), dtype=float)
    n = int(arr.size)
    if n == 0:
        return MeanCI(mean=0.0, low=0.0, high=0.0, confidence=confidence, n=0)
    if n == 1:
        v = float(arr[0])
        return MeanCI(mean=v, low=v, high=v, confidence=confidence, n=1)

    rng = np.random.default_rng(seed)
    # Vectorised resample: (n_resamples, n) index matrix → resampled means.
    idx = rng.integers(0, n, size=(n_resamples, n))
    resample_means = arr[idx].mean(axis=1)

    alpha = 1.0 - confidence
    low = float(np.percentile(resample_means, 100 * (alpha / 2)))
    high = float(np.percentile(resample_means, 100 * (1 - alpha / 2)))
    return MeanCI(
        mean=float(arr.mean()),
        low=low,
        high=high,
        confidence=confidence,
        n=n,
    )


def mean_ci(
    values: Sequence[float], *, confidence: float = 0.95, **kwargs: object
) -> MeanCI:
    """Convenience alias for :func:`bootstrap_ci` with keyword passthrough."""
    return bootstrap_ci(values, confidence=confidence, **kwargs)  # type: ignore[arg-type]
