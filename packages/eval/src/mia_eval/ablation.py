"""Ablation matrix runner: {retriever} × {rerank} × {critic}.

The defensible-claim engine.  The matrix is::

    {bm25, dense, hybrid} × {no-rerank, rerank} × {no-critic, critic} = 12 cells

Two layers of metrics are collected:

- **Retrieval layer** (every cell, cheap, deterministic): Recall@k / Precision@k
  / MRR / nDCG@k from :mod:`mia_eval.retrieval_metrics`.  The retriever-and-
  rerank dimensions are what move these numbers; the critic dimension does not
  touch retrieval, so retrieval metrics are shared across the critic on/off pair
  (computed once per retrieval config, not re-run).
- **End-to-end layer** (optional, expensive): when ``with_e2e=True`` each
  critic on/off configuration runs the full LangGraph and records Pass@1 and
  the mean number of self-correction iterations — this is what isolates the
  critic's contribution.

The headline "28% precision lift" is read off the retrieval layer as the
nDCG@10 (or Precision@10) delta between ``hybrid+rerank`` and the ``bm25``
baseline, with the 95% CI from :mod:`mia_eval.stats`.

Pandas is imported lazily inside :func:`results_dataframe` so importing this
module stays light.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Sequence

from mia_eval.retrieval_metrics import (
    RetrievalMetrics,
    evidence_doc_id,
    score_ranking,
)

if TYPE_CHECKING:
    from mia_eval.golden import GoldenQA
    from mia_retrieval.retriever import Retriever

logger = logging.getLogger(__name__)

RETRIEVAL_MODES = ("bm25", "dense", "hybrid")
RERANK_OPTIONS = (False, True)
CRITIC_OPTIONS = (False, True)


@dataclass(slots=True)
class AblationCell:
    """One cell of the ablation matrix and its aggregate metrics."""

    mode: str
    rerank: bool
    critic: bool
    n: int = 0
    # retrieval-layer means
    recall: float = 0.0
    precision: float = 0.0
    mrr: float = 0.0
    ndcg: float = 0.0
    # end-to-end layer (only when with_e2e)
    pass_at_1: float | None = None
    mean_iterations: float | None = None
    # raw per-query retrieval scores (for bootstrap CIs downstream)
    ndcg_raw: list[float] = field(default_factory=list)
    precision_raw: list[float] = field(default_factory=list)

    @property
    def label(self) -> str:
        r = "+rerank" if self.rerank else ""
        c = "+critic" if self.critic else ""
        return f"{self.mode}{r}{c}" or self.mode

    def as_row(self) -> dict[str, object]:
        d = asdict(self)
        d.pop("ndcg_raw", None)
        d.pop("precision_raw", None)
        d["label"] = self.label
        return d


def ablation_grid(
    *, with_critic: bool = True
) -> list[tuple[str, bool, bool]]:
    """Enumerate the matrix cells as ``(mode, rerank, critic)`` tuples.

    ``with_critic=False`` collapses the critic dimension (6 cells) — useful for
    a fast retrieval-only ablation that needs no LLM.
    """
    critic_opts = CRITIC_OPTIONS if with_critic else (False,)
    return list(itertools.product(RETRIEVAL_MODES, RERANK_OPTIONS, critic_opts))


def _aggregate(metrics: Sequence[RetrievalMetrics]) -> dict[str, float]:
    if not metrics:
        return {"recall": 0.0, "precision": 0.0, "mrr": 0.0, "ndcg": 0.0}
    n = len(metrics)
    return {
        "recall": sum(m.recall for m in metrics) / n,
        "precision": sum(m.precision for m in metrics) / n,
        "mrr": sum(m.mrr for m in metrics) / n,
        "ndcg": sum(m.ndcg for m in metrics) / n,
    }


async def _retrieval_metrics_for(
    retriever: "Retriever",
    golden: "Sequence[GoldenQA]",
    mode: str,
    rerank: bool,
    k: int,
) -> list[RetrievalMetrics]:
    """Per-query retrieval metrics for one (mode, rerank) config."""
    from mia_retrieval.retriever import RetrieveMode  # noqa: PLC0415

    rmode = RetrieveMode(mode)
    out: list[RetrievalMetrics] = []
    for qa in golden:
        evidence = await retriever.retrieve(
            qa.question,
            mode=rmode,
            rerank=rerank,
            ticker_filter=qa.tickers or None,
            top_k=k,
        )
        ranking = [evidence_doc_id(ev) for ev in evidence]
        out.append(score_ranking(ranking, qa.relevant_set, k=k))
    return out


async def run_ablation(
    retriever: "Retriever",
    golden: "Sequence[GoldenQA]",
    *,
    k: int = 10,
    with_critic: bool = True,
    with_e2e: bool = False,
    e2e_runner=None,
) -> list[AblationCell]:
    """Run the ablation matrix and return one :class:`AblationCell` per cell.

    Parameters
    ----------
    retriever   : configured :class:`~mia_retrieval.retriever.Retriever`
    golden      : the golden Q/A set
    k           : retrieval cut-off
    with_critic : include the critic on/off dimension (12 cells vs 6)
    with_e2e    : also run the full graph for Pass@1 / iteration counts
    e2e_runner  : optional async ``(golden, mode, rerank, critic) -> (pass@1,
                  mean_iters)`` callable.  Injected so the expensive,
                  infra-touching graph run is mocked in tests and swappable.

    Returns
    -------
    list[AblationCell]
    """
    # Retrieval metrics depend only on (mode, rerank); compute once and reuse
    # across the critic dimension to avoid 2× redundant retrieval passes.
    retrieval_cache: dict[tuple[str, bool], list[RetrievalMetrics]] = {}

    cells: list[AblationCell] = []
    for mode, rerank, critic in ablation_grid(with_critic=with_critic):
        key = (mode, rerank)
        if key not in retrieval_cache:
            retrieval_cache[key] = await _retrieval_metrics_for(
                retriever, golden, mode, rerank, k
            )
        per_query = retrieval_cache[key]
        agg = _aggregate(per_query)

        cell = AblationCell(
            mode=mode,
            rerank=rerank,
            critic=critic,
            n=len(per_query),
            recall=agg["recall"],
            precision=agg["precision"],
            mrr=agg["mrr"],
            ndcg=agg["ndcg"],
            ndcg_raw=[m.ndcg for m in per_query],
            precision_raw=[m.precision for m in per_query],
        )

        if with_e2e and e2e_runner is not None:
            pass1, mean_iters = await e2e_runner(golden, mode, rerank, critic)
            cell.pass_at_1 = pass1
            cell.mean_iterations = mean_iters

        logger.info(
            "ablation cell %-22s ndcg=%.3f precision=%.3f",
            cell.label, cell.ndcg, cell.precision,
        )
        cells.append(cell)

    return cells


def baseline_vs_best(
    cells: Sequence[AblationCell],
    *,
    metric: str = "ndcg",
    baseline_mode: str = "bm25",
) -> dict[str, object]:
    """Compute the headline lift: best retrieval config vs the baseline.

    Returns a dict with the baseline value, best value, absolute and relative
    lift, and the raw per-query arrays for both so a bootstrap CI of the *lift*
    can be computed by the caller.
    """
    # Critic dimension does not affect retrieval metrics — restrict to one slice.
    retrieval_cells = [c for c in cells if not c.critic]
    if not retrieval_cells:
        retrieval_cells = list(cells)

    baseline = next(
        (c for c in retrieval_cells if c.mode == baseline_mode and not c.rerank),
        None,
    )
    if baseline is None:
        raise ValueError(f"no baseline cell for mode={baseline_mode!r}, rerank=False")

    raw_attr = f"{metric}_raw"
    best = max(retrieval_cells, key=lambda c: getattr(c, metric))
    base_val = getattr(baseline, metric)
    best_val = getattr(best, metric)
    abs_lift = best_val - base_val
    rel_lift = (abs_lift / base_val) if base_val > 0 else float("inf")

    return {
        "metric": metric,
        "baseline_label": baseline.label,
        "best_label": best.label,
        "baseline_value": base_val,
        "best_value": best_val,
        "absolute_lift": abs_lift,
        "relative_lift_pct": rel_lift * 100.0,
        "baseline_raw": list(getattr(baseline, raw_attr, [])),
        "best_raw": list(getattr(best, raw_attr, [])),
    }


def results_dataframe(cells: Sequence[AblationCell]):
    """Convert cells to a pandas DataFrame (lazy import)."""
    import pandas as pd  # noqa: PLC0415

    return pd.DataFrame([c.as_row() for c in cells])
