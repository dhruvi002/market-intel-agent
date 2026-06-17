"""Custom retrieval metrics: Recall@k, Precision@k, MRR, nDCG@k.

These are the metrics that quantify and defend the headline "28% retrieval
precision improvement" claim.  They are intentionally **dependency-free**
(pure Python + math) so they are fast to unit-test and produce identical
numbers across machines — no LLM, no torch, no randomness.

Conventions
-----------
- A *ranking* is an ordered list of doc ids, best-first (rank 1 = index 0).
- *Relevant* is the ground-truth set of doc ids (from ``GoldenQA.relevant_set``).
- Binary relevance is assumed (a doc is relevant or not) — standard for IR
  benchmarks built from human-tagged ground truth and what RAGAS/BEIR use.
- Metrics are undefined when there are no relevant docs; we return 0.0 for
  Recall/Precision/MRR and 1.0 for nDCG over an empty relevant set is also 0.0
  (an empty query is excluded upstream, but we stay total to avoid crashes).

The ``evaluate_retrieval`` coroutine runs an actual ``Retriever`` over the
golden set and aggregates per-query metrics — that's the only async, infra-
touching part of this module and it is mocked in tests.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, Sequence

from mia_shared.schemas import Evidence

if TYPE_CHECKING:  # avoid importing the heavy retrieval stack at module load
    from mia_eval.golden import GoldenQA
    from mia_retrieval.retriever import RetrieveMode, Retriever

logger = logging.getLogger(__name__)


# ── Primitive metrics (pure functions) ─────────────────────────────────────────

def recall_at_k(ranking: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of relevant docs that appear in the top-*k* of *ranking*."""
    rel = set(relevant)
    if not rel:
        return 0.0
    hits = sum(1 for doc in ranking[:k] if doc in rel)
    return hits / len(rel)


def precision_at_k(ranking: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of the top-*k* retrieved docs that are relevant."""
    if k <= 0:
        return 0.0
    rel = set(relevant)
    topk = ranking[:k]
    if not topk:
        return 0.0
    hits = sum(1 for doc in topk if doc in rel)
    return hits / min(k, len(topk))


def reciprocal_rank(ranking: Sequence[str], relevant: Iterable[str]) -> float:
    """1 / rank of the first relevant doc (0.0 if none retrieved). Basis of MRR."""
    rel = set(relevant)
    for idx, doc in enumerate(ranking, start=1):
        if doc in rel:
            return 1.0 / idx
    return 0.0


def _dcg(gains: Sequence[float]) -> float:
    # DCG with the standard log2(rank+1) discount; rank is 1-based.
    return sum(g / math.log2(i + 1) for i, g in enumerate(gains, start=1))


def ndcg_at_k(ranking: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Normalised DCG@k under binary relevance.

    DCG of the actual top-k ranking divided by the ideal DCG (all relevant
    docs ranked first). Returns 0.0 when there are no relevant docs.
    """
    rel = set(relevant)
    if not rel:
        return 0.0
    gains = [1.0 if doc in rel else 0.0 for doc in ranking[:k]]
    dcg = _dcg(gains)
    ideal_hits = min(len(rel), k)
    idcg = _dcg([1.0] * ideal_hits)
    return dcg / idcg if idcg > 0 else 0.0


# ── Per-query metric bundle ────────────────────────────────────────────────────

@dataclass(slots=True)
class RetrievalMetrics:
    """All retrieval metrics for a single query at a fixed *k*."""

    k: int
    recall: float
    precision: float
    mrr: float
    ndcg: float
    num_relevant: int
    num_retrieved: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "k": self.k,
            "recall": self.recall,
            "precision": self.precision,
            "mrr": self.mrr,
            "ndcg": self.ndcg,
            "num_relevant": self.num_relevant,
            "num_retrieved": self.num_retrieved,
        }


def score_ranking(
    ranking: Sequence[str], relevant: Iterable[str], k: int = 10
) -> RetrievalMetrics:
    """Compute the full :class:`RetrievalMetrics` bundle for one ranking."""
    rel = set(relevant)
    return RetrievalMetrics(
        k=k,
        recall=recall_at_k(ranking, rel, k),
        precision=precision_at_k(ranking, rel, k),
        mrr=reciprocal_rank(ranking, rel),
        ndcg=ndcg_at_k(ranking, rel, k),
        num_relevant=len(rel),
        num_retrieved=len(ranking),
    )


# ── Evidence → doc id ──────────────────────────────────────────────────────────

def evidence_doc_id(ev: Evidence) -> str:
    """Extract the comparable doc id from an :class:`Evidence`.

    The Phase 2 indexer stores the stable chunk id under
    ``metadata["doc_id"]``; older evidence may only carry the random
    ``Evidence.id``.  We prefer ``doc_id``, then ``chunk_id``, then the UUID.
    """
    meta = ev.metadata or {}
    for key in ("doc_id", "chunk_id", "id"):
        val = meta.get(key)
        if val:
            return str(val)
    return str(ev.id)


# ── Aggregate runner (async, infra-touching — mocked in tests) ─────────────────

@dataclass(slots=True)
class RetrievalRunResult:
    """Aggregate metrics across the whole golden set for one retrieval config."""

    mode: str
    rerank: bool
    k: int
    per_query: list[RetrievalMetrics] = field(default_factory=list)

    def _mean(self, attr: str) -> float:
        vals = [getattr(m, attr) for m in self.per_query]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def mean_recall(self) -> float:
        return self._mean("recall")

    @property
    def mean_precision(self) -> float:
        return self._mean("precision")

    @property
    def mean_mrr(self) -> float:
        return self._mean("mrr")

    @property
    def mean_ndcg(self) -> float:
        return self._mean("ndcg")

    def summary(self) -> dict[str, float | str | bool | int]:
        return {
            "mode": self.mode,
            "rerank": self.rerank,
            "k": self.k,
            "n": len(self.per_query),
            "recall": self.mean_recall,
            "precision": self.mean_precision,
            "mrr": self.mean_mrr,
            "ndcg": self.mean_ndcg,
        }


async def evaluate_retrieval(
    retriever: "Retriever",
    golden: "Sequence[GoldenQA]",
    *,
    mode: "RetrieveMode | str" = "hybrid",
    rerank: bool = True,
    k: int = 10,
) -> RetrievalRunResult:
    """Run *retriever* over every golden question and aggregate metrics.

    Parameters
    ----------
    retriever : configured :class:`~mia_retrieval.retriever.Retriever`
    golden    : the golden Q/A set
    mode      : ``bm25`` | ``dense`` | ``hybrid``
    rerank    : apply the cross-encoder reranker
    k         : cut-off for Recall@k / Precision@k / nDCG@k

    Returns
    -------
    RetrievalRunResult
    """
    from mia_retrieval.retriever import RetrieveMode  # noqa: PLC0415

    rmode = RetrieveMode(mode) if not isinstance(mode, RetrieveMode) else mode
    result = RetrievalRunResult(mode=rmode.value, rerank=rerank, k=k)

    for qa in golden:
        evidence = await retriever.retrieve(
            qa.question,
            mode=rmode,
            rerank=rerank,
            ticker_filter=qa.tickers or None,
            top_k=k,
        )
        ranking = [evidence_doc_id(ev) for ev in evidence]
        metrics = score_ranking(ranking, qa.relevant_set, k=k)
        result.per_query.append(metrics)
        logger.debug(
            "q=%s mode=%s rerank=%s recall=%.3f ndcg=%.3f",
            qa.id, rmode.value, rerank, metrics.recall, metrics.ndcg,
        )

    return result
