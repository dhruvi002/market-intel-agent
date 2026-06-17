"""mia_eval — evaluation harness for the Market Intelligence Agent.

Phase 8 deliverable.  Provides:

- ``golden``            : hand-authored Q/A set schema + loader
- ``retrieval_metrics`` : Recall@k, MRR, nDCG, Precision@k (custom, no LLM)
- ``stats``             : bootstrap 95% confidence intervals
- ``ragas_eval``        : RAGAS generation metrics wired to the free LLM stack
- ``ablation``          : 12-cell ablation matrix runner
- ``tracing``           : Langfuse v2 observability helpers (no-op when unconfigured)
- ``report``            : markdown tables + matplotlib/seaborn plots

Lazy imports keep heavy optional deps (ragas, pandas, matplotlib, torch) out of
the import path until the specific symbol is first accessed — so collecting the
fast pure-Python tests (metrics, stats, golden) never drags in the ML stack.
"""

from __future__ import annotations

import importlib

_lazy: dict[str, str] = {
    # golden set
    "GoldenQA": "mia_eval.golden",
    "QuestionType": "mia_eval.golden",
    "load_golden_set": "mia_eval.golden",
    "DEFAULT_GOLDEN_PATH": "mia_eval.golden",
    # retrieval metrics
    "recall_at_k": "mia_eval.retrieval_metrics",
    "precision_at_k": "mia_eval.retrieval_metrics",
    "reciprocal_rank": "mia_eval.retrieval_metrics",
    "ndcg_at_k": "mia_eval.retrieval_metrics",
    "RetrievalMetrics": "mia_eval.retrieval_metrics",
    "score_ranking": "mia_eval.retrieval_metrics",
    "evaluate_retrieval": "mia_eval.retrieval_metrics",
    # stats
    "bootstrap_ci": "mia_eval.stats",
    "MeanCI": "mia_eval.stats",
    "mean_ci": "mia_eval.stats",
    # ragas
    "RagasScores": "mia_eval.ragas_eval",
    "evaluate_generation": "mia_eval.ragas_eval",
    # ablation
    "AblationCell": "mia_eval.ablation",
    "ablation_grid": "mia_eval.ablation",
    "run_ablation": "mia_eval.ablation",
    # tracing
    "get_langfuse_handler": "mia_eval.tracing",
    "langfuse_enabled": "mia_eval.tracing",
    "observe_run": "mia_eval.tracing",
    # report
    "results_to_markdown": "mia_eval.report",
    "plot_retrieval_lift": "mia_eval.report",
    "plot_ablation_heatmap": "mia_eval.report",
    "write_eval_report": "mia_eval.report",
}


def __getattr__(name: str) -> object:
    if name in _lazy:
        mod = importlib.import_module(_lazy[name])
        return getattr(mod, name)
    raise AttributeError(f"module 'mia_eval' has no attribute {name!r}")


__all__ = list(_lazy)
