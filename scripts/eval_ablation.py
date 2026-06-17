#!/usr/bin/env python
"""CLI: Run the ablation matrix and write the report + plots.

Runs {bm25,dense,hybrid} × {rerank} (× {critic} when --with-critic), prints the
table, computes the headline lift with a 95% CI on the per-query lift, writes
docs/EVAL.md and saves plots under notebooks/figures/.

Usage:
    python scripts/eval_ablation.py                 # retrieval-only, 6 cells
    python scripts/eval_ablation.py --with-critic   # 12 cells (no e2e yet)
    python scripts/eval_ablation.py --k 10 --no-plots

Makefile:
    make eval-ablation
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the ablation matrix")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--with-critic", action="store_true", help="Include critic dimension (12 cells)")
    p.add_argument("--golden", type=Path, default=None)
    p.add_argument("--bm25-path", type=Path, default=Path("data/bm25_index.pkl"))
    p.add_argument("--eval-md", type=Path, default=Path("docs/EVAL.md"))
    p.add_argument("--figures", type=Path, default=Path("notebooks/figures"))
    p.add_argument("--no-plots", dest="plots", action="store_false", default=True)
    p.add_argument("--metric", default="ndcg", choices=["ndcg", "precision", "recall", "mrr"])
    return p.parse_args()


async def main() -> None:
    args = _parse_args()

    from mia_eval.ablation import baseline_vs_best, results_to_markdown, run_ablation
    from mia_eval.golden import load_golden_set
    from mia_eval.report import lift_line, write_eval_report
    from mia_eval.stats import bootstrap_ci
    from mia_retrieval.retriever import build_retriever

    golden = load_golden_set(args.golden)
    print(f"Loaded {len(golden)} golden questions")

    retriever = build_retriever(bm25_path=args.bm25_path)
    cells = await run_ablation(
        retriever, golden, k=args.k, with_critic=args.with_critic
    )

    print("\n" + results_to_markdown(cells) + "\n")

    lift = baseline_vs_best(cells, metric=args.metric)
    # Bootstrap CI on the *paired* per-query lift (best - baseline).
    base_raw = lift["baseline_raw"]
    best_raw = lift["best_raw"]
    lift_ci = None
    if len(base_raw) == len(best_raw) and base_raw:
        diffs = [b - a for a, b in zip(base_raw, best_raw)]
        lift_ci = bootstrap_ci(diffs)
    print(lift_line(lift, lift_ci))

    write_eval_report(cells, args.eval_md, lift=lift, lift_ci=lift_ci)
    print(f"\nWrote {args.eval_md}")

    if args.plots:
        from mia_eval.report import plot_ablation_heatmap, plot_retrieval_lift

        plot_retrieval_lift(cells, args.figures / "retrieval_lift.png", metric=args.metric)
        plot_ablation_heatmap(cells, args.figures / "ablation_heatmap.png", metric=args.metric)
        print(f"Saved plots under {args.figures}/")


if __name__ == "__main__":
    asyncio.run(main())
