#!/usr/bin/env python
"""CLI: Retrieval-quality eval over the golden set (Recall/Precision/MRR/nDCG).

Runs a single retrieval configuration against the golden Q/A set and prints the
aggregate metrics with 95% bootstrap CIs. No LLM required.

Usage:
    python scripts/eval_retrieval.py --mode hybrid --rerank
    python scripts/eval_retrieval.py --mode bm25 --no-rerank --k 10

Makefile:
    make eval-retrieval mode=hybrid
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Retrieval eval over the golden set")
    p.add_argument("--mode", choices=["bm25", "dense", "hybrid"], default="hybrid")
    p.add_argument("--rerank", dest="rerank", action="store_true", default=True)
    p.add_argument("--no-rerank", dest="rerank", action="store_false")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--golden", type=Path, default=None, help="Override golden JSONL path")
    p.add_argument("--bm25-path", type=Path, default=Path("data/bm25_index.pkl"))
    return p.parse_args()


async def main() -> None:
    args = _parse_args()

    from mia_eval.golden import load_golden_set
    from mia_eval.retrieval_metrics import evaluate_retrieval
    from mia_eval.stats import bootstrap_ci
    from mia_retrieval.retriever import build_retriever

    golden = load_golden_set(args.golden)
    print(f"Loaded {len(golden)} golden questions")

    retriever = build_retriever(bm25_path=args.bm25_path)
    result = await evaluate_retrieval(
        retriever, golden, mode=args.mode, rerank=args.rerank, k=args.k
    )

    ndcg_ci = bootstrap_ci([m.ndcg for m in result.per_query])
    prec_ci = bootstrap_ci([m.precision for m in result.per_query])

    sep = "─" * 60
    print(f"\n{sep}\nRETRIEVAL EVAL — mode={args.mode} rerank={args.rerank} k={args.k}\n{sep}")
    print(f"  n questions    : {len(result.per_query)}")
    print(f"  Recall@{args.k}      : {result.mean_recall:.3f}")
    print(f"  Precision@{args.k}   : {prec_ci}")
    print(f"  MRR            : {result.mean_mrr:.3f}")
    print(f"  nDCG@{args.k}        : {ndcg_ci}")
    print(sep)


if __name__ == "__main__":
    asyncio.run(main())
