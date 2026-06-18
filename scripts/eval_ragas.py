#!/usr/bin/env python
"""CLI: RAGAS generation eval over the golden set.

For each golden question: runs the single-agent RAG baseline to produce an
answer + retrieved contexts, then scores faithfulness / answer-relevancy /
context-precision / context-recall with RAGAS (judged by the free LLM stack).

Usage:
    python scripts/eval_ragas.py --mode hybrid --rerank
    python scripts/eval_ragas.py --limit 10           # cheap smoke run

Makefile:
    make eval-ragas
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RAGAS generation eval over the golden set")
    p.add_argument("--mode", choices=["bm25", "dense", "hybrid"], default="hybrid")
    p.add_argument("--rerank", dest="rerank", action="store_true", default=True)
    p.add_argument("--no-rerank", dest="rerank", action="store_false")
    p.add_argument("--limit", type=int, default=None, help="Eval only the first N questions (quota saver)")
    p.add_argument("--golden", type=Path, default=None)
    p.add_argument("--bm25-path", type=Path, default=Path("data/bm25_index.pkl"))
    p.add_argument("--provider", choices=["gemini", "groq", "cerebras"], default=None)
    p.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="RAGAS judge concurrency (default 1 — serial, safe for free-tier shared queues)",
    )
    return p.parse_args()


async def main() -> None:
    args = _parse_args()

    from mia_agents.llm import get_llm
    from mia_agents.rag_agent import RAGAgent
    from mia_eval.golden import load_golden_set
    from mia_eval.ragas_eval import build_samples_from_responses, evaluate_generation
    from mia_retrieval.retriever import RetrieveMode, build_retriever

    golden = load_golden_set(args.golden)
    if args.limit:
        golden = golden[: args.limit]
    print(f"Evaluating {len(golden)} golden questions with RAGAS")

    retriever = build_retriever(bm25_path=args.bm25_path)
    llm = get_llm(provider=args.provider)
    agent = RAGAgent(retriever=retriever, llm=llm)

    answers: list[str] = []
    contexts: list[list[str]] = []
    for qa in golden:
        resp = await agent.run(
            qa.question,
            tickers=qa.tickers or None,
            mode=RetrieveMode(args.mode),
            rerank=args.rerank,
        )
        answers.append(resp.answer)
        contexts.append([ev.text for ev in resp.evidence])
        print(f"  ✓ {qa.id}: {len(resp.evidence)} contexts")

    samples = build_samples_from_responses(golden, answers, contexts)
    scores = evaluate_generation(samples, llm=llm, max_workers=args.max_workers)

    sep = "─" * 60
    print(f"\n{sep}\nRAGAS SCORES — mode={args.mode} rerank={args.rerank} n={scores.n}\n{sep}")
    print(f"  faithfulness       : {scores.faithfulness:.3f}")
    print(f"  answer_relevancy   : {scores.answer_relevancy:.3f}")
    print(f"  context_precision  : {scores.context_precision:.3f}")
    print(f"  context_recall     : {scores.context_recall:.3f}")
    print(sep)


if __name__ == "__main__":
    asyncio.run(main())
