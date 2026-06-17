#!/usr/bin/env python
"""CLI: Run the Phase-4 multi-agent LangGraph pipeline.

Usage:
    python scripts/graph_run.py "How is NVDA's data center revenue evolving?"
    python scripts/graph_run.py "AMD vs NVDA GPU margins" --provider groq

Makefile shortcuts:
    make graph-run q="How is NVDA revenue growing?"
    make graph-run-groq q="AMD vs NVDA GPU margins"
"""

from __future__ import annotations

import argparse
import asyncio
import textwrap
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the MIA multi-agent LangGraph pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Examples:
              python scripts/graph_run.py "NVDA data center revenue growth"
              python scripts/graph_run.py "Compare AMD and NVDA GPU margins" --provider groq
            """
        ),
    )
    parser.add_argument("query", help="Natural-language question")
    parser.add_argument(
        "--provider",
        choices=["gemini", "groq", "cerebras"],
        default=None,
        help="Pin a specific LLM provider (default: full fallback chain)",
    )
    parser.add_argument(
        "--bm25-path",
        type=Path,
        default=Path("data/bm25_index.pkl"),
        help="Path to BM25 index pickle (default: data/bm25_index.pkl)",
    )
    return parser.parse_args()


async def main() -> None:
    args = _parse_args()

    from mia_agents.graph import build_graph
    from mia_agents.llm import get_llm
    from mia_retrieval.retriever import build_retriever

    print("Loading retriever…", flush=True)
    retriever = build_retriever(bm25_path=args.bm25_path)

    print("Loading LLM…", flush=True)
    llm = get_llm(provider=args.provider)

    print("Compiling graph…", flush=True)
    graph = build_graph(retriever=retriever, llm=llm)

    # Phase 8: attach Langfuse callbacks (no-op when keys are unconfigured).
    # LangGraph propagates these to every node's LLM call, so one trace per run
    # captures the full span tree, token counts and synthetic cost.
    from mia_eval.tracing import langchain_callbacks, observe_run

    print(f"\nQuery: {args.query}\n")
    print("Running multi-agent pipeline…", flush=True)

    async with observe_run("graph_run", metadata={"query": args.query}):
        result = await graph.ainvoke(
            {"query": args.query},
            config={"callbacks": langchain_callbacks()},
        )

    _print_result(result)


def _print_result(state: dict) -> None:
    sep = "─" * 72

    print(f"\n{sep}")
    print("FINAL DRAFT")
    print(sep)
    print(state.get("draft") or "(no draft produced)")

    evidence = state.get("evidence", [])
    if evidence:
        print(f"\n{sep}")
        print(f"EVIDENCE  ({len(evidence)} chunks)")
        print(sep)
        for i, ev in enumerate(evidence, 1):
            header = f"[{i}]"
            if hasattr(ev, "ticker") and ev.ticker:
                header += f" {ev.ticker}"
            if hasattr(ev, "filing_type") and ev.filing_type:
                header += f" {ev.filing_type}"
            if hasattr(ev, "section") and ev.section:
                header += f" — {ev.section}"
            print(f"\n{header}")
            snippet = str(ev.text)[:200].replace("\n", " ")
            if len(str(ev.text)) > 200:
                snippet += "…"
            print(f"  {snippet}")

    critique = state.get("critique")
    if critique:
        print(f"\n{sep}")
        print("CRITIC VERDICT")
        print(sep)
        verdict = getattr(critique, "verdict", "unknown")
        summary = getattr(critique, "summary", "")
        print(f"  Verdict : {verdict.value if hasattr(verdict, 'value') else verdict}")
        if summary:
            print(f"  Summary : {summary}")

    iterations = state.get("iteration_count", 0)
    print(f"\n{sep}")
    print(f"iterations={iterations} | evidence_chunks={len(evidence)}")
    print(sep)


if __name__ == "__main__":
    asyncio.run(main())
