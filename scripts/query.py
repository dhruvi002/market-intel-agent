#!/usr/bin/env python
"""CLI: Run the single-agent RAG baseline against indexed filings.

Usage:
    python scripts/query.py "How is NVDA's data center revenue evolving?"
    python scripts/query.py "AMD vs NVDA GPU margins" --tickers NVDA AMD
    python scripts/query.py "Risk factors for MSFT" --mode bm25 --no-rerank

Makefile shortcuts:
    make query q="How is NVDA revenue growing?"
    make query-bm25 q="NVDA supply chain risk"
    make query-dense q="AMD vs NVDA"
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import textwrap
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the MIA single-agent RAG baseline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Examples:
              python scripts/query.py "NVDA data center revenue growth"
              python scripts/query.py "Compare AMD and NVDA GPU margins" \\
                  --tickers NVDA AMD --mode hybrid
              python scripts/query.py "MSFT cloud risk factors" --mode bm25 --no-rerank
            """
        ),
    )
    parser.add_argument("query", help="Natural-language question")
    parser.add_argument(
        "--tickers",
        nargs="*",
        metavar="TICKER",
        help="Restrict retrieval to specific tickers (e.g. NVDA AMD)",
    )
    parser.add_argument(
        "--mode",
        choices=["hybrid", "bm25", "dense"],
        default="hybrid",
        help="Retrieval mode (default: hybrid)",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Skip cross-encoder reranking",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Max evidence chunks to pass to the LLM (default: settings.rerank_top_k)",
    )
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

    # ── Imports deferred so --help is fast ────────────────────────────────────
    from mia_agents.llm import get_llm
    from mia_agents.rag_agent import RAGAgent
    from mia_retrieval.retriever import RetrieveMode, build_retriever

    # ── Build components ──────────────────────────────────────────────────────
    print("Loading retriever…", flush=True)
    retriever = build_retriever(bm25_path=args.bm25_path)

    print("Loading LLM…", flush=True)
    llm = get_llm(provider=args.provider)

    agent = RAGAgent(retriever=retriever, llm=llm)

    # ── Run ───────────────────────────────────────────────────────────────────
    mode = RetrieveMode(args.mode)
    tickers = [t.upper() for t in args.tickers] if args.tickers else None

    print(f"\nQuery : {args.query}")
    if tickers:
        print(f"Tickers: {', '.join(tickers)}")
    print(f"Mode  : {mode.value}  |  rerank={not args.no_rerank}\n")
    print("Running…", flush=True)

    response = await agent.run(
        args.query,
        tickers=tickers,
        mode=mode,
        rerank=not args.no_rerank,
        top_k=args.top_k,
    )

    # ── Print results ─────────────────────────────────────────────────────────
    _print_response(response)


def _print_response(response) -> None:
    sep = "─" * 72

    print(f"\n{sep}")
    print("ANSWER")
    print(sep)
    print(response.answer)

    if response.evidence:
        print(f"\n{sep}")
        print(f"EVIDENCE  ({len(response.evidence)} chunks)")
        print(sep)
        for i, ev in enumerate(response.evidence, 1):
            header = f"[{i}]"
            if ev.ticker:
                header += f" {ev.ticker}"
            if ev.filing_type:
                header += f" {ev.filing_type}"
            if ev.section:
                header += f" — {ev.section}"
            score = f"  (score: {ev.relevance_score:.3f})" if ev.relevance_score else ""
            print(f"\n{header}{score}")
            # Print first 200 chars of each chunk
            snippet = ev.text[:200].replace("\n", " ")
            if len(ev.text) > 200:
                snippet += "…"
            print(f"  {snippet}")

    if response.citations:
        print(f"\n{sep}")
        print(f"CITATIONS  ({len(response.citations)} unique sources)")
        print(sep)
        for cit in response.citations:
            ev_match = next((e for e in response.evidence if e.id == cit.evidence_id), None)
            src = ""
            if ev_match:
                src = f" [{ev_match.ticker} {ev_match.filing_type}]"
            print(f"  •{src}  {cit.claim_text[:80]}")

    print(f"\n{sep}")
    print(
        f"model={response.model_used or 'unknown'} | "
        f"mode={response.retrieval_mode} | "
        f"reranked={response.reranked} | "
        f"{response.latency_ms:.0f} ms"
    )
    print(sep)


if __name__ == "__main__":
    asyncio.run(main())
