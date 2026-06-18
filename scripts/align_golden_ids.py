#!/usr/bin/env python
"""Align golden_set.jsonl doc IDs to real chunk UUIDs via semantic span matching.

For each golden question's relevant_spans, embeds the span and searches Qdrant
for the most semantically similar chunk (per ticker). This is non-circular:
ground truth is determined by span semantics, not by any retrieval config.

A chunk is accepted as relevant if its cosine similarity to the span exceeds
--threshold (default 0.75). If no chunk clears the threshold, we warn but keep
the question with an empty relevant_doc_ids (it will score 0 — honest).

Usage:
    uv run python scripts/align_golden_ids.py               # write golden set
    uv run python scripts/align_golden_ids.py --dry-run     # preview only
    uv run python scripts/align_golden_ids.py --threshold 0.70
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path

GOLDEN_PATH = Path("packages/eval/src/mia_eval/data/golden_set.jsonl")
BM25_PATH   = Path("data/bm25_index.pkl")

# Similarity threshold for accepting a chunk as relevant.
# bge-large cosine scores: >0.85 = near-duplicate, >0.75 = strong match,
# >0.65 = topically related. 0.75 is intentionally conservative.
DEFAULT_THRESHOLD = 0.75


async def _embed_and_search(
    span: str,
    tickers: list[str],
    *,
    embedder,
    qdrant_client,
    collection: str,
    top_k: int,
    threshold: float,
) -> list[tuple[str, float]]:
    """Embed *span* and return (doc_id, score) pairs above *threshold*."""
    import numpy as np
    from qdrant_client.models import FieldCondition, Filter, MatchAny  # noqa

    vec = embedder.embed([span])[0]
    # Ensure L2-normalised (bge embedder already does this, but be safe)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm

    query_filter = None
    if tickers:
        query_filter = Filter(
            must=[FieldCondition(key="ticker", match=MatchAny(any=tickers))]
        )

    results = await qdrant_client.query_points(
        collection_name=collection,
        query=vec.tolist(),
        limit=top_k,
        query_filter=query_filter,
        with_payload=True,
    )

    hits = []
    for point in results.points:
        score = float(point.score)
        if score >= threshold:
            doc_id = str(point.id)
            hits.append((doc_id, score))
    return hits


async def align(threshold: float, top_k: int, dry_run: bool) -> None:
    from mia_eval.golden import load_golden_set
    from mia_retrieval.embedder import get_embedder
    from mia_retrieval.qdrant_store import QdrantStore
    from mia_shared.config import get_settings
    from qdrant_client import AsyncQdrantClient

    settings = get_settings()
    embedder  = get_embedder(settings.embedding_model)

    qdrant_store = QdrantStore(url=settings.qdrant_url, collection=settings.qdrant_collection)
    # Access internal async client
    client = await qdrant_store._get_client()

    golden = load_golden_set(GOLDEN_PATH)
    print(f"Loaded {len(golden)} golden questions")
    print(f"Similarity threshold: {threshold}  |  top_k per span: {top_k}\n")

    updated: list[dict] = []
    stats = {"ok": 0, "partial": 0, "miss": 0}

    for qa in golden:
        found_ids: list[str] = []
        span_scores: list[float] = []

        for span in (qa.relevant_spans or []):
            hits = await _embed_and_search(
                span,
                tickers=qa.tickers or [],
                embedder=embedder,
                qdrant_client=client,
                collection=settings.qdrant_collection,
                top_k=top_k,
                threshold=threshold,
            )
            for doc_id, score in hits:
                if doc_id not in found_ids:
                    found_ids.append(doc_id)
                    span_scores.append(score)

        n_spans = len(qa.relevant_spans or [])
        n_found = len(found_ids)

        if n_found == 0:
            status = "[MISS]    "
            stats["miss"] += 1
        elif n_found < n_spans:
            status = "[PARTIAL] "
            stats["partial"] += 1
        else:
            status = "[OK]      "
            stats["ok"] += 1

        score_str = ", ".join(f"{s:.3f}" for s in span_scores[:3])
        print(f"  {status} {qa.id}: {n_found}/{n_spans} spans matched  scores=[{score_str}]")

        row = {
            "id": qa.id,
            "question": qa.question,
            "answer": qa.answer,
            "question_type": qa.question_type,
            "tickers": qa.tickers,
            "relevant_doc_ids": found_ids,
            "relevant_spans": qa.relevant_spans or [],
        }
        updated.append(row)

    print(f"\nSummary:  OK={stats['ok']}  PARTIAL={stats['partial']}  MISS={stats['miss']}")
    print(f"Coverage: {stats['ok'] + stats['partial']}/{len(golden)} questions have ≥1 relevant chunk\n")

    if dry_run:
        print("[DRY RUN] No files written.")
        return

    lines = [
        "# Golden Q/A set — Autonomous Market Intelligence Agent",
        f"# Aligned via semantic span matching (threshold={threshold})",
        "# script: scripts/align_golden_ids.py",
        "#",
    ]
    for row in updated:
        lines.append(json.dumps(row))

    GOLDEN_PATH.write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(updated)} entries → {GOLDEN_PATH}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                   help="Min cosine similarity to accept a chunk as relevant (default 0.75)")
    p.add_argument("--top-k", type=int, default=5,
                   help="Qdrant candidates per span (default 5)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    asyncio.run(align(args.threshold, args.top_k, args.dry_run))


if __name__ == "__main__":
    main()
