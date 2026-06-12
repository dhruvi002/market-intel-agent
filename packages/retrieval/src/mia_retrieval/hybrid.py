"""Reciprocal Rank Fusion (RRF) for BM25 + dense search results.

Design decisions:
- RRF over score-based fusion: RRF is parameter-free (k=60 is the standard
  constant from the original Cormack et al. 2009 paper), rank-based (immune
  to scale differences between BM25 scores and cosine similarities), and
  empirically competitive with learned fusion weights on most benchmarks.
  Score normalisation (min-max / softmax) would require tuning per corpus.
- k=60: the classic choice from the original paper.  Higher k smooths the
  fusion (all lists contribute more equally); lower k amplifies the top of
  each list.  60 is the safe default for heterogeneous IR systems.
- Deduplication: the same chunk can appear in both BM25 and dense results.
  RRF naturally handles this by summing contributions from each list — a
  chunk at rank 1 in both lists gets the highest fused score.
- Chunk reconstruction from ScoredPoint: Qdrant search returns ScoredPoint
  objects with full payloads; we reconstruct Chunk objects to keep the
  downstream API uniform.
"""

from __future__ import annotations

from typing import Any

from mia_retrieval.chunker import Chunk
from mia_retrieval.qdrant_store import chunk_from_scored_point

# Original Cormack et al. 2009 constant.  Increase to weight lower-ranked
# results more heavily; decrease to sharpen the top-rank advantage.
RRF_K: int = 60


def reciprocal_rank_fusion(
    bm25_results: list[tuple[Chunk, float]],
    dense_results: list[Any],   # list[qdrant_client.models.ScoredPoint]
    k: int = RRF_K,
) -> list[tuple[Chunk, float]]:
    """Fuse BM25 and dense search results using Reciprocal Rank Fusion.

    Parameters
    ----------
    bm25_results  : ranked list of (Chunk, bm25_score) pairs, best-first
    dense_results : ranked list of Qdrant ``ScoredPoint`` objects, best-first
    k             : RRF smoothing constant (default 60)

    Returns
    -------
    list[tuple[Chunk, float]]
        Merged list sorted by fused RRF score descending.  Each entry contains
        the reconstructed Chunk and its RRF score.
    """
    scores: dict[str, float] = {}
    chunk_map: dict[str, Chunk] = {}

    # Accumulate RRF scores from BM25 ranking
    for rank, (chunk, _bm25_score) in enumerate(bm25_results):
        cid = chunk.id
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        chunk_map[cid] = chunk

    # Accumulate RRF scores from dense ranking
    for rank, point in enumerate(dense_results):
        cid = str(point.id)
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        if cid not in chunk_map:
            chunk_map[cid] = chunk_from_scored_point(point)

    # Sort by fused score descending
    sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [(chunk_map[cid], scores[cid]) for cid in sorted_ids if cid in chunk_map]
