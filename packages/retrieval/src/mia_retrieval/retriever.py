"""Public retrieval interface: query → list[Evidence].

Supports three retrieval modes:
  - bm25   : sparse retrieval only (fast, no ML model needed)
  - dense  : vector search only (requires loaded embedder + Qdrant)
  - hybrid : BM25 + dense with RRF fusion (default, best quality)

All modes optionally pass candidates through the cross-encoder reranker.

Design decisions:
- Returns ``list[Evidence]`` (from mia_shared.schemas) rather than raw Chunks:
  the Evidence schema is the currency of the multi-agent system; downstream
  agents (Summarizer, Critic) operate on Evidence objects.
- ``retrieve_mode`` enum avoids stringly-typed mode selection in agent code.
- Async function: enables use directly in FastAPI handlers and LangGraph nodes
  without blocking the event loop.  BM25 search is synchronous (CPU-bound) but
  fast enough (<5ms) to run in the event loop without ``to_thread``.
- Ticker filter is passed through to both BM25 (post-filter on Chunk.ticker)
  and Qdrant (payload pre-filter) for consistent scoping.
"""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import Any

from mia_shared.config import get_settings
from mia_shared.schemas import Evidence

from mia_retrieval.bm25_index import BM25Index
from mia_retrieval.chunker import Chunk
from mia_retrieval.embedder import Embedder
from mia_retrieval.hybrid import reciprocal_rank_fusion
from mia_retrieval.qdrant_store import QdrantStore, chunk_from_scored_point
from mia_retrieval.reranker import Reranker

logger = logging.getLogger(__name__)

_DEFAULT_BM25_PATH = Path("data/bm25_index.pkl")


class RetrieveMode(str, Enum):
    BM25 = "bm25"
    DENSE = "dense"
    HYBRID = "hybrid"


class Retriever:
    """Stateful retriever holding references to BM25, Qdrant, Embedder, Reranker.

    Instantiate once per process (or per ARQ worker) and reuse.

    Parameters
    ----------
    qdrant     : configured :class:`QdrantStore`
    embedder   : loaded :class:`Embedder`
    bm25       : loaded :class:`BM25Index`
    reranker   : loaded :class:`Reranker`
    settings   : shared settings (for ``bm25_top_k``, ``dense_top_k``, ``rerank_top_k``)
    """

    def __init__(
        self,
        qdrant: QdrantStore,
        embedder: Embedder,
        bm25: BM25Index,
        reranker: Reranker,
        settings: Any = None,
    ) -> None:
        self._qdrant = qdrant
        self._embedder = embedder
        self._bm25 = bm25
        self._reranker = reranker
        self._settings = settings or get_settings()

    # ── Public API ────────────────────────────────────────────────────────────

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        mode: RetrieveMode = RetrieveMode.HYBRID,
        rerank: bool = True,
        ticker_filter: list[str] | None = None,
    ) -> list[Evidence]:
        """Retrieve relevant chunks for *query* and return as :class:`Evidence` objects.

        Parameters
        ----------
        query         : natural-language query
        top_k         : final number of results (default: ``settings.rerank_top_k``)
        mode          : retrieval mode (bm25 | dense | hybrid)
        rerank        : apply cross-encoder reranking to the candidate set
        ticker_filter : restrict to specific tickers (None = no restriction)

        Returns
        -------
        list[Evidence]
            Sorted by relevance score descending.
        """
        cfg = self._settings
        final_k = top_k or cfg.rerank_top_k

        candidates: list[tuple[Chunk, float]] = await self._get_candidates(
            query=query,
            mode=mode,
            bm25_top_k=cfg.bm25_top_k,
            dense_top_k=cfg.dense_top_k,
            ticker_filter=ticker_filter,
        )

        if not candidates:
            logger.debug("No candidates found for query: %r", query)
            return []

        # Optionally rerank
        if rerank and candidates:
            chunks_only = [c for c, _ in candidates]
            ranked = self._reranker.rerank(query, chunks_only, top_k=final_k)
        else:
            ranked = candidates[:final_k]

        return [_chunk_to_evidence(chunk, score) for chunk, score in ranked]

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _get_candidates(
        self,
        query: str,
        mode: RetrieveMode,
        bm25_top_k: int,
        dense_top_k: int,
        ticker_filter: list[str] | None,
    ) -> list[tuple[Chunk, float]]:
        """Run the appropriate retrieval path and return (Chunk, score) pairs."""
        bm25_results: list[tuple[Chunk, float]] = []
        dense_results: list[Any] = []

        if mode in (RetrieveMode.BM25, RetrieveMode.HYBRID):
            bm25_results = self._run_bm25(query, bm25_top_k, ticker_filter)

        if mode in (RetrieveMode.DENSE, RetrieveMode.HYBRID):
            dense_results = await self._run_dense(query, dense_top_k, ticker_filter)

        if mode == RetrieveMode.HYBRID:
            return reciprocal_rank_fusion(bm25_results, dense_results)

        if mode == RetrieveMode.BM25:
            return bm25_results

        # DENSE: reconstruct Chunk from ScoredPoint
        return [(chunk_from_scored_point(p), p.score) for p in dense_results]

    def _run_bm25(
        self,
        query: str,
        top_k: int,
        ticker_filter: list[str] | None,
    ) -> list[tuple[Chunk, float]]:
        results = self._bm25.search(query, top_k=top_k)
        if ticker_filter:
            tickers = {t.upper() for t in ticker_filter}
            results = [(c, s) for c, s in results if c.ticker.upper() in tickers]
        return results

    async def _run_dense(
        self,
        query: str,
        top_k: int,
        ticker_filter: list[str] | None,
    ) -> list[Any]:
        query_vec = self._embedder.embed_query(query).tolist()
        return await self._qdrant.search(
            query_vector=query_vec,
            top_k=top_k,
            ticker_filter=ticker_filter,
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _chunk_to_evidence(chunk: Chunk, score: float) -> Evidence:
    """Convert a :class:`Chunk` + score into an :class:`Evidence` object."""
    return Evidence(
        source_type="rag_chunk",
        ticker=chunk.ticker,
        filing_type=chunk.filing_type,
        section=chunk.section,
        text=chunk.text,
        relevance_score=score,
        metadata={
            "doc_id": chunk.id,
            "filing_id": chunk.filing_id,
            "accession_number": chunk.accession_number,
            "chunk_index": chunk.chunk_index,
            "total_chunks": chunk.total_chunks,
        },
    )


# ── Module-level factory ──────────────────────────────────────────────────────

def build_retriever(
    qdrant_url: str | None = None,
    qdrant_collection: str | None = None,
    bm25_path: Path = _DEFAULT_BM25_PATH,
) -> Retriever:
    """Convenience factory: load all components and return a ready Retriever.

    Loads the BM25 index from *bm25_path* if it exists; starts empty otherwise.
    Heavy models (Embedder, Reranker) are lazy — they load on first use.
    """
    from mia_retrieval.embedder import get_embedder
    from mia_retrieval.reranker import get_reranker

    cfg = get_settings()
    url = qdrant_url or cfg.qdrant_url
    collection = qdrant_collection or cfg.qdrant_collection

    qdrant = QdrantStore(url=url, collection=collection)
    embedder = get_embedder(model_name=cfg.embedding_model)
    reranker = get_reranker(model_name=cfg.reranker_model)

    bm25 = BM25Index()
    if bm25_path.exists():
        bm25 = BM25Index.load(bm25_path)
        logger.info("Loaded BM25 index (%d chunks) from %s", bm25.size, bm25_path)
    else:
        logger.info("No BM25 index at %s — starting empty", bm25_path)

    return Retriever(qdrant=qdrant, embedder=embedder, bm25=bm25, reranker=reranker)
