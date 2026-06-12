"""Indexing pipeline: filing text → chunks → Qdrant + BM25.

Design decisions:
- FilingRecord dataclass: the indexer accepts plain data objects rather than
  SQLAlchemy ORM instances.  This decouples mia_retrieval from mia_ingestion's
  DB layer — the caller (script or ARQ task) loads ORM objects and converts
  them.  No circular imports, no hidden SQLAlchemy session state.
- Skip-if-indexed: ``QdrantStore.filing_is_indexed`` checks before processing
  each filing.  Re-running ``make index ticker=NVDA`` is fully idempotent.
  To force re-index, pass ``force=True``.
- Batch embedding: we embed ``batch_size`` (default 32) chunks at once to
  maximise GPU/CPU throughput.  Embedding one chunk at a time would be 30×
  slower due to model overhead.
- BM25 rebuild-on-add: each ``index_filing`` call appends new chunks to the
  BM25 index (triggering a full rebuild).  For Phase 2 scale (few hundred
  filings, ~20k chunks), rebuild takes <1s.  At S&P-100 scale (~1500 filings,
  ~150k chunks), consider batching all filings before a single rebuild.
- IndexStats: a structured return value for observability — consumed by the
  ARQ task wrapper and logged to Langfuse in Phase 8.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from mia_retrieval.bm25_index import BM25Index
from mia_retrieval.chunker import Chunk, Chunker
from mia_retrieval.embedder import Embedder
from mia_retrieval.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)

_DEFAULT_BM25_PATH = Path("data/bm25_index.pkl")


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class FilingRecord:
    """Minimal filing data needed by the indexer.

    Populated by the caller from ``mia_ingestion.models.Filing`` ORM objects.
    """

    id: str               # filing UUID as string
    ticker: str           # e.g. "NVDA"
    filing_type: str      # "10-K" | "10-Q" | "8-K"
    accession_number: str
    raw_text: str | None


@dataclass
class IndexStats:
    """Outcome of one indexing run."""

    ticker: str
    filings_processed: int = 0
    filings_skipped: int = 0    # already indexed + force=False
    filings_empty: int = 0      # raw_text is None or blank
    chunks_created: int = 0
    chunks_upserted: int = 0    # written to Qdrant
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return not self.errors


# ── Pipeline ──────────────────────────────────────────────────────────────────

class IndexingPipeline:
    """Orchestrates chunk → embed → upsert (Qdrant + BM25) for filing text.

    Parameters
    ----------
    qdrant      : configured :class:`QdrantStore`
    embedder    : loaded or lazy :class:`Embedder`
    bm25        : in-memory :class:`BM25Index` (pre-loaded or empty)
    bm25_path   : where to persist the BM25 index after each update
    chunker     : :class:`Chunker` (uses package defaults if not provided)
    batch_size  : chunks per embedding batch (default 32)
    """

    def __init__(
        self,
        qdrant: QdrantStore,
        embedder: Embedder,
        bm25: BM25Index,
        bm25_path: Path = _DEFAULT_BM25_PATH,
        chunker: Chunker | None = None,
        batch_size: int = 32,
    ) -> None:
        self._qdrant = qdrant
        self._embedder = embedder
        self._bm25 = bm25
        self._bm25_path = bm25_path
        self._chunker = chunker or Chunker()
        self._batch_size = batch_size

    # ── Public API ────────────────────────────────────────────────────────────

    async def ensure_ready(self) -> None:
        """Idempotent setup: create Qdrant collection + payload indexes."""
        await self._qdrant.ensure_collection()

    async def index_ticker(
        self,
        ticker: str,
        filings: list[FilingRecord],
        force: bool = False,
    ) -> IndexStats:
        """Index all *filings* for *ticker*.

        Parameters
        ----------
        ticker   : used for logging and returned in :class:`IndexStats`
        filings  : list of :class:`FilingRecord` to index
        force    : if True, re-index even if already present in Qdrant

        Returns
        -------
        :class:`IndexStats`
        """
        stats = IndexStats(ticker=ticker)

        for record in filings:
            try:
                result = await self._index_one(record, force=force)
                if result == "skipped":
                    stats.filings_skipped += 1
                elif result == "empty":
                    stats.filings_empty += 1
                else:
                    stats.filings_processed += 1
                    stats.chunks_created += result["chunks_created"]
                    stats.chunks_upserted += result["chunks_upserted"]
            except Exception as exc:  # noqa: BLE001
                msg = f"[{record.id}] {exc}"
                logger.exception("Error indexing filing %s: %s", record.id, exc)
                stats.errors.append(msg)

        # Persist updated BM25 index after processing all filings for ticker
        if stats.chunks_created > 0:
            self._bm25.save(self._bm25_path)
            logger.info(
                "BM25 index saved (%d total chunks)", self._bm25.size
            )

        logger.info(
            "Indexed %s: %d processed, %d skipped, %d empty, %d chunks, %d errors",
            ticker,
            stats.filings_processed,
            stats.filings_skipped,
            stats.filings_empty,
            stats.chunks_created,
            len(stats.errors),
        )
        return stats

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _index_one(
        self,
        record: FilingRecord,
        force: bool,
    ) -> str | dict[str, int]:
        """Index a single filing.

        Returns "skipped", "empty", or {"chunks_created": N, "chunks_upserted": N}.
        """
        # Skip if already indexed and not forcing
        if not force and await self._qdrant.filing_is_indexed(record.id):
            logger.debug("Skipping already-indexed filing %s", record.id)
            return "skipped"

        if not record.raw_text or not record.raw_text.strip():
            logger.debug("Skipping filing %s (no raw_text)", record.id)
            return "empty"

        # Chunk the filing text
        chunks = self._chunker.chunk(
            text=record.raw_text,
            filing_id=record.id,
            ticker=record.ticker,
            filing_type=record.filing_type,
            accession_number=record.accession_number,
        )

        if not chunks:
            logger.debug("Skipping filing %s (text too short to chunk)", record.id)
            return "empty"

        # Embed and upsert in batches
        total_upserted = 0
        all_chunks: list[Chunk] = []
        for batch_start in range(0, len(chunks), self._batch_size):
            batch = chunks[batch_start : batch_start + self._batch_size]
            texts = [c.text for c in batch]
            embeddings = self._embedder.embed(texts)
            upserted = await self._qdrant.upsert(batch, embeddings)
            total_upserted += upserted
            all_chunks.extend(batch)

        # Add to BM25 index (triggers rebuild)
        self._bm25.add(all_chunks)

        logger.info(
            "Indexed filing %s (%s %s): %d chunks",
            record.id,
            record.ticker,
            record.filing_type,
            len(chunks),
        )
        return {"chunks_created": len(chunks), "chunks_upserted": total_upserted}
