"""Qdrant collection management, vector upsert, and dense search.

Design decisions:
- AsyncQdrantClient: matches the async FastAPI / ARQ worker event loops.
  Qdrant's gRPC transport (port 6334) is faster for bulk upserts; we use REST
  (port 6333) for simplicity since upserts are batched in the indexer.
- Payload indexes on ``ticker`` and ``filing_type``: Qdrant requires explicit
  payload index creation for field-based filtering.  Without them, filtered
  search degrades to a full-scan post-filter.
- Cosine distance + normalised vectors: bge-large embeddings are L2-normalised
  in the Embedder, so cosine similarity equals dot product — Qdrant optimises
  HNSW builds accordingly.
- Upsert (not insert): Qdrant's upsert is idempotent on the point ID, so
  re-indexing the same filing's chunks is safe.
- Chunk IDs are UUID strings (Qdrant accepts both int and UUID point IDs).
- ``ensure_collection`` is idempotent: safe to call on every startup.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from mia_retrieval.chunker import Chunk

logger = logging.getLogger(__name__)

VECTOR_DIM: int = 1024   # bge-large-en-v1.5


class QdrantStore:
    """Async Qdrant client wrapper.

    Parameters
    ----------
    url        : Qdrant REST endpoint, e.g. ``http://localhost:6333``
    collection : collection name (default: ``filings``)
    """

    def __init__(self, url: str, collection: str = "filings") -> None:
        self._url = url
        self._collection = collection
        self._client: Any = None   # qdrant_client.AsyncQdrantClient

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def ensure_collection(self) -> None:
        """Create the Qdrant collection (and payload indexes) if absent.

        Safe to call on every startup — no-op if already exists.
        """
        from qdrant_client.models import (  # noqa: PLC0415
            Distance,
            PayloadSchemaType,
            VectorParams,
        )

        client = await self._get_client()

        existing = {c.name for c in (await client.get_collections()).collections}
        if self._collection not in existing:
            await client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
            )
            logger.info("Created Qdrant collection %r (dim=%d)", self._collection, VECTOR_DIM)

        # Payload indexes — create only if collection was just created OR
        # if they don't exist yet.  Qdrant silently skips existing indexes.
        await client.create_payload_index(
            self._collection, "ticker", PayloadSchemaType.KEYWORD
        )
        await client.create_payload_index(
            self._collection, "filing_type", PayloadSchemaType.KEYWORD
        )

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    # ── Write ─────────────────────────────────────────────────────────────────

    async def upsert(self, chunks: list[Chunk], embeddings: np.ndarray) -> int:
        """Upsert *chunks* with their pre-computed *embeddings* into Qdrant.

        Parameters
        ----------
        chunks     : list of :class:`~mia_retrieval.chunker.Chunk`
        embeddings : (N, EMBED_DIM) float32 array — must be L2-normalised

        Returns
        -------
        int
            Number of points upserted.
        """
        if not chunks:
            return 0

        from qdrant_client.models import PointStruct  # noqa: PLC0415

        points = [
            PointStruct(
                id=chunk.id,
                vector=embeddings[i].tolist(),
                payload={
                    "filing_id": chunk.filing_id,
                    "ticker": chunk.ticker,
                    "filing_type": chunk.filing_type,
                    "accession_number": chunk.accession_number,
                    "section": chunk.section,
                    "text": chunk.text,
                    "chunk_index": chunk.chunk_index,
                    "total_chunks": chunk.total_chunks,
                },
            )
            for i, chunk in enumerate(chunks)
        ]

        client = await self._get_client()
        await client.upsert(collection_name=self._collection, points=points)
        logger.debug("Upserted %d points into %r", len(points), self._collection)
        return len(points)

    async def delete_by_filing(self, filing_id: str) -> int:
        """Delete all points for *filing_id*.  Used when re-indexing a filing."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue  # noqa: PLC0415

        client = await self._get_client()
        result = await client.delete(
            collection_name=self._collection,
            points_selector=Filter(
                must=[FieldCondition(key="filing_id", match=MatchValue(value=filing_id))]
            ),
        )
        deleted = getattr(result, "deleted", 0) or 0
        logger.debug("Deleted %d points for filing %s", deleted, filing_id)
        return deleted

    # ── Read ──────────────────────────────────────────────────────────────────

    async def search(
        self,
        query_vector: list[float],
        top_k: int,
        ticker_filter: list[str] | None = None,
    ) -> list[Any]:
        """Dense vector search.  Returns list of ``ScoredPoint`` objects.

        Parameters
        ----------
        query_vector   : L2-normalised query embedding (length 1024)
        top_k          : maximum results to return
        ticker_filter  : if provided, restrict to these tickers only
        """
        from qdrant_client.models import FieldCondition, Filter, MatchAny  # noqa: PLC0415

        query_filter = None
        if ticker_filter:
            query_filter = Filter(
                must=[FieldCondition(key="ticker", match=MatchAny(any=ticker_filter))]
            )

        client = await self._get_client()
        return await client.search(
            collection_name=self._collection,
            query_vector=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )

    async def count(self) -> int:
        """Return the total number of indexed points."""
        client = await self._get_client()
        result = await client.count(collection_name=self._collection)
        return result.count

    async def filing_is_indexed(self, filing_id: str) -> bool:
        """Return True if at least one chunk for *filing_id* exists in Qdrant."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue  # noqa: PLC0415

        client = await self._get_client()
        result = await client.count(
            collection_name=self._collection,
            count_filter=Filter(
                must=[FieldCondition(key="filing_id", match=MatchValue(value=filing_id))]
            ),
        )
        return result.count > 0

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _get_client(self) -> Any:
        if self._client is None:
            from qdrant_client import AsyncQdrantClient  # noqa: PLC0415

            self._client = AsyncQdrantClient(url=self._url)
        return self._client


# ── Helpers ───────────────────────────────────────────────────────────────────

def chunk_from_scored_point(point: Any) -> Chunk:
    """Reconstruct a :class:`Chunk` from a Qdrant ``ScoredPoint`` payload."""
    p = point.payload
    return Chunk(
        id=str(point.id),
        filing_id=p.get("filing_id", ""),
        ticker=p.get("ticker", ""),
        filing_type=p.get("filing_type", ""),
        accession_number=p.get("accession_number", ""),
        section=p.get("section"),
        text=p.get("text", ""),
        chunk_index=p.get("chunk_index", 0),
        total_chunks=p.get("total_chunks", 1),
    )
