"""mia_retrieval — lazy imports to avoid loading heavy ML deps at import time.

Heavy dependencies (sentence-transformers, torch, rank_bm25) are deferred until
the specific class or function is first accessed.  This keeps test collection
fast and prevents import errors in environments where ML deps aren't installed.
"""

from __future__ import annotations

import importlib

_lazy: dict[str, str] = {
    # chunker
    "Chunk": "mia_retrieval.chunker",
    "Chunker": "mia_retrieval.chunker",
    # embedder
    "Embedder": "mia_retrieval.embedder",
    "get_embedder": "mia_retrieval.embedder",
    "EMBED_DIM": "mia_retrieval.embedder",
    # bm25
    "BM25Index": "mia_retrieval.bm25_index",
    # qdrant
    "QdrantStore": "mia_retrieval.qdrant_store",
    # hybrid
    "reciprocal_rank_fusion": "mia_retrieval.hybrid",
    # reranker
    "Reranker": "mia_retrieval.reranker",
    "get_reranker": "mia_retrieval.reranker",
    # indexer
    "FilingRecord": "mia_retrieval.indexer",
    "IndexStats": "mia_retrieval.indexer",
    "IndexingPipeline": "mia_retrieval.indexer",
    # retriever
    "RetrieveMode": "mia_retrieval.retriever",
    "Retriever": "mia_retrieval.retriever",
    "build_retriever": "mia_retrieval.retriever",
}


def __getattr__(name: str) -> object:
    if name in _lazy:
        mod = importlib.import_module(_lazy[name])
        return getattr(mod, name)
    raise AttributeError(f"module 'mia_retrieval' has no attribute {name!r}")


__all__ = list(_lazy)
