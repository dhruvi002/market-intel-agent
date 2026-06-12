"""mia-ingestion: EDGAR downloader, XBRL→Postgres ETL, PDF pipeline, MinIO storage."""

from __future__ import annotations

# Lazy import — avoids loading heavy deps (boto3, docling, etc.) at package import time.
# Import directly: from mia_ingestion.pipeline import IngestionPipeline


def __getattr__(name: str):
    if name == "IngestionPipeline":
        from mia_ingestion.pipeline import IngestionPipeline

        return IngestionPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["IngestionPipeline"]
