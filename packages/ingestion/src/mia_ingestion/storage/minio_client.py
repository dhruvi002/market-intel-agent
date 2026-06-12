"""MinIO client for storing and retrieving raw SEC filing documents.

Uses boto3 pointed at the local MinIO instance (S3-compatible API).
All operations are synchronous — call from asyncio.to_thread() if needed,
or use the async wrappers below.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from mia_shared.config import get_settings

logger = logging.getLogger(__name__)


class MinIOClient:
    """Thin boto3 S3 wrapper pointing at the local MinIO instance."""

    def __init__(self) -> None:
        settings = get_settings()
        self._bucket = settings.minio_bucket_filings
        scheme = "https" if settings.minio_secure else "http"
        self._client = boto3.client(
            "s3",
            endpoint_url=f"{scheme}://{settings.minio_endpoint}",
            aws_access_key_id=settings.minio_access_key,
            aws_secret_access_key=settings.minio_secret_key.get_secret_value(),
            config=Config(signature_version="s3v4"),
        )

    # ── Bucket management ─────────────────────────────────────────────────────

    def ensure_bucket(self) -> None:
        """Create the configured bucket if it does not already exist."""
        try:
            self._client.head_bucket(Bucket=self._bucket)
            logger.debug("Bucket %r already exists", self._bucket)
        except ClientError as exc:
            code = exc.response["Error"]["Code"]
            if code in ("404", "NoSuchBucket"):
                self._client.create_bucket(Bucket=self._bucket)
                logger.info("Created MinIO bucket %r", self._bucket)
            else:
                raise

    # ── Key helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def filing_key(ticker: str, filing_type: str, accession: str, filename: str) -> str:
        """Return the canonical S3 key for a filing document.

        Example: filings/NVDA/10-K/0001045810-23-000017/primary-document.htm
        """
        return f"filings/{ticker.upper()}/{filing_type}/{accession}/{filename}"

    # ── Sync I/O ──────────────────────────────────────────────────────────────

    def upload_file(self, local_path: Path, key: str) -> str:
        """Upload a local file to MinIO. Returns the s3:// URI."""
        self._client.upload_file(str(local_path), self._bucket, key)
        uri = f"s3://{self._bucket}/{key}"
        logger.debug("Uploaded %s → %s", local_path.name, uri)
        return uri

    def download_file(self, key: str, local_path: Path) -> None:
        """Download an object from MinIO to a local path."""
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self._client.download_file(self._bucket, key, str(local_path))

    def object_exists(self, key: str) -> bool:
        """Return True if the key already exists in the bucket."""
        try:
            self._client.head_object(Bucket=self._bucket, Key=key)
            return True
        except ClientError as exc:
            if exc.response["Error"]["Code"] == "404":
                return False
            raise

    # ── Async wrappers ────────────────────────────────────────────────────────

    async def async_upload_file(self, local_path: Path, key: str) -> str:
        return await asyncio.to_thread(self.upload_file, local_path, key)

    async def async_download_file(self, key: str, local_path: Path) -> None:
        await asyncio.to_thread(self.download_file, key, local_path)
