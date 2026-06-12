"""End-to-end ingestion pipeline: download → MinIO → extract → Postgres.

Entry points:
  IngestionPipeline.ingest_ticker(ticker)        — all recent filings + XBRL facts
  IngestionPipeline.ingest_filing(ticker, accn)  — one specific filing by accession number

Filing status progression:
  pending → downloading → storing → extracting → indexed
                                              ↘ error (on any failure)
"""

from __future__ import annotations

import logging
import tempfile
import uuid
from pathlib import Path

from sqlalchemy import delete, select

from mia_ingestion.db import get_db_session
from mia_ingestion.edgar.downloader import EDGARDownloader, FilingMeta
from mia_ingestion.edgar.xbrl_parser import XBRLParser
from mia_ingestion.models import Filing, XBRLFact
from mia_ingestion.pdf.extractor import PDFExtractor
from mia_ingestion.storage.minio_client import MinIOClient

logger = logging.getLogger(__name__)

# Form types to ingest per ticker and how many of each to pull
_DEFAULT_FORM_TYPES = ["10-K", "10-Q", "8-K"]
_FILING_LIMITS: dict[str, int] = {"10-K": 3, "10-Q": 8, "8-K": 10}

# Hard cap on raw text stored in Postgres — keeps row sizes manageable.
# Anything beyond this is accessible via MinIO and will be chunked in Phase 2.
_TEXT_CHAR_CAP = 200_000


class IngestionPipeline:
    """Orchestrates the complete filing ingestion workflow.

    Stateless — safe to instantiate per task or share across tasks.
    """

    def __init__(self) -> None:
        self._minio = MinIOClient()
        self._extractor = PDFExtractor()

    # ── Public entry points ────────────────────────────────────────────────────

    async def ingest_ticker(
        self,
        ticker: str,
        form_types: list[str] | None = None,
    ) -> list[uuid.UUID]:
        """Ingest all recent filings and XBRL facts for a ticker.

        Returns UUIDs of all filings created or updated.
        """
        form_types = form_types or _DEFAULT_FORM_TYPES
        self._minio.ensure_bucket()

        filing_ids: list[uuid.UUID] = []

        async with EDGARDownloader() as dl:
            cik = await dl.get_cik(ticker)
            logger.info("Starting ingestion for %s (CIK %s)", ticker, cik)

            # 1. XBRL facts — one API call covers all history for this company
            try:
                facts_json = await dl.get_xbrl_facts(cik)
                n = await self._upsert_xbrl_facts(ticker, cik, facts_json)
                logger.info("XBRL: upserted %d facts for %s", n, ticker)
            except Exception as exc:
                logger.warning("XBRL facts fetch failed for %s: %s", ticker, exc)

            # 2. Filing documents
            for form_type in form_types:
                limit = _FILING_LIMITS.get(form_type, 5)
                filings = await dl.get_recent_filings(cik, form_type, limit=limit)
                for meta in filings:
                    meta.ticker = ticker  # company_tickers.json sometimes returns empty ticker
                    fid = await self._ingest_one_filing(dl, meta)
                    if fid is not None:
                        filing_ids.append(fid)

        logger.info(
            "Ingestion complete for %s: %d filing(s) processed", ticker, len(filing_ids)
        )
        return filing_ids

    async def ingest_filing(self, ticker: str, accession_number: str) -> uuid.UUID | None:
        """Ingest a single filing identified by its accession number.

        Useful for re-ingesting a specific document or for targeted updates.
        """
        self._minio.ensure_bucket()
        async with EDGARDownloader() as dl:
            cik = await dl.get_cik(ticker)
            # Fetch submissions to find the filing metadata
            meta: FilingMeta | None = None
            for form_type in _DEFAULT_FORM_TYPES:
                candidates = await dl.get_recent_filings(cik, form_type, limit=20)
                match = next(
                    (f for f in candidates if f.accession_number == accession_number), None
                )
                if match:
                    meta = match
                    meta.ticker = ticker
                    break

            if meta is None:
                raise ValueError(
                    f"Accession number {accession_number!r} not found in recent filings for {ticker}"
                )

            return await self._ingest_one_filing(dl, meta)

    # ── Internal helpers ───────────────────────────────────────────────────────

    async def _ingest_one_filing(
        self,
        dl: EDGARDownloader,
        meta: FilingMeta,
    ) -> uuid.UUID | None:
        """Download, store, extract, and persist one filing.  Returns the filing UUID."""
        # ── 1. Check for existing indexed filing ─────────────────────────────
        async with get_db_session() as session:
            existing = await session.scalar(
                select(Filing).where(Filing.accession_number == meta.accession_number)
            )
            if existing and existing.status == "indexed":
                logger.debug(
                    "Skipping %s/%s (already indexed)", meta.ticker, meta.accession_number
                )
                return existing.id

            # Create or reuse the filing row
            if existing:
                filing = existing
            else:
                filing = Filing(
                    ticker=meta.ticker,
                    cik=meta.cik,
                    filing_type=meta.filing_type,
                    accession_number=meta.accession_number,
                    filed_date=meta.filed_date,
                    period_of_report=meta.period_of_report,
                    primary_doc=meta.primary_doc,
                    status="pending",
                )
                session.add(filing)
            filing.status = "downloading"
            await session.flush()
            filing_id: uuid.UUID = filing.id

        if not meta.primary_doc:
            logger.warning(
                "No primary document recorded for %s/%s — skipping",
                meta.ticker,
                meta.accession_number,
            )
            await self._update_status(filing_id, "error", "no primary document")
            return None

        try:
            with tempfile.TemporaryDirectory(prefix="mia_edgar_") as tmpdir:
                tmp_path = Path(tmpdir)

                # ── 2. Download document ──────────────────────────────────
                local_file = await dl.download_filing_document(
                    cik=meta.cik,
                    accession_number=meta.accession_number,
                    primary_doc=meta.primary_doc,
                    download_dir=tmp_path,
                )
                await self._update_status(filing_id, "storing")

                # ── 3. Upload to MinIO ────────────────────────────────────
                key = MinIOClient.filing_key(
                    meta.ticker,
                    meta.filing_type,
                    meta.accession_number,
                    meta.primary_doc,
                )
                minio_path = await self._minio.async_upload_file(local_file, key)
                await self._update_status(filing_id, "extracting")

                # ── 4. Extract text & sections ────────────────────────────
                doc = await self._extractor.extract(local_file)

                # ── 5. Persist results ────────────────────────────────────
                async with get_db_session() as session:
                    f = await session.get(Filing, filing_id)
                    if f:
                        f.minio_path = minio_path
                        f.raw_text = doc.text[:_TEXT_CHAR_CAP]
                        f.page_count = doc.page_count
                        f.char_count = doc.char_count
                        f.status = "indexed"
                        f.error_message = None

                logger.info(
                    "Indexed %s %s %s — %d chars, %s backend",
                    meta.ticker,
                    meta.filing_type,
                    meta.accession_number,
                    doc.char_count,
                    doc.extraction_backend,
                )
                return filing_id

        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            logger.error(
                "Failed to ingest %s/%s: %s",
                meta.ticker,
                meta.accession_number,
                msg,
                exc_info=True,
            )
            await self._update_status(filing_id, "error", msg)
            return None

    async def _upsert_xbrl_facts(
        self,
        ticker: str,
        cik: str,
        facts_json: dict,
    ) -> int:
        """Parse XBRL facts JSON and replace all existing facts for this ticker.

        Full replace (delete + insert) is safe here because the companyfacts API
        returns the complete history in one response.
        """
        parser = XBRLParser(ticker=ticker, cik=cik)
        facts = parser.parse(facts_json)
        if not facts:
            return 0

        async with get_db_session() as session:
            await session.execute(
                delete(XBRLFact).where(XBRLFact.ticker == ticker)
            )
            session.add_all(facts)

        return len(facts)

    async def _update_status(
        self,
        filing_id: uuid.UUID,
        status: str,
        error: str | None = None,
    ) -> None:
        async with get_db_session() as session:
            f = await session.get(Filing, filing_id)
            if f:
                f.status = status
                if error is not None:
                    f.error_message = error
