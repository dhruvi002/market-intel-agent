"""Integration-style tests for the ingestion pipeline.

All external I/O is mocked: no DB, no MinIO, no HTTP.
Tests verify the orchestration logic — status transitions, skip-if-indexed, etc.
"""

from __future__ import annotations

import uuid
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mia_ingestion.edgar.downloader import FilingMeta
from mia_ingestion.models import Filing
from mia_ingestion.pipeline import IngestionPipeline


def _make_meta(**kwargs) -> FilingMeta:
    defaults = dict(
        ticker="NVDA",
        cik="0001045810",
        filing_type="10-K",
        accession_number="0001045810-23-000017",
        filed_date=date(2023, 2, 24),
        period_of_report=date(2023, 1, 29),
        primary_doc="filing.htm",
    )
    defaults.update(kwargs)
    return FilingMeta(**defaults)


def _make_indexed_filing(accession: str = "0001045810-23-000017") -> Filing:
    return Filing(
        id=uuid.uuid4(),
        ticker="NVDA",
        cik="0001045810",
        filing_type="10-K",
        accession_number=accession,
        status="indexed",
    )


# ── _ingest_one_filing ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_skip_already_indexed_filing() -> None:
    """Pipeline must skip a filing that's already been indexed."""
    pipeline = IngestionPipeline()
    meta = _make_meta()
    existing = _make_indexed_filing()

    mock_session = AsyncMock()
    mock_session.scalar = AsyncMock(return_value=existing)
    mock_session.get = AsyncMock(return_value=existing)

    ctx_mgr = MagicMock()
    ctx_mgr.__aenter__ = AsyncMock(return_value=mock_session)
    ctx_mgr.__aexit__ = AsyncMock(return_value=None)

    dl_mock = AsyncMock()

    with patch("mia_ingestion.pipeline.get_db_session", return_value=ctx_mgr):
        result = await pipeline._ingest_one_filing(dl_mock, meta)

    assert result == existing.id
    dl_mock.download_filing_document.assert_not_called()


@pytest.mark.asyncio
async def test_no_primary_doc_returns_none() -> None:
    """Pipeline returns None and sets status=error when primary_doc is missing."""
    pipeline = IngestionPipeline()
    pipeline._minio = MagicMock()
    meta = _make_meta(primary_doc=None)

    # First DB call: no existing filing
    new_filing = Filing(
        id=uuid.uuid4(),
        ticker="NVDA",
        cik="0001045810",
        filing_type="10-K",
        accession_number="0001045810-23-000017",
        status="downloading",
    )
    mock_session = AsyncMock()
    mock_session.scalar = AsyncMock(return_value=None)
    mock_session.add = MagicMock()
    mock_session.flush = AsyncMock()
    mock_session.get = AsyncMock(return_value=new_filing)
    new_filing.id = uuid.uuid4()

    ctx_mgr = MagicMock()
    ctx_mgr.__aenter__ = AsyncMock(return_value=mock_session)
    ctx_mgr.__aexit__ = AsyncMock(return_value=None)

    dl_mock = AsyncMock()

    # _update_status also calls get_db_session
    with patch("mia_ingestion.pipeline.get_db_session", return_value=ctx_mgr):
        result = await pipeline._ingest_one_filing(dl_mock, meta)

    assert result is None
    dl_mock.download_filing_document.assert_not_called()


# ── ingest_ticker ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ingest_ticker_calls_xbrl_and_filings() -> None:
    """ingest_ticker should fetch XBRL facts and filing documents."""
    pipeline = IngestionPipeline()
    pipeline._minio = MagicMock()
    pipeline._minio.ensure_bucket = MagicMock()

    meta = _make_meta()

    mock_dl = AsyncMock()
    mock_dl.get_cik = AsyncMock(return_value="0001045810")
    mock_dl.get_xbrl_facts = AsyncMock(return_value={"facts": {}})
    mock_dl.get_recent_filings = AsyncMock(return_value=[meta])
    mock_dl.__aenter__ = AsyncMock(return_value=mock_dl)
    mock_dl.__aexit__ = AsyncMock(return_value=None)

    # _ingest_one_filing → skip (already indexed)
    existing = _make_indexed_filing()
    mock_session = AsyncMock()
    mock_session.scalar = AsyncMock(return_value=existing)
    mock_session.get = AsyncMock(return_value=existing)
    mock_session.execute = AsyncMock()
    mock_session.add_all = MagicMock()

    ctx_mgr = MagicMock()
    ctx_mgr.__aenter__ = AsyncMock(return_value=mock_session)
    ctx_mgr.__aexit__ = AsyncMock(return_value=None)

    with (
        patch("mia_ingestion.pipeline.EDGARDownloader", return_value=mock_dl),
        patch("mia_ingestion.pipeline.get_db_session", return_value=ctx_mgr),
    ):
        ids = await pipeline.ingest_ticker("NVDA", form_types=["10-K"])

    mock_dl.get_xbrl_facts.assert_called_once()
    mock_dl.get_recent_filings.assert_called()
    assert existing.id in ids
