"""Unit tests for the EDGAR downloader.

All HTTP calls are mocked — no live network access required.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mia_ingestion.edgar.downloader import EDGARDownloader, FilingMeta, _parse_date


# ── _parse_date ───────────────────────────────────────────────────────────────


def test_parse_date_valid() -> None:
    assert _parse_date("2023-01-29") == date(2023, 1, 29)


def test_parse_date_none() -> None:
    assert _parse_date(None) is None


def test_parse_date_empty_string() -> None:
    assert _parse_date("") is None


def test_parse_date_invalid_string() -> None:
    assert _parse_date("not-a-date") is None


def test_parse_date_partial() -> None:
    # EDGAR sometimes emits "2023-01" — should gracefully return None
    assert _parse_date("2023-01") is None


# ── FilingMeta dataclass ──────────────────────────────────────────────────────


def test_filing_meta_defaults() -> None:
    meta = FilingMeta(ticker="NVDA", cik="0001045810", filing_type="10-K", accession_number="abc")
    assert meta.filed_date is None
    assert meta.period_of_report is None
    assert meta.primary_doc is None


# ── EDGARDownloader.get_cik ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_cik_known_ticker() -> None:
    sample_tickers = {
        "0": {"cik_str": 1045810, "ticker": "NVDA", "title": "NVIDIA CORP"},
        "1": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    }
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = sample_tickers

    dl = EDGARDownloader()
    with patch.object(dl, "_get", AsyncMock(return_value=mock_resp)):
        cik = await dl.get_cik("NVDA")
    assert cik == "0001045810"
    await dl._client.aclose()


@pytest.mark.asyncio
async def test_get_cik_case_insensitive() -> None:
    sample_tickers = {"0": {"cik_str": 1045810, "ticker": "NVDA", "title": "NVIDIA CORP"}}
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = sample_tickers

    dl = EDGARDownloader()
    with patch.object(dl, "_get", AsyncMock(return_value=mock_resp)):
        cik = await dl.get_cik("nvda")
    assert cik == "0001045810"
    await dl._client.aclose()


@pytest.mark.asyncio
async def test_get_cik_unknown_ticker_raises() -> None:
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {}

    dl = EDGARDownloader()
    with patch.object(dl, "_get", AsyncMock(return_value=mock_resp)):
        with pytest.raises(ValueError, match="Could not resolve ticker"):
            await dl.get_cik("FAKE_XYZ")
    await dl._client.aclose()


# ── EDGARDownloader.get_recent_filings ────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_recent_filings_returns_correct_form_type() -> None:
    sample_submissions = {
        "tickers": ["NVDA"],
        "filings": {
            "recent": {
                "form": ["10-K", "10-Q", "10-K", "8-K"],
                "accessionNumber": ["acc-001", "acc-002", "acc-003", "acc-004"],
                "filingDate": ["2023-02-24", "2023-05-22", "2022-02-25", "2023-06-01"],
                "reportDate": ["2023-01-29", "2023-04-30", "2022-01-30", ""],
                "primaryDocument": ["doc1.htm", "doc2.htm", "doc3.htm", "doc4.htm"],
            }
        },
    }
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = sample_submissions

    dl = EDGARDownloader()
    with patch.object(dl, "_get", AsyncMock(return_value=mock_resp)):
        filings = await dl.get_recent_filings("0001045810", "10-K", limit=5)

    assert len(filings) == 2
    assert all(f.filing_type == "10-K" for f in filings)
    assert filings[0].accession_number == "acc-001"
    assert filings[1].accession_number == "acc-003"
    await dl._client.aclose()


@pytest.mark.asyncio
async def test_get_recent_filings_respects_limit() -> None:
    sample_submissions = {
        "tickers": ["NVDA"],
        "filings": {
            "recent": {
                "form": ["10-K"] * 10,
                "accessionNumber": [f"acc-{i:03d}" for i in range(10)],
                "filingDate": ["2023-01-01"] * 10,
                "reportDate": ["2022-12-31"] * 10,
                "primaryDocument": [f"doc{i}.htm" for i in range(10)],
            }
        },
    }
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = sample_submissions

    dl = EDGARDownloader()
    with patch.object(dl, "_get", AsyncMock(return_value=mock_resp)):
        filings = await dl.get_recent_filings("0001045810", "10-K", limit=3)

    assert len(filings) == 3
    await dl._client.aclose()


@pytest.mark.asyncio
async def test_get_recent_filings_empty_when_no_match() -> None:
    sample_submissions = {
        "tickers": ["NVDA"],
        "filings": {
            "recent": {
                "form": ["10-Q", "10-Q"],
                "accessionNumber": ["acc-001", "acc-002"],
                "filingDate": ["2023-05-22", "2023-08-22"],
                "reportDate": ["2023-04-30", "2023-07-31"],
                "primaryDocument": ["doc1.htm", "doc2.htm"],
            }
        },
    }
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = sample_submissions

    dl = EDGARDownloader()
    with patch.object(dl, "_get", AsyncMock(return_value=mock_resp)):
        filings = await dl.get_recent_filings("0001045810", "10-K", limit=5)

    assert filings == []
    await dl._client.aclose()
