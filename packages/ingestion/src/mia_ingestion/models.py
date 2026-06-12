"""SQLAlchemy 2.0 ORM models for the ingestion pipeline.

Tables live in two schemas:
  mia.filings   — one row per SEC filing (10-K / 10-Q / 8-K)
  xbrl.facts    — structured XBRL financial facts per company
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from typing import Optional

from sqlalchemy import (
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ── mia.filings ──────────────────────────────────────────────────────────────

class Filing(Base):
    """One row per SEC filing document ingested into the system."""

    __tablename__ = "filings"
    __table_args__ = (
        UniqueConstraint("accession_number", name="uq_filings_accession"),
        Index("ix_filings_ticker_type", "ticker", "filing_type"),
        {"schema": "mia"},
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    ticker: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    cik: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    # e.g. "10-K" | "10-Q" | "8-K"
    filing_type: Mapped[str] = mapped_column(String(16), nullable=False)
    period_of_report: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    filed_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    # EDGAR accession number, e.g. "0001045810-23-000017"
    accession_number: Mapped[str] = mapped_column(String(25), nullable=False)
    # Filename of the primary document within the filing
    primary_doc: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    # s3://sec-filings/filings/NVDA/10-K/...
    minio_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    # Extracted plain text (capped at 200k chars to keep row sizes reasonable)
    raw_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # pending | downloading | storing | extracting | indexed | error
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="pending"
    )
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    page_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    char_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow
    )

    xbrl_facts: Mapped[list[XBRLFact]] = relationship(
        "XBRLFact", back_populates="filing", lazy="selectin"
    )


# ── xbrl.facts ───────────────────────────────────────────────────────────────

class XBRLFact(Base):
    """One row per XBRL financial data point from EDGAR's companyfacts API."""

    __tablename__ = "facts"
    __table_args__ = (
        Index("ix_xbrl_facts_ticker_concept", "ticker", "concept"),
        Index("ix_xbrl_facts_ticker_period", "ticker", "period_end"),
        {"schema": "xbrl"},
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    # Nullable FK — XBRL facts are loaded per-ticker, not per-filing
    filing_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("mia.filings.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    ticker: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    cik: Mapped[str] = mapped_column(String(20), nullable=False)
    # e.g. "us-gaap" | "dei" | "ifrs-full"
    taxonomy: Mapped[str] = mapped_column(String(32), nullable=False)
    # e.g. "Revenues", "NetIncomeLoss"
    concept: Mapped[str] = mapped_column(String(256), nullable=False)
    label: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    # "USD" | "shares" | "pure"
    unit: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    # "instant" (balance-sheet snapshot) | "duration" (income/cashflow period)
    period_type: Mapped[str] = mapped_column(String(16), nullable=False)
    period_start: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    period_end: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    # "10-K" | "10-Q" | "20-F" etc.
    form: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    # EDGAR frame tag, e.g. "CY2023Q4I"
    frame: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    filing: Mapped[Optional[Filing]] = relationship(
        "Filing", back_populates="xbrl_facts"
    )
