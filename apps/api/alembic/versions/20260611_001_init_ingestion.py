"""init ingestion tables

Creates mia.filings and xbrl.facts tables.
Both schemas (mia, xbrl) are pre-created by infra/init/postgres/00_init.sql.

Revision ID: 001
Revises:
Create Date: 2026-06-11
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── mia.filings ──────────────────────────────────────────────────────────
    op.create_table(
        "filings",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("ticker", sa.String(16), nullable=False),
        sa.Column("cik", sa.String(20), nullable=False),
        sa.Column("filing_type", sa.String(16), nullable=False),
        sa.Column("period_of_report", sa.Date(), nullable=True),
        sa.Column("filed_date", sa.Date(), nullable=True),
        sa.Column("accession_number", sa.String(25), nullable=False),
        sa.Column("primary_doc", sa.String(256), nullable=True),
        sa.Column("minio_path", sa.String(512), nullable=True),
        sa.Column("raw_text", sa.Text(), nullable=True),
        sa.Column(
            "status",
            sa.String(32),
            nullable=False,
            server_default="pending",
        ),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("page_count", sa.Integer(), nullable=True),
        sa.Column("char_count", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("accession_number", name="uq_filings_accession"),
        schema="mia",
    )
    op.create_index("ix_filings_ticker", "filings", ["ticker"], schema="mia")
    op.create_index("ix_filings_cik", "filings", ["cik"], schema="mia")
    op.create_index(
        "ix_filings_ticker_type",
        "filings",
        ["ticker", "filing_type"],
        schema="mia",
    )

    # ── xbrl.facts ───────────────────────────────────────────────────────────
    op.create_table(
        "facts",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("filing_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("ticker", sa.String(16), nullable=False),
        sa.Column("cik", sa.String(20), nullable=False),
        sa.Column("taxonomy", sa.String(32), nullable=False),
        sa.Column("concept", sa.String(256), nullable=False),
        sa.Column("label", sa.String(512), nullable=True),
        sa.Column("value", sa.Float(), nullable=True),
        sa.Column("unit", sa.String(32), nullable=True),
        sa.Column("period_type", sa.String(16), nullable=False),
        sa.Column("period_start", sa.Date(), nullable=True),
        sa.Column("period_end", sa.Date(), nullable=True),
        sa.Column("form", sa.String(16), nullable=True),
        sa.Column("frame", sa.String(32), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["filing_id"],
            ["mia.filings.id"],
            name="fk_xbrl_facts_filing_id",
            ondelete="SET NULL",
        ),
        schema="xbrl",
    )
    op.create_index(
        "ix_xbrl_facts_filing_id", "facts", ["filing_id"], schema="xbrl"
    )
    op.create_index("ix_xbrl_facts_ticker", "facts", ["ticker"], schema="xbrl")
    op.create_index(
        "ix_xbrl_facts_ticker_concept",
        "facts",
        ["ticker", "concept"],
        schema="xbrl",
    )
    op.create_index(
        "ix_xbrl_facts_ticker_period",
        "facts",
        ["ticker", "period_end"],
        schema="xbrl",
    )


def downgrade() -> None:
    op.drop_table("facts", schema="xbrl")
    op.drop_index("ix_filings_ticker_type", table_name="filings", schema="mia")
    op.drop_index("ix_filings_cik", table_name="filings", schema="mia")
    op.drop_index("ix_filings_ticker", table_name="filings", schema="mia")
    op.drop_table("filings", schema="mia")
