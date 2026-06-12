"""Alembic migration environment.

Async-aware: uses create_async_engine so Alembic can run against the
same asyncpg-backed database that the application uses.

All ORM model Bases must be imported here so that autogenerate can
detect schema changes across every table in the codebase.
"""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy.ext.asyncio import create_async_engine

# ── Import every ORM Base so autogenerate sees all tables ─────────────────────
# Add new packages' Bases here as they introduce models.
from mia_ingestion.models import Base as IngestionBase  # noqa: F401

# Merge all metadata into one target.  For now only ingestion models exist;
# Phase 4+ will add agent / session tables here.
target_metadata = IngestionBase.metadata

# ─────────────────────────────────────────────────────────────────────────────
config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)


def _get_url() -> str:
    from mia_shared.config import get_settings

    return get_settings().database_url


def run_migrations_offline() -> None:
    """Generate SQL without a live DB connection (for review / audit)."""
    context.configure(
        url=_get_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_schemas=True,
        version_table_schema="mia",
    )
    with context.begin_transaction():
        context.run_migrations()


def _do_migrations(connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        include_schemas=True,
        # Keep Alembic's version table in the mia schema alongside our tables
        version_table_schema="mia",
        # Emit CREATE SCHEMA IF NOT EXISTS for schemas referenced by models
        include_object=_include_object,
    )
    with context.begin_transaction():
        context.run_migrations()


def _include_object(object, name, type_, reflected, compare_to):  # noqa: A002
    """Tell autogenerate to track objects in all non-public schemas."""
    return True


async def run_migrations_online() -> None:
    """Run migrations against a live DB via an async engine."""
    engine = create_async_engine(_get_url(), echo=False)
    async with engine.connect() as conn:
        await conn.run_sync(_do_migrations)
    await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
