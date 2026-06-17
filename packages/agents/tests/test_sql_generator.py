"""Tests for packages/agents/src/mia_agents/nodes/sql_generator.py

All tests are fully mocked — no real LLM calls, no real DB connections.

Patch targets
-------------
- ``mia_agents.nodes.sql_generator._get_engine``        — SQLAlchemy engine
- ``mia_agents.nodes.sql_generator.get_settings``       — settings
- LLM: passed directly as a ``MagicMock`` into ``sql_generator_node``
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from mia_agents.nodes.sql_generator import (
    _rows_to_markdown,
    _strip_code_fence,
    _validate_sql,
    sql_generator_node,
)
from mia_shared.schemas import AgentState


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_state(query: str = "What is NVDA revenue?") -> AgentState:
    return AgentState(session_id=uuid4(), query=query)


def _fake_settings(**overrides):
    s = MagicMock()
    s.sql_max_rows = 50
    s.database_url = "postgresql+asyncpg://mia:mia@localhost/market_intel"
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


def _make_llm(sql: str):
    """Return a mock LLM whose ainvoke resolves to a message with *sql* content."""
    llm = MagicMock()
    msg = MagicMock()
    msg.content = sql
    llm.ainvoke = AsyncMock(return_value=msg)
    return llm


def _make_engine_mock(columns: list[str], rows: list[tuple]):
    """Return a context-manager mock for create_async_engine / conn.execute."""
    result_mock = MagicMock()
    result_mock.keys.return_value = columns
    result_mock.fetchmany.return_value = rows

    conn_mock = AsyncMock()
    conn_mock.execute = AsyncMock(return_value=result_mock)
    conn_mock.__aenter__ = AsyncMock(return_value=conn_mock)
    conn_mock.__aexit__ = AsyncMock(return_value=False)

    engine_mock = MagicMock()
    engine_mock.connect.return_value = conn_mock
    engine_mock.dispose = AsyncMock()
    return engine_mock


def _run(coro):
    return asyncio.run(coro)


# ── Unit: _validate_sql ───────────────────────────────────────────────────────

class TestValidateSQL:
    def test_valid_select(self):
        ok, msg = _validate_sql("SELECT ticker, value FROM xbrl.facts LIMIT 10")
        assert ok is True
        assert msg == ""

    def test_lowercase_select(self):
        ok, _ = _validate_sql("select * from xbrl.facts")
        assert ok is True

    def test_rejects_non_select(self):
        ok, msg = _validate_sql("UPDATE xbrl.facts SET value=0")
        assert ok is False
        assert "SELECT" in msg

    def test_rejects_drop(self):
        ok, msg = _validate_sql("SELECT 1; DROP TABLE xbrl.facts")
        assert ok is False
        assert "DROP" in msg.upper()

    def test_rejects_insert(self):
        ok, msg = _validate_sql("INSERT INTO xbrl.facts VALUES (1,2,3)")
        assert ok is False

    def test_rejects_delete(self):
        ok, msg = _validate_sql("DELETE FROM xbrl.facts")
        assert ok is False

    def test_rejects_create(self):
        ok, msg = _validate_sql("CREATE TABLE hack (id int)")
        assert ok is False


# ── Unit: _strip_code_fence ───────────────────────────────────────────────────

class TestStripCodeFence:
    def test_strips_sql_fence(self):
        result = _strip_code_fence("```sql\nSELECT 1\n```")
        assert result == "SELECT 1"

    def test_strips_plain_fence(self):
        result = _strip_code_fence("```\nSELECT 2\n```")
        assert result == "SELECT 2"

    def test_passthrough_bare_sql(self):
        result = _strip_code_fence("SELECT ticker FROM xbrl.facts")
        assert result == "SELECT ticker FROM xbrl.facts"

    def test_strips_trailing_whitespace(self):
        result = _strip_code_fence("  SELECT 1  ")
        assert result == "SELECT 1"


# ── Unit: _rows_to_markdown ───────────────────────────────────────────────────

class TestRowsToMarkdown:
    def test_empty_rows(self):
        result = _rows_to_markdown(["a", "b"], [])
        assert "no rows" in result.lower()

    def test_single_row(self):
        result = _rows_to_markdown(["ticker", "value"], [("NVDA", 22600000000.0)])
        assert "NVDA" in result
        assert "ticker" in result

    def test_null_cell(self):
        result = _rows_to_markdown(["x"], [(None,)])
        assert "NULL" in result


# ── Integration: sql_generator_node ──────────────────────────────────────────

class TestSQLGeneratorNode:
    """End-to-end tests for sql_generator_node with mocked LLM and DB."""

    def _run_node(self, state, llm, engine_mock, settings=None):
        """Run node with patched engine and settings."""
        if settings is None:
            settings = _fake_settings()
        with (
            patch(
                "mia_agents.nodes.sql_generator._get_engine",
                return_value=engine_mock,
            ),
            patch(
                "mia_agents.nodes.sql_generator.get_settings",
                return_value=settings,
            ),
        ):
            return _run(sql_generator_node(state, llm=llm))

    def test_successful_query_adds_evidence(self):
        state = _make_state("NVDA quarterly revenue")
        llm = _make_llm("SELECT ticker, value FROM xbrl.facts WHERE ticker='NVDA' LIMIT 50")
        engine = _make_engine_mock(["ticker", "value"], [("NVDA", 22.6e9)])

        result = self._run_node(state, llm, engine)

        assert len(result["evidence"]) == 1
        ev = result["evidence"][0]
        assert ev.source_type == "sql_result"
        assert "SQL Query" in ev.text
        assert "NVDA" in ev.text
        assert ev.metadata["row_count"] == 1

    def test_evidence_accumulates_with_existing(self):
        """New evidence is appended, not replaced."""
        from mia_shared.schemas import Evidence

        existing = Evidence(source_type="rag_chunk", text="existing")
        state = _make_state()
        state = state.model_copy(update={"evidence": [existing]})
        llm = _make_llm("SELECT * FROM xbrl.facts LIMIT 50")
        engine = _make_engine_mock(["id"], [("abc",)])

        result = self._run_node(state, llm, engine)

        assert len(result["evidence"]) == 2
        assert result["evidence"][0].source_type == "rag_chunk"
        assert result["evidence"][1].source_type == "sql_result"

    def test_llm_error_returns_unchanged_state(self):
        state = _make_state()
        llm = MagicMock()
        llm.ainvoke = AsyncMock(side_effect=RuntimeError("LLM quota exceeded"))
        engine = _make_engine_mock([], [])

        result = self._run_node(state, llm, engine)

        assert result["evidence"] == state.evidence
        assert result["citations"] == state.citations

    def test_invalid_sql_returns_unchanged_state(self):
        """LLM returns DROP instead of SELECT → validation fails → no evidence."""
        state = _make_state()
        llm = _make_llm("DROP TABLE xbrl.facts")
        engine = _make_engine_mock([], [])

        result = self._run_node(state, llm, engine)

        assert result["evidence"] == state.evidence

    def test_db_error_returns_unchanged_state(self):
        state = _make_state()
        llm = _make_llm("SELECT * FROM xbrl.facts LIMIT 50")
        engine = MagicMock()
        conn_mock = AsyncMock()
        conn_mock.execute = AsyncMock(side_effect=Exception("connection refused"))
        conn_mock.__aenter__ = AsyncMock(return_value=conn_mock)
        conn_mock.__aexit__ = AsyncMock(return_value=False)
        engine.connect.return_value = conn_mock

        result = self._run_node(state, llm, engine)

        assert result["evidence"] == state.evidence

    def test_code_fence_stripped_from_llm_output(self):
        """LLM wraps SQL in markdown fences despite instructions — should still work."""
        state = _make_state()
        llm = _make_llm("```sql\nSELECT ticker FROM xbrl.facts LIMIT 50\n```")
        engine = _make_engine_mock(["ticker"], [("MSFT",)])

        result = self._run_node(state, llm, engine)

        assert len(result["evidence"]) == 1
        # The stored SQL should be clean (no fences)
        assert "```sql" not in result["evidence"][0].metadata["sql"]

    def test_zero_rows_produces_evidence(self):
        """A valid query returning no rows still produces an Evidence object."""
        state = _make_state()
        llm = _make_llm("SELECT * FROM xbrl.facts WHERE ticker='FAKE' LIMIT 50")
        engine = _make_engine_mock(["ticker", "value"], [])

        result = self._run_node(state, llm, engine)

        assert len(result["evidence"]) == 1
        assert result["evidence"][0].metadata["row_count"] == 0
        assert "no rows" in result["evidence"][0].text.lower()

    def test_evidence_metadata_columns_recorded(self):
        state = _make_state()
        llm = _make_llm("SELECT ticker, concept, value FROM xbrl.facts LIMIT 50")
        engine = _make_engine_mock(["ticker", "concept", "value"], [("NVDA", "Revenues", 1.0)])

        result = self._run_node(state, llm, engine)

        cols = result["evidence"][0].metadata["columns"]
        assert cols == ["ticker", "concept", "value"]

    def test_citations_passthrough_unchanged(self):
        """sql_generator_node never modifies citations."""
        from mia_shared.schemas import Citation, Evidence

        ev = Evidence(source_type="rag_chunk", text="x")
        cit = Citation(evidence_id=ev.id, claim_text="NVDA revenue exceeded $20B")
        state = _make_state()
        state = state.model_copy(update={"evidence": [ev], "citations": [cit]})

        llm = _make_llm("SELECT * FROM xbrl.facts LIMIT 50")
        engine = _make_engine_mock(["val"], [(1,)])

        result = self._run_node(state, llm, engine)

        assert len(result["citations"]) == 1
        assert result["citations"][0].claim_text == "NVDA revenue exceeded $20B"
