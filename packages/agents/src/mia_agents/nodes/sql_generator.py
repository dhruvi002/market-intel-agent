"""Phase 6 SQL Generator node — NL→SQL against the XBRL facts warehouse.

Architecture
------------
1. Prompt Gemini Flash with the xbrl.facts + mia.filings schema + the user
   query → receive a single SELECT SQL statement.
2. Validate the SQL: must start with SELECT; no DDL/DML keywords allowed.
3. Execute using SQLAlchemy async (read-only pattern: no BEGIN/COMMIT mutations).
4. Format results as a markdown table and return as an Evidence object.

The DB engine is a module-level lazy singleton (same pattern as the NLI model
and embedder) — one pool per process, created on first call.

Design decisions
----------------
- LLM call is a simple HumanMessage → raw content string (no structured output):
  structured output is not needed because we immediately validate the returned
  SQL and reject anything that isn't a plain SELECT.
- SQL validation uses a whitelist approach (must start with SELECT) plus a
  blacklist of forbidden DDL/DML keywords, rather than a full SQL parser —
  fast, zero extra deps, and sufficient for the read-only constraint.
- ``_get_engine()`` uses ``@lru_cache`` so the connection pool is reused across
  calls.  ``echo=False`` in production; tests override the engine via patch.
- Results are capped at ``settings.sql_max_rows`` (default 50) to prevent
  context bloat in the Summarizer.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Any

import sqlalchemy as sa
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from mia_shared.config import get_settings
from mia_shared.schemas import AgentName, AgentState, Evidence

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Forbidden SQL keywords — single source of truth
_FORBIDDEN_RE = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|GRANT|REVOKE|EXECUTE|COPY)\b",
    re.IGNORECASE,
)

# Maximum columns rendered in the markdown table (prevent huge tables)
_MAX_COLS = 20

# NLI label entailment index (for NLI scoring in Critic — kept here for reference)
_ENTAILMENT_IDX = 2

# ── Schema description injected into the system prompt ────────────────────────

_SCHEMA = """\
PostgreSQL tables available:

1. xbrl.facts — Structured XBRL financial facts from SEC filings
   Columns: id (UUID), filing_id (UUID FK→mia.filings.id), ticker (VARCHAR),
   cik (VARCHAR), taxonomy (VARCHAR, e.g. 'us-gaap'), concept (VARCHAR,
   e.g. 'Revenues'), label (VARCHAR), value (FLOAT), unit (VARCHAR, e.g. 'USD'),
   period_type (VARCHAR: 'instant'|'duration'), period_start (DATE),
   period_end (DATE), form (VARCHAR: '10-K'|'10-Q'|'8-K'), frame (VARCHAR)

2. mia.filings — SEC filing metadata
   Columns: id (UUID), ticker (VARCHAR), cik (VARCHAR), filing_type (VARCHAR),
   period_of_report (DATE), filed_date (DATE), accession_number (VARCHAR)

Common XBRL concepts (us-gaap taxonomy):
- Revenues                                 total revenue
- RevenueFromContractWithCustomerExcludingAssessedTax  (alternate revenue tag)
- NetIncomeLoss                            net income / loss
- GrossProfit                              gross profit
- OperatingIncomeLoss                      operating income
- ResearchAndDevelopmentExpense            R&D expense
- EarningsPerShareBasic / EarningsPerShareDiluted
- CashAndCashEquivalentsAtCarryingValue    cash and equivalents
- LongTermDebt                             long-term debt
- CommonStockSharesOutstanding             shares outstanding

Ticker matching: always use UPPER(ticker) = UPPER('NVDA') for case insensitivity."""

_SYSTEM_PROMPT = """\
You are a financial data SQL expert.  Convert a natural-language question into a
single PostgreSQL SELECT query.

Rules (must follow all):
- Return ONLY the raw SQL — no markdown, no code fences, no explanation
- Only SELECT statements are allowed — no DDL, DML, or procedural SQL
- Always add LIMIT {max_rows} unless the query is explicitly a COUNT query
- Order by period_end DESC to show the most recent data first (unless question specifies otherwise)
- When filtering by ticker use: WHERE UPPER(ticker) = UPPER('NVDA')
- Use explicit schema prefixes: xbrl.facts, mia.filings

Schema:
{schema}"""


# ── Engine singleton ──────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_engine() -> AsyncEngine:
    """Lazy singleton: one SQLAlchemy async engine per process."""
    settings = get_settings()
    logger.info("sql_generator: creating async engine")
    return create_async_engine(
        settings.database_url,
        echo=False,
        pool_size=5,
        max_overflow=10,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _validate_sql(sql: str) -> tuple[bool, str]:
    """Return ``(is_valid, error_message)``.

    A query is valid iff it starts with SELECT (case-insensitive) and contains
    no forbidden DDL/DML keywords.
    """
    stripped = sql.strip()
    if not stripped.upper().startswith("SELECT"):
        return False, "Query must start with SELECT"
    if m := _FORBIDDEN_RE.search(stripped):
        return False, f"Forbidden keyword: {m.group()!r}"
    return True, ""


def _rows_to_markdown(columns: list[str], rows: list[Any]) -> str:
    """Format SQL result rows as a GitHub-flavoured markdown table.

    Columns are capped at ``_MAX_COLS``; long cell values are truncated to
    80 chars to prevent context bloat.
    """
    if not rows:
        return "*(no rows returned)*"

    cols = columns[:_MAX_COLS]
    header = " | ".join(str(c) for c in cols)
    sep = " | ".join("---" for _ in cols)
    lines = [f"| {header} |", f"| {sep} |"]
    for row in rows:
        cells = " | ".join(
            (str(v)[:80] if v is not None else "NULL")
            for v in list(row)[:_MAX_COLS]
        )
        lines.append(f"| {cells} |")
    return "\n".join(lines)


def _strip_code_fence(text: str) -> str:
    """Remove markdown ``` code fences that some LLMs add despite instructions."""
    text = text.strip()
    # ```sql\n...\n``` or ```\n...\n```
    text = re.sub(r"^```(?:sql)?\s*\n?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


# ── Node ──────────────────────────────────────────────────────────────────────

async def sql_generator_node(
    state: AgentState,
    llm: BaseChatModel | None = None,
) -> dict:
    """NL→SQL worker: converts the user query to SELECT SQL and runs it.

    Returns a state-update dict with the SQL result appended to
    ``state.evidence`` as an ``Evidence(source_type="sql_result")`` object.

    On any failure (LLM error, invalid SQL, DB error) the node returns
    ``state.evidence`` unchanged so the graph can continue with other workers'
    evidence — same graceful-degradation pattern as web_search and edgar_parser.
    """
    settings = get_settings()

    if llm is None:
        from mia_agents.llm import get_llm  # noqa: PLC0415

        llm = get_llm()

    # ── 1. Generate SQL via LLM ───────────────────────────────────────────────
    system_content = _SYSTEM_PROMPT.format(
        max_rows=settings.sql_max_rows,
        schema=_SCHEMA,
    )
    try:
        response = await llm.ainvoke(
            [
                SystemMessage(content=system_content),
                HumanMessage(content=f"Question: {state.query}\n\nSQL:"),
            ]
        )
        raw_sql = _strip_code_fence(str(response.content))
    except Exception as exc:
        logger.warning("sql_generator: LLM error: %s", exc)
        return {"evidence": state.evidence, "citations": state.citations}

    # ── 2. Validate ───────────────────────────────────────────────────────────
    is_valid, err_msg = _validate_sql(raw_sql)
    if not is_valid:
        logger.warning(
            "sql_generator: invalid SQL (%s): %r",
            err_msg,
            raw_sql[:200],
        )
        return {"evidence": state.evidence, "citations": state.citations}

    logger.info("sql_generator: executing SQL: %.200s", raw_sql)

    # ── 3. Execute ────────────────────────────────────────────────────────────
    try:
        engine = _get_engine()
        async with engine.connect() as conn:
            result = await conn.execute(sa.text(raw_sql))
            columns = list(result.keys())
            rows = result.fetchmany(settings.sql_max_rows)
    except Exception as exc:
        logger.warning("sql_generator: DB error: %s", exc)
        return {"evidence": state.evidence, "citations": state.citations}

    # ── 4. Format as Evidence ─────────────────────────────────────────────────
    table_md = _rows_to_markdown(columns, rows)
    evidence_text = (
        f"**SQL Query**\n```sql\n{raw_sql}\n```\n\n"
        f"**Results** ({len(rows)} row{'s' if len(rows) != 1 else ''})\n{table_md}"
    )

    new_evidence = Evidence(
        source_type="sql_result",
        text=evidence_text,
        metadata={
            "sql": raw_sql,
            "row_count": len(rows),
            "columns": columns,
            "agent": AgentName.SQL_GENERATOR.value,
        },
    )

    logger.info(
        "sql_generator: query returned %d rows, %d columns",
        len(rows),
        len(columns),
    )

    return {
        "evidence": [*state.evidence, new_evidence],
        "citations": state.citations,
    }
