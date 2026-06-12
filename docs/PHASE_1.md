# Phase 1 — Ingestion Pipelines

**Status:** ✅ Complete
**Duration:** ~1 session
**Unlocks:** Phase 2 (retrieval stack — Qdrant indexing, BM25, hybrid search)

---

## What Was Built

### `packages/ingestion/src/mia_ingestion/`

| File | Purpose |
|---|---|
| `models.py` | SQLAlchemy 2.0 ORM: `mia.filings`, `xbrl.facts` tables |
| `db.py` | Async engine + session factory (`get_db_session` context manager) |
| `edgar/downloader.py` | Async EDGAR client: CIK resolution, filings list, document download, XBRL facts |
| `edgar/xbrl_parser.py` | Parses EDGAR companyfacts JSON → `XBRLFact` ORM objects |
| `pdf/extractor.py` | Docling (primary) + PyMuPDF (fallback) + BeautifulSoup for HTML filings |
| `storage/minio_client.py` | boto3 S3 wrapper pointing at local MinIO |
| `pipeline.py` | Orchestrates the full workflow: download → MinIO → extract → Postgres |

### `apps/api/alembic/`

| File | Purpose |
|---|---|
| `alembic.ini` | Alembic config pointing to `apps/api/alembic/` |
| `alembic/env.py` | Async-aware env using `create_async_engine`; imports all ORM Bases |
| `alembic/script.py.mako` | Migration file template |
| `alembic/versions/20260611_001_init_ingestion.py` | Creates `mia.filings` + `xbrl.facts` + indexes |

### `apps/worker/src/mia_worker/`

| File | Purpose |
|---|---|
| `tasks/ingest.py` | `ingest_ticker` + `ingest_filing` ARQ tasks |
| `main.py` | `WorkerSettings` class; `make worker` entry point |

### Scripts & Makefile

| Target / File | Purpose |
|---|---|
| `scripts/init_minio.py` | Bootstrap MinIO bucket (`make init-minio`) |
| `scripts/ingest_ticker.py` | CLI entry point for manual ingestion (`make ingest ticker=NVDA`) |
| `make migrate` | Run Alembic migrations |
| `make init-minio` | Create `sec-filings` bucket in MinIO |
| `make ingest ticker=NVDA` | Trigger ingestion synchronously (dev convenience) |
| `make worker` | Start the ARQ worker process |

### Tests

| File | Coverage |
|---|---|
| `packages/ingestion/tests/test_xbrl_parser.py` | 14 tests — parse, filter, period types, edge cases |
| `packages/ingestion/tests/test_edgar_downloader.py` | 10 tests — CIK resolution, filings list, date parsing |
| `packages/ingestion/tests/test_pipeline.py` | 3 tests — skip-if-indexed, missing primary doc, XBRL+filings orchestration |

---

## How to Run Phase 1

```bash
# 1. Start infra (Postgres, MinIO, Redis)
make up-infra

# 2. Run migrations to create tables
make migrate

# 3. Bootstrap MinIO bucket
make init-minio

# 4. Ingest filings for a ticker (dev convenience script)
make ingest ticker=NVDA

# 5. Or start the ARQ worker and enqueue via API (Phase 6+)
make worker
```

---

## Decision Log

### 1. Why direct EDGAR API (httpx) instead of sec-edgar-downloader?

`sec-edgar-downloader` is synchronous and downloads to a fixed folder structure on disk — not async, not configurable enough for our pipeline. The EDGAR REST API at `data.sec.gov` is free, undocumented-but-stable, and gives us exactly what we need with full async control. Two endpoints cover everything:

- `GET /submissions/CIK{cik}.json` — company metadata + last 1000 filings
- `GET /api/xbrl/companyfacts/CIK{cik}.json` — all XBRL facts, ever, for that company
- `GET https://www.sec.gov/Archives/edgar/data/{cik}/{clean_accn}/{filename}` — actual document

The EDGAR user-agent requirement (`User-Agent: CompanyName/version email`) is handled via `Settings.edgar_user_agent`.

**Rate limiting:** `_RateLimiter` uses an async mutex + `time.monotonic()` to enforce ≤8 req/sec (under the 10 req/sec fair-access policy). A single `asyncio.Lock` means concurrent downloads automatically queue; no request ever fires in parallel unless the interval has elapsed.

**Interview answer:** "We call EDGAR's REST API directly via httpx with an async rate limiter, rather than using sec-edgar-downloader which is synchronous and doesn't give us control over where files land or how requests are throttled."

### 2. Why filter XBRL to ~25 concepts instead of storing everything?

The EDGAR companyfacts endpoint returns every tagged XBRL value ever reported by the company — for a large-cap with 20+ years of filings this is 50k–200k rows per company. At S&P 100 scale that's 5–20M rows, most of which are obscure accounting line items the SQL Generator would never be asked about.

`CONCEPTS_OF_INTEREST` covers the income statement, balance sheet, and cash flow statement at the level analysts actually use. It can be expanded by adding to the frozenset with no schema changes.

**Interview answer:** "We filter XBRL to ~25 curated concepts on ingest. The SQL Generator agent operates over this warehouse, and broad coverage isn't worth the storage cost — we can always re-ingest to add concepts. The EDGAR API returns the full history in one call anyway."

### 3. Why full-replace (delete+insert) for XBRL upsert instead of true upsert?

EDGAR's companyfacts endpoint is an authoritative snapshot: if you fetch it today, it reflects all amendments and restated values. A true upsert (ON CONFLICT DO UPDATE) would require a composite unique key across (ticker, taxonomy, concept, unit, period_start, period_end, form), which is fragile as any of those fields can be null.

Delete-then-insert on a per-ticker basis is atomic within a transaction, idempotent on re-run, and never leaves stale rows from superseded filings. The only downside is a brief gap in reads, which doesn't matter since XBRL facts are used batch-style by the SQL Generator, not real-time.

### 4. Why Docling → PyMuPDF → BeautifulSoup fallback hierarchy?

Most SEC 10-Ks filed after ~1996 are HTML (`.htm`), not PDF — EDGAR accepted HTML as the primary filing format. BeautifulSoup handles these natively and is the most reliable path. PDFs do appear (older filings, exhibits, ARS), and Docling is best-in-class for table extraction which matters for financial statements. PyMuPDF is the fallback because it's fast and handles edge cases (malformed PDFs, scanned documents with embedded text layers) where Docling sometimes raises.

**Interview answer:** "The extractor first checks the file extension. Most EDGAR filings since the mid-90s are HTML — we use BeautifulSoup with lxml. For PDFs, Docling gives us structured table extraction which matters for reading financial statement tables, and PyMuPDF is a 10ms fallback for anything Docling chokes on."

### 5. Why cap raw_text at 200k chars in Postgres?

The full text of a 10-K can be 500k–1M characters. Storing it in Postgres is useful for debugging and for Phase 3's single-agent RAG baseline, but we don't want to hit Postgres row-size limits or bloat the `filings` table. The full document is always retrievable from MinIO. Phase 2 (retrieval) will chunk the full text from MinIO into Qdrant — the Postgres copy is just a convenience.

### 6. Why asyncio.to_thread for Docling/PyMuPDF?

Docling and PyMuPDF are synchronous C-extension libraries. Calling them directly in an async function would block the event loop, preventing other filings from downloading concurrently. `asyncio.to_thread` runs them in a thread pool so the event loop stays free. This is the correct pattern for CPU/blocking I/O work in an async codebase — not subprocess, not ProcessPoolExecutor.

---

## Gotchas

**1. Alembic with `include_schemas=True` requires `version_table_schema`**
Without `version_table_schema="mia"`, Alembic puts its `alembic_version` table in the `public` schema by default. With `include_schemas=True`, autogenerate then sees the public-schema tables from other projects (e.g. Langfuse's `public.users`) and tries to drop them. Setting `version_table_schema="mia"` keeps Alembic's state in our schema and prevents cross-project interference.

**2. EDGAR submissions API returns integer CIK in URL path (not zero-padded)**
The URL `https://www.sec.gov/Archives/edgar/data/{cik}/{accn}/{doc}` uses the integer CIK (no leading zeros), while `data.sec.gov/submissions/CIK{cik}.json` uses the zero-padded 10-digit form. Mixing these up causes 404s. Fixed by casting to `int()` for archive URLs.

**3. `company_tickers.json` is the right CIK lookup (not EDGAR full-text search)**
The EDGAR EFTS search endpoint (used by the web UI) is rate-limited and unstable for programmatic use. The static `company_tickers.json` file is the authoritative, always-up-to-date mapping and resolves in one cached HTTP call.

**4. sec-edgar-downloader package is NOT the same as the edgar package**
Both exist on PyPI. `sec-edgar-downloader` (jadchaar) is modern (5.x). `edgar` (edgartools) is a different package entirely. The pyproject.toml dependency was updated to `sec-edgar-downloader>=5.0` but the actual downloader in Phase 1 calls EDGAR REST directly — `sec-edgar-downloader` is kept in dependencies for Phase 4 if needed.

---

## What Phase 2 Needs From Here

- `mia.filings` rows with `status="indexed"` and `raw_text` populated
- MinIO with filing documents stored under `filings/{ticker}/{form_type}/{accession}/{doc}`
- `xbrl.facts` populated for tickers of interest
- `make ingest ticker=NVDA` should complete cleanly before starting Phase 2

---

## Open Items (carry into Phase 2)

- [ ] Add `pytest.ini` fixture for `asyncio_mode = auto` if not already set
- [ ] S3 presigned URLs for MinIO (useful for frontend citation viewer in Phase 7)
- [ ] Handle filing amendments (forms "10-K/A", "10-Q/A") — currently filtered out
- [ ] Older filings in SGML format — currently skipped; add parser if needed for historical data
- [ ] Add `ingestion_jobs` table (Phase 6) for tracking ARQ job status in the API
