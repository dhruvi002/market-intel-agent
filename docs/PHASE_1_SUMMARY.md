# Phase 1 — Design Decisions & Interview Prep

> **Use this doc to:** challenge design decisions, prep interview answers, and restore context in a new Claude session.
> **Phase:** 1 — Ingestion Pipeline (EDGAR downloader, XBRL→Postgres ETL, PDF/HTML extractor, MinIO storage)
> **Status:** ✅ Complete — live run confirmed against SEC EDGAR

---

## What Phase 1 Built

Every file with its purpose — know this cold, you'll be asked "walk me through what you built."

| File | What it does |
|---|---|
| `packages/ingestion/src/mia_ingestion/models.py` | SQLAlchemy 2.0 ORM: `Filing` (one row per SEC filing) and `XBRLFact` (one row per structured financial data point) |
| `packages/ingestion/src/mia_ingestion/db.py` | Async engine + `async_sessionmaker` + `get_db_session()` context manager (commit on exit, rollback on error) |
| `packages/ingestion/src/mia_ingestion/edgar/downloader.py` | Async EDGAR REST client: ticker→CIK resolution, recent filings list, document download, XBRL companyfacts fetch. Rate-limited to 8 req/sec, retried via tenacity. |
| `packages/ingestion/src/mia_ingestion/edgar/xbrl_parser.py` | Converts raw EDGAR companyfacts JSON into typed `XBRLFact` objects. Filters to ~25 curated financial concepts. |
| `packages/ingestion/src/mia_ingestion/pdf/extractor.py` | Three-tier extractor: `.htm`/`.html` → BeautifulSoup; `.pdf` → Docling (IBM open-source, tables as Markdown) → PyMuPDF fallback. Heuristic section detection for MD&A, Risk Factors, Financials. |
| `packages/ingestion/src/mia_ingestion/storage/minio_client.py` | boto3 wrapper for local MinIO: `ensure_bucket()`, structured key paths (`filings/NVDA/10-K/accession/doc.htm`), async-safe via `asyncio.to_thread`. |
| `packages/ingestion/src/mia_ingestion/pipeline.py` | `IngestionPipeline` orchestrator: skip-if-indexed check, download→MinIO→extract→persist flow, XBRL full-replace logic. |
| `packages/ingestion/src/mia_ingestion/__init__.py` | Lazy `__getattr__` import — defers heavy deps until first use. |
| `apps/api/alembic/env.py` | Async-aware Alembic env: `asyncio.run(run_migrations_online())`, `include_schemas=True`, `version_table_schema="mia"`. |
| `apps/api/alembic/versions/20260611_001_init_ingestion.py` | Creates `mia.filings` and `xbrl.facts` with all constraints and indexes. |
| `apps/worker/src/mia_worker/tasks/ingest.py` | ARQ task wrappers: `ingest_ticker` and `ingest_filing` — thin ARQ glue around `IngestionPipeline`. |
| `apps/worker/src/mia_worker/main.py` | `WorkerSettings` class: functions list, startup/shutdown hooks, `_redis_settings()` URL parser, `max_jobs=10`, `job_timeout=1800`. |
| `scripts/init_minio.py` | One-time MinIO bootstrap: creates `sec-filings` bucket. |
| `scripts/ingest_ticker.py` | CLI entry point: `python scripts/ingest_ticker.py NVDA AAPL --forms 10-K 10-Q` |

**Tests:** 29 unit tests across 3 files (14 XBRL parser, 12 EDGAR downloader, 3 pipeline). All pure-Python; no DB or network calls.

**Live run numbers (NVDA, 2026-06-11):**
- 4,937 XBRL facts inserted into `xbrl.facts`
- 21 filings indexed: 3 × 10-K, 8 × 10-Q, 10 × 8-K
- Extraction backend: `beautifulsoup` for all (EDGAR files are `.htm`)
- Duration: ~90 seconds end-to-end

---

## How to Run

```bash
# Start infra
make up-infra

# Apply DB migrations (creates mia.filings + xbrl.facts)
make migrate

# Bootstrap MinIO bucket
make init-minio

# Ingest a ticker (downloads + extracts + persists)
make ingest ticker=NVDA

# Run tests
make test

# Start the ARQ worker (for background job processing)
make worker
```

---

## Decision Log — The "Why" Behind Every Choice

### 1. Why direct EDGAR REST API (httpx) instead of sec-edgar-downloader?

**Short answer:** `sec-edgar-downloader` is synchronous, has opinionated download paths, and lacks the XBRL companyfacts endpoint. Direct httpx gives us async, full control, and access to both APIs we need.

**Deeper:** The project needs two things from EDGAR: (1) filing document download (primary `.htm` files) and (2) structured XBRL facts from the companyfacts API (`data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json`). `sec-edgar-downloader` only wraps filing downloads and does so synchronously. Wrapping a synchronous library in `asyncio.to_thread` works but means a separate thread per download — with 21+ filings per ticker, that's 21 threads. Writing a thin async client over two well-documented REST endpoints is less than 200 lines and gives us rate limiting, retry logic, and full control over headers (required by EDGAR's User-Agent policy).

**Interview follow-up — "Why does EDGAR require a User-Agent header?"**
SEC EDGAR explicitly requires a `User-Agent` string identifying the caller (your name/email). Requests without it get blocked with a 403. This is documented in their developer FAQ. Our `Settings.edgar_user_agent` config value is set in `.env` and sent with every request.

---

### 2. Why filter XBRL to ~25 concepts instead of storing all of them?

**Short answer:** The full companyfacts JSON for a company like NVDA contains ~15,000+ entries across ~700 concepts. 99% are obscure accounting footnote tags we'd never query. Filtering to 25 curated concepts covers all income statement, balance sheet, and cash flow fundamentals without adding noise to the SQL agent's search space.

**Deeper:** The `CONCEPTS_OF_INTEREST` frozenset in `xbrl_parser.py` includes: `Revenues`, `NetIncomeLoss`, `GrossProfit`, `OperatingIncomeLoss`, `EarningsPerShareBasic/Diluted`, `Assets`, `Liabilities`, `StockholdersEquity`, `CashAndCashEquivalentsAtCarryingValue`, `LongTermDebt`, `NetCashProvidedByUsedInOperatingActivities`, and ~15 others. These map directly to the financial metrics an analyst would ask about ("what was NVDA's revenue growth?" "what's their cash position?"). Storing 15k rows per company would bloat the `xbrl.facts` table and force the SQL agent to filter out noise on every query.

**Interview follow-up — "What if a user asks for a concept you didn't include?"**
Good question — it's a known gap. The mitigation is that Phase 2 stores the full filing text in MinIO; the RAG pipeline can retrieve MD&A sections that discuss any metric. The XBRL table is for fast structured queries on the ~25 most common KPIs; the vector store handles the long tail.

---

### 3. Why delete-then-insert (not true upsert) for XBRL facts?

**Short answer:** The companyfacts API returns an authoritative, complete history for a company. There's no meaningful "update" case — the canonical state is "whatever the API returned just now." Delete-then-insert per ticker is simple, atomic, and correct.

**Deeper:** A proper upsert would require a composite unique key across `(cik, taxonomy, concept, unit, period_end, period_start, accession)` — most of those columns are nullable, which makes ON CONFLICT unwieldy. Worse, EDGAR periodically amends historical filings, so an old row's `value` might legitimately change. With delete-then-insert, we always end up with exactly what EDGAR's API reports, with no stale rows. The entire operation runs inside a single transaction: we delete all existing rows for the ticker, then bulk-insert the new rows; if the insert fails, the delete is rolled back — no data loss.

**Interview follow-up — "Isn't this slow? You're deleting 4,937 rows and re-inserting them."**
For nightly refresh cadence, yes this is fine. Delete+insert on 5k rows in Postgres takes well under a second. If we were refreshing intraday at high frequency we'd add a `last_seen_at` timestamp and implement a real upsert. For monthly-updated 10-K/10-Q data, this is the right call.

---

### 4. Why Docling → PyMuPDF → BeautifulSoup, in that order?

**Short answer:** Most EDGAR filings are HTML, so BeautifulSoup is the common path. For the minority that are PDF, Docling is first because it extracts tables as Markdown (preserving financial statement structure); PyMuPDF is the fallback because it's faster and has no heavy dependencies.

**Deeper hierarchy:**
| Backend | Handles | Quality | Speed | Tables |
|---|---|---|---|---|
| BeautifulSoup (lxml) | `.htm`/`.html` | High — respects DOM structure | Fast | ✅ extracts `<table>` elements |
| Docling | `.pdf` | Highest — ML-based layout detection | Slow (~5-10s/page) | ✅ returns Markdown tables |
| PyMuPDF | `.pdf` (fallback) | Medium — heuristic text blocks | Fast | ❌ plain text only |

The extractor dispatches on file extension, not on content sniffing. This is intentional — EDGAR's submissions API tells us the primary document filename, so we know at download time whether it's `.htm` or `.pdf`. In practice, all post-1996 annual reports are HTML, and the `.pdf` path exists primarily for older filings or any future edge cases.

**Interview follow-up — "Why not always use Docling for HTML too?"**
Docling is primarily a PDF layout engine. For HTML it essentially serializes the DOM, which BeautifulSoup does better and 10× faster. Docling's value is its ML-based table detection in scanned PDFs — not needed when we have the clean HTML source.

---

### 5. Why cap `raw_text` at 200,000 characters in Postgres?

**Short answer:** A 10-K can be 300–500 pages of dense text — up to 2M+ characters. Storing that verbatim in Postgres is expensive on disk and slow to read. The full text is always available in MinIO; Postgres holds a capped extract for lightweight queries.

**Deeper:** The `Filing.raw_text` column is used for Phase 2 chunking: we'll read it, split into 512-token overlapping chunks, embed each chunk, and index in Qdrant. At 200k chars, a typical 10-K's MD&A + Risk Factors section fits entirely within the cap, which is the content that matters most for retrieval. Financials tables (mostly numbers, low semantic density) are the first thing truncated. The `_TEXT_CHAR_CAP = 200_000` constant is in `pipeline.py`; changing it to 0 would disable the cap and store the full text. The full original file is always retrievable from MinIO via `filing.minio_path`.

**Interview follow-up — "How do you know you're not truncating important content?"**
The section extractor in `extractor.py` runs before the cap is applied — it identifies MD&A, Risk Factors, and Financials sections by EDGAR Item headers. Those sections are stored separately in a `sections` dict. The 200k cap is applied to the flat full text before it hits Postgres; the section dict is not stored in the DB (Phase 2 will handle that separately).

---

### 6. Why `asyncio.to_thread` for Docling, PyMuPDF, and boto3?

**Short answer:** All three are synchronous and CPU/I/O-bound. Running them directly in an async function would block the event loop for the entire duration of each call, preventing other coroutines from running.

**Deeper:** Python's `asyncio` event loop is single-threaded. Calling a blocking function (like `fitz.open()` or `boto3.upload_file()`) directly inside a coroutine stalls the entire event loop — no other coroutines can progress until that call returns. `asyncio.to_thread` runs the function in a ThreadPoolExecutor thread, allowing the event loop to continue scheduling other coroutines while the blocking work happens in the background. The function returns an awaitable, so the calling coroutine suspends cleanly.

This is the standard pattern for wrapping legacy sync libraries in async code. The alternative — rewriting Docling/PyMuPDF/boto3 to be async — isn't viable. A second alternative — using `loop.run_in_executor` with `ProcessPoolExecutor` for CPU-bound work — is better for truly CPU-heavy tasks (Docling's ML inference) but adds pickling overhead and complicates error propagation. For a capstone where we're processing dozens of files, not thousands, `to_thread` with the default `ThreadPoolExecutor` is the right balance.

---

### 7. Why `trust_env=False` on `httpx.AsyncClient`?

**Short answer:** Prevents the EDGAR client from picking up system proxy environment variables (`HTTP_PROXY`, `SOCKS5_PROXY`, etc.) that could route EDGAR traffic through an unexpected or broken proxy.

**Deeper:** `httpx.AsyncClient()` defaults to `trust_env=True`, which means it inherits proxy settings from the environment. In development environments (CI containers, WSL2, certain VPNs), there's often a `SOCKS5_PROXY` or `ALL_PROXY` environment variable set system-wide. When httpx sees a `socks5://` proxy URL but `socksio` isn't installed, it raises `ImportError: Using SOCKS proxy, but the 'socksio' package is not installed`. This was the actual error hit during development.

The fix is `trust_env=False`, which is also the right architectural decision: EDGAR calls are public internet requests that should go direct. A proxy in front of EDGAR adds latency and the proxy operator could log/modify the traffic. `trust_env=False` makes EDGAR calls explicit and predictable regardless of the environment.

---

### 8. Why lazy imports in `__init__.py` using `__getattr__`?

**Short answer:** Prevents heavy package-level deps (boto3, Docling, PyMuPDF) from loading at test collection time, when most tests only need `XBRLParser` or `EDGARDownloader`.

**Deeper:** When Python imports `mia_ingestion`, it executes `__init__.py`. If that file does `from mia_ingestion.pipeline import IngestionPipeline`, it triggers the entire import chain: `pipeline.py` → `minio_client.py` → `import boto3`. If boto3 isn't installed (e.g., in a minimal test environment), this fails immediately — even when the test being collected only tests the XBRL parser with no DB or MinIO involvement.

The `__getattr__` pattern defers the import until the attribute is first accessed:
```python
def __getattr__(name: str):
    if name == "IngestionPipeline":
        from mia_ingestion.pipeline import IngestionPipeline
        return IngestionPipeline
    raise AttributeError(...)
```
This was the actual error hit during development — pytest collected `test_xbrl_parser.py`, triggered package `__init__.py`, which triggered `boto3` import, which failed. The lazy import fixed it. The alternative (just not importing in `__init__.py` at all) works too but breaks `from mia_ingestion import IngestionPipeline`.

**Interview follow-up — "What's the trade-off?"**
The module appears to have `IngestionPipeline` as an attribute based on `__all__`, but accessing it for the first time is slightly slower (one extra `__getattr__` call). For a class that's only instantiated once per pipeline run, this is immeasurable.

---

### 9. Why `asyncio.Lock` + `time.monotonic()` for rate limiting instead of a library?

**Short answer:** The logic is 12 lines and has exactly one job. Adding a rate-limiting library dependency for 12 lines of well-understood token-bucket logic is not the right trade-off.

**Deeper:** The `_RateLimiter` class implements a simple "minimum interval" rate limiter:
1. Acquire the lock (serializes concurrent callers).
2. Compute how much time has elapsed since the last call.
3. If the elapsed time is less than `1/rate`, sleep the remainder.
4. Record `time.monotonic()` as the new last-call time.
5. Release the lock.

`time.monotonic()` is used instead of `time.time()` because it's guaranteed to be monotonically increasing — wall clock can jump backward (NTP adjustments, DST). `asyncio.Lock` ensures that when two coroutines try to make EDGAR calls simultaneously, they queue through the rate limiter rather than both firing at once. At 8 req/sec (under EDGAR's 10 req/sec limit), the 20% buffer absorbs jitter.

**Interview follow-up — "How does this behave with 10 concurrent coroutines?"**
Each coroutine calls `acquire()`, which grabs the lock (the others wait), sleeps 125ms if needed, records the timestamp, then releases. The next coroutine then runs through the same logic. Net effect: requests flow at ≤8/sec regardless of how many coroutines are waiting. The worst-case latency for the 10th coroutine is 10 × 125ms = 1.25 seconds of waiting, which is acceptable for a background ingestion job.

---

### 10. Why `include_schemas=True` + `version_table_schema="mia"` in Alembic?

**Short answer:** The project uses two Postgres schemas (`mia` and `xbrl`). Without `include_schemas=True`, Alembic only looks at the `public` schema and misses our tables. `version_table_schema="mia"` keeps the migration version table in the `mia` schema rather than `public`, preventing collision with other projects on the same Postgres instance.

**Deeper:** Postgres schemas are namespaces within a database. We created `mia` (for application tables) and `xbrl` (for structured financial data) in `00_init.sql`. Alembic's default `autogenerate` only inspects the `public` schema, which means it would never see `mia.filings` or `xbrl.facts`. Setting `include_schemas=True` in `env.py` tells Alembic to introspect all schemas visible to the connection. The `version_table_schema="mia"` option controls where Alembic writes its `alembic_version` tracking table — we put it in `mia` rather than `public` to avoid stomping on another project's version table if they share the same Postgres instance.

**Interview follow-up — "Why two schemas? Why not just prefix the table names?"**
Schema separation is cleaner than table name prefixes. `xbrl.facts` is more readable than `public.xbrl_facts`, and schema-level permissions let you grant a read-only role access to `xbrl.*` only — useful for the SQL agent (Phase 4), which should have `SELECT` on `xbrl.facts` but not `mia.filings`.

---

### 11. Why SQLAlchemy 2.0 `Mapped[]` / `mapped_column` instead of the 1.x Column() style?

**Short answer:** `Mapped[Optional[str]]` is type-annotated — mypy can infer column nullability from the Python type, removing an entire class of bugs where a nullable DB column was typed as non-optional in Python.

**Deeper:** In SQLAlchemy 1.x, `Column(String, nullable=True)` gives you no type inference — the attribute's type at runtime is opaque to mypy. In 2.0's `mapped_column`, writing `name: Mapped[Optional[str]] = mapped_column(String(64))` tells both SQLAlchemy (nullable in DB) and mypy (Optional in Python) the same thing from one declaration. When you access `filing.raw_text` in typed code, mypy knows it might be `None` and will complain if you use it without a null check. This catches real bugs.

---

### 12. Why MinIO instead of storing files on the local filesystem?

**Short answer:** MinIO is S3-compatible, so the production code path (AWS S3) requires changing only one environment variable (`MINIO_ENDPOINT` → `s3.amazonaws.com`). Local filesystem storage would need a completely different code path in production.

**Deeper:** Phase 9 includes an optional cloud deploy to HuggingFace Spaces + Fly.io. Those environments don't have persistent local filesystems. If we stored filings on disk during development, we'd have to rewrite the storage layer at deploy time. MinIO's boto3 interface is identical to AWS S3 — same `put_object`, `get_object`, `head_bucket` calls, same `signature_version="s3v4"` auth, same presigned URL patterns. The only difference is the endpoint URL. Writing against MinIO now means writing against S3 for free.

---

## Gotchas from Real Execution

These came up during the actual `make ingest ticker=NVDA` run — knowing them signals real hands-on experience.

**1. `XMLParsedAsHTMLWarning` on XBRL `.htm` files**
EDGAR's inline XBRL files have an `.htm` extension but are actually XML-flavored HTML (XHTML). BeautifulSoup's `lxml` parser raises `XMLParsedAsHTMLWarning` when it encounters these. The warning is harmless — lxml's HTML5 mode handles the content correctly. It's suppressed by parsing intent (we want the text content, not the XBRL tags), and the extracted text is accurate.

**2. Integer CIK in archive URL vs. zero-padded CIK everywhere else**
EDGAR's submissions API returns CIKs as zero-padded 10-digit strings: `"0001045810"`. But the filing archive URL uses the integer representation: `https://www.sec.gov/Archives/edgar/data/1045810/...`. If you pass the zero-padded string, you get a 404. Fix: `cik_int = int(cik)` before constructing the archive URL. This is not documented prominently; it was discovered by inspecting actual EDGAR archive URLs in a browser.

**3. Eager boto3 import killed test collection**
Before the lazy import fix, importing `mia_ingestion` (triggered by pytest's test collection of any file in the package) immediately imported `pipeline.py` → `minio_client.py` → `import boto3`. In a minimal test environment without boto3 installed, this caused `ModuleNotFoundError` before a single test ran. The fix was lazy `__getattr__` imports in `__init__.py` — `boto3` only loads when `IngestionPipeline` is first accessed, not at import time.

**4. SOCKS proxy in `httpx.AsyncClient`**
The development sandbox had a `SOCKS5_PROXY` environment variable set. `httpx.AsyncClient()` defaults to `trust_env=True`, which picked this up and tried to route EDGAR requests through the SOCKS proxy. Without `socksio` installed, this raised `ImportError: Using SOCKS proxy, but the 'socksio' package is not installed`. The symptom looked like an httpx configuration problem, not a proxy problem. Fix: `trust_env=False`. Lesson: always set `trust_env=False` for HTTP clients making calls to known public APIs — you don't want environment variables silently rerouting your traffic.

**5. `pytest --cov` requires `pytest-cov`**
`pyproject.toml` had `addopts = "--cov=packages --cov-report=..."` which runs by default on every `pytest` invocation. In the sandbox, `pytest-cov` wasn't installed, causing `error: unrecognized arguments: --cov`. Workaround: `pytest --override-ini="addopts="`. Long-term fix: add `pytest-cov` to dev dependencies.

---

## How to Use This Doc for Interview Prep

**Pattern for answering design questions (same as Phase 0):**
1. One-sentence answer — the choice you made
2. Two sentences — the problem it solves, with a concrete example
3. One sentence — the trade-off you accepted

**Example — "Why filter XBRL to 25 concepts?"**
> "We filter the EDGAR companyfacts response to ~25 curated financial concepts because a company like NVDA has 15,000+ XBRL entries across ~700 concepts, and the long tail is accounting footnote tags that no analyst query would reference. The 25 concepts cover every income statement, balance sheet, and cash flow KPI needed for the SQL agent. The trade-off is that ad-hoc queries on non-standard concepts fall back to full-text RAG retrieval instead of the structured SQL path."

**Example — "Walk me through the ingestion pipeline."**
> "When `ingest_ticker` is called with NVDA: first we resolve the ticker to a CIK via EDGAR's company_tickers.json. Then we make one call to the XBRL companyfacts endpoint — that single JSON contains all historical structured financial facts. We filter to ~25 concepts and bulk-insert into `xbrl.facts`. In parallel, we fetch filing metadata for the most recent 3 × 10-K, 8 × 10-Q, and 10 × 8-K. For each filing, we skip if already indexed, download the primary document, upload to MinIO, extract text with BeautifulSoup or Docling depending on file type, cap at 200k chars, and persist a `Filing` row with a status state machine (pending→downloading→storing→extracting→indexed). All HTTP calls are rate-limited to 8 req/sec and retried with exponential backoff."

**Red flags to avoid:**
- Don't say "I used BeautifulSoup because it's popular" — say it handles EDGAR's HTML DOM structure cleanly and is 10× faster than Docling for HTML input
- Don't say "MinIO is like S3" without knowing WHY that matters (production path requires only an env var change)
- Don't confuse zero-padded CIK (submissions API) with integer CIK (archive URL) — interviewers who know EDGAR will catch this

**Questions to be ready for:**
- "What's in the `xbrl.facts` table?" (one row per structured financial data point: ticker, CIK, taxonomy, concept, label, value, unit, period_type, period_start, period_end, form, frame)
- "Why does the status column have 6 states?" (pending → downloading → storing → extracting → indexed → error — each state is a checkpoint; if the pipeline crashes mid-run, you can resume from the last successful state rather than re-downloading)
- "How does the pipeline handle re-ingesting a filing that's already in the DB?" (skip-if-indexed: `_ingest_one_filing` checks whether `accession_number` exists with `status='indexed'` before downloading; idempotent by design)
- "How do you handle EDGAR's rate limit?" (`_RateLimiter` with `asyncio.Lock` + `time.monotonic()` at 8 req/sec; EDGAR's limit is 10 req/sec, giving 20% safety margin)
- "What happens if Docling fails on a PDF?" (silent fallback to PyMuPDF; the `extraction_backend` field on `ExtractedDocument` records which was used so you can audit quality later)
- "How do you know the XBRL data is correct?" (it's the canonical data from EDGAR's own companyfacts endpoint — if NVDA's 10-K is wrong there, it's wrong everywhere; we don't validate the values, we trust the source)
- "Why ARQ tasks instead of calling the pipeline directly from the API?" (decoupling: ingestion is slow, up to 90 seconds per ticker; if the API blocked on it, the HTTP timeout would fire and the client would see a failure even when the work completed successfully; ARQ lets the API return a job_id immediately and let the worker run asynchronously)
