# Session Context — Autonomous Enterprise Market Intelligence Agent
> **Purpose:** Drop this file into a new Claude session to restore full project context instantly.
> **Last updated:** 2026-06-11
> **Session recap:** Phase 3 complete. Single-agent RAG baseline built: LLM factory (Gemini → Groq → Cerebras fallback chain), RAG prompt template with inline [N] citation instructions, RAGAgent (retrieve → format context → LLM → parse citations → RAGResponse), CLI query script (make query/query-bm25/query-dense/query-groq). 43/43 tests green. Phase 4 (LangGraph multi-agent skeleton) is next.

---

## 1. Who You're Working With

- **Name:** Dhruvi Shah
- **Level:** Master's in Computer Science student
- **GitHub:** `dhruvi002`
- **Email:** worksofdhruvi@gmail.com
- **Background:** Comfortable with ML/RL concepts (has a separate SAC-based portfolio optimization project). Communicate in technical terms — no need to explain basics.
- **Hard constraint:** $0 budget. No credit card anywhere. Every tool, API, and service must be free without requiring payment info.
- **Commit style:** Never add "Co-Authored-By: Claude" or any Claude attribution to git commits.

---

## 2. What This Project Is

An **Autonomous Enterprise Market Intelligence Agent** — a multi-agent system that, given a natural-language question about a public company or sector, autonomously:

1. Plans a research strategy (Supervisor decomposes into subtasks)
2. Gathers evidence from SEC EDGAR filings, financial data, and open-web sources
3. Synthesizes a cited, citation-grounded report with the ability to stream agent reasoning live
4. Verifies claims against retrieved evidence via a Critic Agent that triggers self-correction loops on hallucination

**Domain:** Public-company financial intelligence (SEC EDGAR + market data). S&P 100 universe.

**Scope:** Capstone-grade, 2–3 months part-time. Not a weekend MVP — full ablation studies, observability, eval harness, writeup.

**Resume bullet (what this expands on):**
> Orchestrated a Full Stack multi-agent system to automate competitive intelligence gathering and document synthesis for enterprise consulting use cases. Designed a "Supervisor-Worker" agentic architecture using LangGraph to manage 4 specialized agents (Web Search, Financial Parser, Summarizer, SQL Generator) with self-correction loops. Built a real-time React.js dashboard with WebSocket integration to stream agent thought-processes and final insights, supporting concurrent analysis of 50+ industry reports. Implemented an advanced RAG pipeline with hybrid search and cross-encoder re-ranking, achieving a 28% improvement in retrieval precision for unstructured PDF disclosures.

---

## 3. Key Decisions Made (Do Not Re-litigate)

| Decision | Choice | Reason |
|---|---|---|
| Domain | SEC EDGAR / financial filings | Free, no auth, strong narrative continuity with user's FinRL background, "Financial Parser" agent maps cleanly |
| LLM strategy | Free cloud APIs: Gemini 2.0 Flash + Groq Llama 3.3 70B + Cerebras fallback | No CC required; Gemini free tier is generous (1M context, 1500 RPD); Groq is fast (800 tok/s) |
| Deployment | Local Docker Compose primary, optional cloud deploy later (HF Spaces + Vercel + Fly.io) | Free everywhere; local-first means no hosting anxiety during build |
| Scope | Capstone-scale (~10 weeks part-time) | User wants a defensible portfolio centerpiece, not a quick MVP |
| Vector DB | Qdrant (self-hosted) instead of Pinecone | No quota, no CC, native hybrid search, better filtering |
| Interview prep storage | Separate private GitHub repo | Public code repo stays clean; no "memorized script" vibe for recruiters |
| Langfuse version | v2 (not v3) | v3 mandates ClickHouse + S3 event storage — too heavy for local dev. v2 only needs Postgres + Redis which we already have. |
| Python package manager | uv workspaces | Fast, lockfile-based, workspace-aware installs per package — keeps Docker images lean |
| pnpm for frontend | pnpm workspaces | Faster than npm, workspace support, stricter about phantom dependencies |

---

## 4. Repositories

### Public Code Repo
- **GitHub:** https://github.com/dhruvi002/market-intel-agent (currently private — flip public once Phase 1 has substantive code)
- **Local:** `/Users/dhruvishah/Documents/Projects/MarketIntelAgent/`
- **Remote:** `https://github.com/dhruvi002/market-intel-agent.git`
- **Branch:** `main`

### Private Interview Prep Repo
- **GitHub:** https://github.com/dhruvi002/prep-market-intel-agent (always private)
- **Local:** `/Users/dhruvishah/Documents/Projects/PrepMarketIntelAgent/`
- **Remote:** `https://github.com/dhruvi002/prep-market-intel-agent.git`
- **Branch:** `main`

---

## 5. Local Folder Structure (as built)

```
/Users/dhruvishah/Documents/Projects/MarketIntelAgent/
├── apps/
│   ├── api/                        # FastAPI gateway (async, WebSocket) — Phase 6
│   │   ├── Dockerfile
│   │   ├── alembic.ini             # Alembic config
│   │   ├── alembic/                # Migrations
│   │   │   ├── env.py              # Async-aware, imports all ORM Bases
│   │   │   ├── script.py.mako
│   │   │   └── versions/
│   │   │       └── 20260611_001_init_ingestion.py
│   │   ├── pyproject.toml          # mia-api package
│   │   └── src/mia_api/__init__.py
│   ├── worker/                     # ARQ async task workers
│   │   ├── Dockerfile
│   │   ├── pyproject.toml          # mia-worker package
│   │   └── src/mia_worker/
│   │       ├── __init__.py
│   │       ├── main.py             # WorkerSettings, startup/shutdown hooks
│   │       └── tasks/
│   │           └── ingest.py       # ingest_ticker + ingest_filing ARQ tasks
│   └── web/                        # React 18 + TypeScript + Vite — Phase 7
│       ├── Dockerfile              # dev + build + prod (nginx) stages
│       ├── nginx.conf
│       ├── package.json
│       ├── vite.config.ts
│       ├── tailwind.config.ts
│       ├── tsconfig.json
│       └── src/
│           ├── main.tsx
│           ├── App.tsx             # 4-panel layout: query | DAG | draft | evidence
│           ├── index.css           # CSS vars for agent node states
│           ├── types/agent.ts      # TypeScript mirror of schemas.py
│           ├── store/sessionStore.ts  # Zustand store, handles all WS events
│           └── components/
│               ├── QueryPanel.tsx
│               ├── AgentDAG.tsx    # React Flow DAG, nodes light up live
│               ├── DraftViewer.tsx # Streaming markdown + Critic verdict badge
│               └── EvidencePanel.tsx
├── packages/
│   ├── shared/                     # mia-shared — Pydantic schemas + Settings
│   │   └── src/mia_shared/
│   │       ├── __init__.py
│   │       ├── config.py           # pydantic-settings Settings + get_settings()
│   │       └── schemas.py          # AgentState, Evidence, Citation, CritiqueResult,
│   │                               #   AgentEvent, QueryRequest, all enums
│   ├── agents/                     # mia-agents — single-agent RAG baseline ✅ Phase 3; LangGraph Phase 4
│   │   └── src/mia_agents/
│   │       ├── __init__.py         # lazy __getattr__ imports
│   │       ├── llm.py              # LLMProvider enum + get_llm() fallback chain (Gemini → Groq → Cerebras)
│   │       ├── prompts.py          # RAG_PROMPT ChatPromptTemplate + build_rag_messages()
│   │       └── rag_agent.py        # RAGResponse model + RAGAgent.run() (retrieve → LLM → citations)
│   ├── retrieval/                  # mia-retrieval — hybrid search + reranker ✅ Phase 2
│   │   └── src/mia_retrieval/
│   │       ├── __init__.py         # lazy __getattr__ imports (same pattern as mia_ingestion)
│   │       ├── chunker.py          # Chunk dataclass + word-count chunker (300w/40w overlap, UUID5 IDs)
│   │       ├── embedder.py         # Embedder singleton: bge-large-en-v1.5, lazy load, L2-norm
│   │       ├── bm25_index.py       # BM25Index: build/add/search/save/load (rank_bm25 + pickle)
│   │       ├── qdrant_store.py     # QdrantStore: async client, ensure_collection, upsert, search
│   │       ├── hybrid.py           # reciprocal_rank_fusion(): RRF k=60 over BM25 + dense
│   │       ├── reranker.py         # Reranker singleton: bge-reranker-v2-m3, lazy load
│   │       ├── indexer.py          # FilingRecord + IndexStats + IndexingPipeline
│   │       └── retriever.py        # RetrieveMode + Retriever.retrieve() → list[Evidence]
│   ├── ingestion/                  # mia-ingestion — EDGAR + XBRL + PDF
│   │   └── src/mia_ingestion/
│   │       ├── models.py           # Filing + XBRLFact ORM models
│   │       ├── db.py               # Async engine + get_db_session()
│   │       ├── pipeline.py         # IngestionPipeline — main entry point
│   │       ├── edgar/
│   │       │   ├── downloader.py   # Async EDGAR client, rate limiter
│   │       │   └── xbrl_parser.py  # companyfacts JSON → XBRLFact list
│   │       ├── pdf/
│   │       │   └── extractor.py    # Docling / PyMuPDF / BeautifulSoup
│   │       └── storage/
│   │           └── minio_client.py # boto3 S3 wrapper for MinIO
│   └── eval/                       # mia-eval — RAGAS harness (Phase 8)
├── infra/
│   ├── docker-compose.yml          # 6 running services (see below)
│   └── init/postgres/00_init.sql   # Enables vector, pg_trgm, btree_gin; creates schemas
├── data/                           # gitignored — raw filings, parquet, model weights
├── notebooks/                      # Exploration, ablation plots
├── scripts/
│   ├── init_minio.py               # Bootstrap MinIO sec-filings bucket (run once)
│   ├── ingest_ticker.py            # CLI: make ingest ticker=NVDA (Phase 1 — EDGAR → Postgres + MinIO)
│   ├── index_ticker.py             # CLI: make index ticker=NVDA (Phase 2 — Qdrant + BM25)
│   ├── retrieve.py                 # CLI: make retrieve query="..." (Phase 2 — test retrieval)
│   └── query.py                    # CLI: make query q="..." (Phase 3 — full RAG answer)
├── docs/
│   ├── PLAN.md                     # Full architecture plan
│   ├── CONTEXT.md                  # This file
│   ├── PHASE_0.md                  # Phase 0 build doc
│   ├── PHASE_0_SUMMARY.md          # Phase 0 decisions & interview prep
│   ├── PHASE_1.md                  # Phase 1 build doc
│   ├── PHASE_1_SUMMARY.md          # Phase 1 decisions & interview prep
│   ├── PHASE_2.md                  # Phase 2 build doc
│   ├── PHASE_2_SUMMARY.md          # Phase 2 decisions & interview prep ← standalone doc
│   ├── PHASE_3.md                  # Phase 3 build doc
│   ├── PHASE_3_SUMMARY.md          # Phase 3 decisions & interview prep ← standalone doc
│   ├── EVAL.md                     # Eval results (populated Phase 8)
│   └── WRITEUP.md                  # Final reflection (populated Phase 9)
├── .github/workflows/ci.yml        # Python lint → pytest → web build → compose validate
├── .pre-commit-config.yaml         # ruff, mypy, eslint, docker-compose-check
├── .env.example                    # All env vars documented with signup links
├── .gitignore
├── pyproject.toml                  # uv workspace root + ruff/mypy/pytest config
├── pnpm-workspace.yaml
├── package.json                    # root pnpm scripts
└── Makefile                        # make up/down/logs/migrate/lint/test/etc.
```

---

## 6. Running Services (Phase 0 complete)

| Service | Image | Host Port | Status |
|---|---|---|---|
| `postgres` | `pgvector/pgvector:pg16` | 5432 | ✅ healthy |
| `redis` | `redis:7-alpine` | 6379 | ✅ healthy |
| `qdrant` | `qdrant/qdrant:latest` | **6335** (6333 taken by medcomply project) | ✅ healthy |
| `minio` | `minio/minio:latest` | 9000 / 9001 (console) | ✅ healthy |
| `langfuse-db` | `postgres:16-alpine` | internal only | ✅ healthy |
| `langfuse-web` | `langfuse/langfuse:2` | 3000 | ✅ healthy |

**Note:** Qdrant is on port 6335 (not default 6333) because another project (`medcomply-qdrant`) owns 6333/6334 on this machine. This is set in `.env` as `QDRANT_PORT=6335`, `QDRANT_GRPC_PORT=6336`. Services communicate internally over Docker network `mia_network` on port 6333 — only the host binding differs.

**Langfuse v2** (not v3) — v3 requires ClickHouse which is too heavy for local dev. v2 only needs Postgres + Redis.

---

## 7. Full Tech Stack

### LLM Inference (all free, no CC)
| Provider | Model | Free Quota | Role |
|---|---|---|---|
| Google AI Studio | Gemini 2.0 Flash | 15 RPM, 1M TPM, 1500 RPD, 1M-token context | Long-context reasoning, structured output, final synthesis |
| Groq | Llama 3.3 70B, Llama 3.1 8B | ~30 RPM, generous TPD | Fast agent steps (800 tok/s) |
| Cerebras | Llama 3.3 70B | ~30 RPM | Fallback if Groq throttles (~2000 tok/s) |
| OpenRouter | Various free models | Variable | Final fallback |

### Embeddings & Reranking (local, free)
- `BAAI/bge-large-en-v1.5` — dense embeddings (1024-dim) via `sentence-transformers`
- `BAAI/bge-reranker-v2-m3` — cross-encoder reranker (matches/beats Cohere Rerank-3, no API cost)
- `rank_bm25` — sparse retrieval for hybrid search

### Data Stores (self-hosted in Docker, free)
- **Qdrant** — vector DB, native hybrid search
- **Postgres 16 + pgvector** — relational warehouse for XBRL facts + backup vector store
- **Redis 7** — cache, ARQ task queue, WebSocket pubsub
- **MinIO** — S3-compatible object store for raw PDFs

### Data Sources (all free)
- **SEC EDGAR** — 10-K, 10-Q, 8-K, XBRL facts API (completely free, no auth, rate limit: 10 req/sec)
- **Tavily API** — 1000 searches/month free, no CC
- **yfinance** — free, unofficial (prices, OHLCV)
- **Financial Modeling Prep** — free tier, 250 req/day, no CC
- **Alpha Vantage** — free tier, no CC (rate-limited backup)

### Document Processing (local, free)
- **Docling** (IBM open source) — best-in-class PDF + table extraction
- **PyMuPDF** — fast text extraction
- **python-edgar 3.x** + custom XBRL parser — structured financial data extraction

### Orchestration & Backend
- **LangGraph** + **LangChain** — agent orchestration, StateGraph, checkpointing
- **FastAPI** + **Uvicorn** + **WebSockets** — async API gateway
- **Pydantic v2** — typed agent state
- **SQLAlchemy 2.0 (async)** + **Alembic** — ORM + migrations
- **ARQ** — Redis-native async task queue
- **Tenacity** — retry/backoff

### Frontend
- **React 18** + **TypeScript** + **Vite**
- **TanStack Query** + **Zustand** — data fetching + state
- **shadcn/ui** + **Tailwind CSS** + **Radix** — component library
- **React Flow** — live agent-DAG visualization (nodes light up as they execute)
- **Recharts** — charts
- **react-markdown** + **remark-gfm** — rich content rendering

### Observability & Eval (self-hosted, free)
- **Langfuse v2** (self-hosted in Docker) — full LLM tracing, cost tracking, prompt management
- **RAGAS** — retrieval & generation eval (faithfulness, answer relevance, context precision/recall)
- **OpenTelemetry** instrumentation

### DevOps
- **Docker** + **docker-compose** — local primary
- **GitHub Actions** — CI (free for public repos)
- **ruff** + **mypy** + **pytest** + **pre-commit**
- **uv** — Python package/workspace manager
- **pnpm** — Node package/workspace manager
- Cloud (optional, Phase 9): **HuggingFace Spaces** (backend) + **Vercel** (frontend) + **Fly.io** (optional)

---

## 8. Agent Architecture

**LangGraph `StateGraph`** with typed `AgentState` (defined in `packages/shared/src/mia_shared/schemas.py`):
```python
class AgentState(BaseModel):
    session_id: UUID
    query: str
    plan: str
    messages: list[dict]
    evidence: list[Evidence]
    citations: list[Citation]
    draft: str
    critique: CritiqueResult | None
    iteration_count: int
    human_approval_required: bool
    active_agent: AgentName | None
    error: str | None
```

**7 Agents (Supervisor + 6 Workers):**

| Agent | Role | Primary LLM | Key Tools |
|---|---|---|---|
| Supervisor / Planner | Decomposes query → plan; routes; decides "done" | Gemini 2.0 Flash | `route()`, `replan()` |
| Web Search | Open-web evidence; source credibility scoring | Groq Llama 3.3 70B | Tavily API, Brave Search fallback, trafilatura |
| EDGAR / Financial Parser | 10-K/10-Q/8-K ingestion, XBRL structured extraction, MD&A/Risk Factors | Groq | `sec-edgar-downloader`, `python-edgar`, XBRL→Postgres ETL |
| Retrieval / RAG | Hybrid search (BM25 + dense) + cross-encoder rerank | Groq | Qdrant, `rank_bm25`, `bge-reranker-v2-m3` |
| SQL Generator | NL→SQL over Postgres XBRL warehouse | Gemini Flash | LangChain SQL toolkit + read-only role + query validator |
| Summarizer / Writer | Final brief with inline citations, charts, tables | Gemini 2.0 Flash | Markdown + Mermaid + Vega-Lite spec emitter |
| Critic / Verifier | Claim-level grounding check; triggers self-correction | Groq Llama 3.3 70B | NLI via `cross-encoder/nli-deberta-v3-base`, citation-checker |

**Self-correction loop:**
- Critic emits `{verdict: pass|revise|escalate, failing_claims: [...]}`
- On `revise` → Supervisor re-routes only failing subtasks (not full restart), up to `max_iterations=3`
- On `escalate` → LangGraph `interrupt()` fires a human-in-the-loop pause

---

## 9. Phase Plan

| # | Phase | Status | Rough Effort |
|---|---|---|---|
| 0 | Bootstrap & DevOps foundation | ✅ Done | — |
| 1 | Ingestion pipelines (EDGAR downloader, XBRL→Postgres ETL, PDF pipeline, MinIO storage) | ✅ Done | — |
| 2 | Retrieval stack (Qdrant indexing, BM25, hybrid search, bge-reranker) | ✅ Done | — |
| 3 | Single-agent RAG baseline (establishes benchmark before multi-agent) | ✅ | — |
| 4 | LangGraph multi-agent skeleton (Supervisor + workers, StateGraph wiring) | ⬜ | 1.5 weeks |
| 5 | Critic agent + self-correction loops (NLI grounding, iteration cap, HITL) | ⬜ | 1 week |
| 6 | FastAPI + WebSocket streaming + ARQ workers | ⬜ | 1 week |
| 7 | React dashboard (chat UI, React Flow DAG, citation viewer, cost panel) | ⬜ | 1.5 weeks |
| 8 | Observability (Langfuse), eval harness (RAGAS), ablation studies (12-cell matrix) | ⬜ | 1.5 weeks |
| 9 | Stress test, hardening, optional cloud deploy, WRITEUP.md, demo video, README polish | ⬜ | 1 week |

**Total:** ~10 weeks part-time.

---

## 10. Evaluation Strategy

**Ablation matrix (12 cells):**
`{BM25 | dense | hybrid} × {no rerank | bge-rerank} × {no critic | critic}`

**Golden eval set (Phase 7–8):**
- 50 hand-authored Q/A pairs: single-hop, multi-hop, comparative, quantitative
- Each Q tagged with ground-truth document IDs and exact text spans
- Used to quantify and defend the "28% retrieval precision improvement"

**Metrics by layer:**

| Layer | Metric | Tool |
|---|---|---|
| Retrieval | Recall@k, MRR, nDCG | Custom + RAGAS |
| Reranking | nDCG@10 lift vs base retrieval | Custom |
| Generation | Faithfulness, answer relevance, context precision | RAGAS |
| Agent path | Plan optimality, tool-call correctness, self-correction accuracy | Custom |
| End-to-end | Pass@1 on golden set | Custom |
| Latency/cost | p50/p95 per agent step | Langfuse |

---

## 11. Free-Tier Budget

| Resource | Daily ceiling | Estimated dev burn | Safety margin |
|---|---|---|---|
| Gemini 2.0 Flash | 1500 req/day | ~200 req/day heavy dev | 7× headroom |
| Groq Llama 3.3 70B | ~14k req/day | ~500 req/day | 28× headroom |
| Tavily Search | ~33 req/day | ~15 req/day | 2× — cache aggressively |
| SEC EDGAR | No limit (10 req/sec rate) | N/A | Safe with async throttle |

**Tavily mitigation:** SQLite-backed search cache, 7-day freshness window, `--no-cache` flag for evals.

---

## 12. Interview Prep Setup

### Structure
- `INTERVIEW_PREP.md` — full depth (one-liner / 30-sec / 90-sec answers, follow-ups, linked artifacts, confidence tracker)
- `CHEATSHEET.md` — auto-extracted one-liners, night-before read
- **53 questions** across 6 tiers (Q1–Q52 + adversarial follow-ups)
- **Confidence legend:** 🟢 sharp | 🟡 needs reps | 🔴 not ready

### What's done
- Q2 (motivation) — fully drafted with all three answer lengths
- All 53 question slots created with empty answer templates
- Phase 0 design decisions documented in `docs/PHASE_0_SUMMARY.md`

### What's pending
- Q1 and Q3–Q52 — answers not yet drafted
- Answers will be filled in back-and-forth with Claude as the project is built

### Important interview notes
- **Never lead with the tech stack** — lead with the *question* being answered
- **Never say "I wanted to learn LangGraph"** — sounds résumé-driven
- When asked "why not ChatGPT with browsing?" → three failure modes: XBRL parsing, no grounding check, no auditable trace

---

## 13. Definition of Done (Full Project)

- `docker compose up` brings the full system live on a clean machine in under 10 minutes
- Submitting a query streams agent events to the dashboard in real-time
- Final brief includes inline citations linking to source PDFs/URLs
- Critic agent demonstrably catches a hallucinated claim in at least one golden eval case
- Ablation table reports retrieval precision lift with 95% confidence intervals
- Langfuse dashboard shows full trace of any query with token/latency/cost breakdown
- `pytest` green; `mypy --strict` clean on `packages/`
- `WRITEUP.md` + 2-minute Loom demo + architecture diagram committed
- README has a one-paste "try it" block
- Code repo flipped to public on GitHub

---

## 14. Open Items (carry into Phase 4)

**Deferred from Phase 1 (still open):**
- [ ] Run `uv lock` and commit the lockfile
- [ ] `uv run pre-commit install` to activate hooks locally
- [ ] Flip repo to public (Phases 1+2+3 have substantive, defensible code now)
- [ ] Open decisions: org vs personal repo? demo persona (analyst vs consultant)? writeup format?
- [ ] Add `ingestion_jobs` tracking table (Phase 6, deferred)
- [ ] Handle 10-K/A and 10-Q/A amendments (future iteration)

**From Phase 2:**
- [ ] Section-aware chunking: detect MD&A/Risk Factors/Financials headers and set `Chunk.section`
- [ ] Sentence-boundary-aware split: avoid mid-sentence cuts (optional: `nltk sent_tokenize`)
- [ ] ARQ task wrapper for `IndexingPipeline.index_ticker()` (Phase 6)
- [ ] S&P 100 batch indexing script: `make index-all` loops all tickers
- [ ] pgvector fallback if Qdrant is down

**From Phase 3:**
- [ ] Langfuse tracing in `RAGAgent.run()` — wrap with `@observe` decorator once keys are configured
- [ ] `make query` E2E smoke test on real indexed data (requires `make ingest + index` first)
- [ ] Add `--verbose` flag to `scripts/query.py` for full chunk text output

**Phase 3 data flow:**
```
make ingest ticker=NVDA    # Phase 1 — filing → Postgres + MinIO
make index ticker=NVDA     # Phase 2 — chunks → Qdrant + BM25
make query q="NVDA data center revenue growth"   # Phase 3 — RAG answer
```

---

## 15. How to Continue in a New Session

**Paste `docs/CONTEXT.md`** at the start of the session, then say what phase/component to work on.

**To resume infra after restart:**
```bash
cd ~/Documents/Projects/MarketIntelAgent
make up-infra   # starts postgres, redis, qdrant, minio, langfuse-db, langfuse-web
make ps         # verify all healthy
```

**Data seeding (run in order if starting fresh):**
```bash
make migrate                   # ensure mia.filings + xbrl.facts exist
make init-minio                # create sec-filings bucket
make ingest ticker=NVDA        # Phase 1: EDGAR → Postgres + MinIO
make index ticker=NVDA         # Phase 2: Postgres → Qdrant + BM25
make retrieve query="NVDA revenue growth"  # smoke test retrieval
make query q="How is NVDA's data center revenue growing?"  # Phase 3: full RAG answer
```

**Useful Phase 3 variants:**
```bash
make query-bm25  q="NVDA supply chain risk"         # BM25-only (no embedder, fastest)
make query-dense q="AMD vs NVDA GPU margins"         # dense-only (Qdrant, no BM25)
make query-groq  q="Apple Services revenue growth"   # pin LLM to Groq
uv run python scripts/query.py "..." --tickers NVDA AMD --mode hybrid
```

**Phase 1 data confirmed seeded (NVDA):** 4,937 XBRL facts, 21 filings in Postgres.
**Phase 2 index:** Run `make index ticker=NVDA` to populate Qdrant + `data/bm25_index.pkl`.
**Phase 3 RAG:** Run `make query q="..."` after Phase 2 is complete.

**Note:** Qdrant uses ports 6335/6336 on this machine (not default 6333/6334).

**Interview prep docs:**
- `docs/PHASE_3_SUMMARY.md` — standalone, paste into any Claude chat to prep on Phase 3 decisions
- `docs/PHASE_2_SUMMARY.md` — same for Phase 2 (retrieval stack)
- `docs/PHASE_1_SUMMARY.md` — same for Phase 1 (ingestion pipeline)
- `docs/PHASE_0_SUMMARY.md` — same for Phase 0 (infrastructure / DevOps)
