# Autonomous Enterprise Market Intelligence Agent — Master Plan

> **Status:** Overview / awaiting approval. After sign-off, this expands into phased build docs (`PHASE_1.md` … `PHASE_N.md`).
> **Author target:** Master's CS student portfolio + capstone-grade artifact.
> **Hard constraint:** $0 budget, no credit card on file *anywhere*.
> **Time budget:** ~2–3 months part-time (capstone scope).
> **Domain:** Public-company financial intelligence (SEC EDGAR + market data).

---

## 1. Elevator Pitch

A multi-agent system that, given a natural-language question about a public company or sector (e.g., *"How is NVDA's data-center revenue concentration evolving vs AMD's, and what's the risk narrative in their latest 10-Ks?"*), autonomously:

1. **Plans** a research strategy (Supervisor decomposes into subtasks).
2. **Gathers** evidence from SEC EDGAR filings, news, web sources, and a structured Postgres warehouse.
3. **Synthesizes** a cited, citation-grounded report with charts, with a **Critic Agent** verifying claims against retrieved evidence and triggering self-correction loops on hallucination.
4. **Streams** each agent's thought process, tool calls, and intermediate state into a live React dashboard via WebSockets.

The user can interrupt, redirect, or approve mid-flight (human-in-the-loop). Final output: a downloadable Markdown/PDF brief with traceable citations back to source PDFs/URLs and chart-of-thought visualization.

---

## 2. What Makes This Capstone-Grade (Beyond the Resume Bullets)

The original resume bullets are solid but interview-fragile. To survive deep technical scrutiny, the build adds:

| Differentiator | Why it matters |
|---|---|
| **Quantified RAG ablation** (BM25 / dense / hybrid / hybrid+rerank) on a *real* SEC-derived golden set — not a synthetic toy | Lets you defend the "28% precision improvement" with a reproducible table and confidence intervals |
| **Self-correction loop with a Critic Agent** using NLI-style claim-grounding | Most student multi-agent projects stop at "supervisor routes; workers run." A grounded critic is genuinely state-of-the-art |
| **LangGraph state checkpointing + time-travel debugging** | Demonstrates production-grade agentic engineering, not just a prompt chain |
| **End-to-end LLM observability** via self-hosted **Langfuse** | Every span, every token, every cost — interviewers love this |
| **RAGAS + custom agent-eval harness** with golden Q/A pairs | Quantitative evaluation, not vibes |
| **Hybrid retrieval over both unstructured (PDFs) AND structured (XBRL/SQL)** | The SQL-Generator agent works over actual SEC-extracted financials, not a toy schema |
| **Live agent DAG visualization** (React Flow) showing nodes light up as they execute | High wow-factor in a 90-second demo |
| **Concurrency stress test**: 50+ documents indexed; 10+ concurrent user sessions | Backs the "50+ industry reports" claim with a benchmark script |
| **Failure-mode taxonomy + short writeup** (`WRITEUP.md`) cataloging where agents fail and why | Capstone-quality reflection that sets you apart from resume-grinders |

---

## 3. System Architecture (High-Level)

```
┌────────────────────────────────────────────────────────────────────────────┐
│                            React 18 + Vite Dashboard                       │
│  • Chat UI  • Live Agent-DAG (React Flow)  • Citation viewer  • Cost panel │
└──────────────────────────────┬─────────────────────────────────────────────┘
                               │ WebSocket (agent events) + REST (resources)
┌──────────────────────────────▼─────────────────────────────────────────────┐
│                        FastAPI Gateway (async)                             │
│   /chat/stream (WS) · /sessions · /documents · /traces · /eval             │
└──┬────────────────────┬──────────────────────────────────┬─────────────────┘
   │                    │                                  │
   │ enqueue            │ stream events                    │ resource ops
   ▼                    ▼                                  ▼
┌──────────────┐   ┌──────────────────────────────┐   ┌───────────────────┐
│  ARQ Worker  │   │   LangGraph Orchestrator     │   │  Postgres + pgvec │
│   (Redis)    │◄──┤   Supervisor + 6 workers     │──►│  Qdrant           │
└──────────────┘   │   StateGraph + Checkpointer  │   │  Redis (cache/PS) │
                   └──┬───────────────────────────┘   │  MinIO (S3 OSS)   │
                      │                                └───────────────────┘
                      │ tool calls
       ┌──────────────┼──────────────────┬──────────────┬────────────┐
       ▼              ▼                  ▼              ▼            ▼
  ┌─────────┐  ┌──────────────┐   ┌─────────────┐ ┌──────────┐ ┌──────────┐
  │ Gemini  │  │  Groq /      │   │ Tavily Web  │ │ SEC      │ │ yfinance │
  │ 2.0     │  │  Cerebras    │   │ Search API  │ │ EDGAR    │ │  Alpha-  │
  │ Flash   │  │  (Llama 3.3) │   │ (free tier) │ │ (free)   │ │ Vantage  │
  └─────────┘  └──────────────┘   └─────────────┘ └──────────┘ └──────────┘

  Cross-cutting: Langfuse (traces) · OpenTelemetry · Prometheus + Grafana (optional)
```

---

## 4. Agent Topology (Supervisor + 6 Workers, expanded from 4)

LangGraph `StateGraph` with a typed `AgentState` carrying: `messages`, `plan`, `evidence[]`, `citations[]`, `draft`, `critique`, `iteration_count`, `human_approval_required`.

| Agent | Role | Primary LLM | Tools |
|---|---|---|---|
| **Supervisor / Planner** | Decomposes query → plan; routes; decides "done" | Gemini 2.0 Flash (1M context, structured output) | `route()`, `replan()` |
| **Web Search** | Open-web evidence; deduplicates; scores source credibility | Groq Llama 3.3 70B | Tavily API, Brave Search (fallback), trafilatura for extraction |
| **EDGAR / Financial Parser** | Pulls 10-K/10-Q/8-K, parses **XBRL** for structured financials, extracts MD&A / Risk Factors sections | Groq | `sec-edgar-downloader`, `python-edgar`, custom XBRL→Postgres ETL |
| **Retrieval / RAG** | Hybrid (BM25 + dense) over indexed corpus, cross-encoder rerank | Groq | Qdrant (dense), `rank_bm25` (sparse), `bge-reranker-v2-m3` |
| **SQL Generator** | NL→SQL over Postgres warehouse of XBRL facts, prices, metrics | Gemini Flash | LangChain SQL toolkit + read-only role + query validator |
| **Summarizer / Writer** | Composes final brief with inline citations, charts, tables | Gemini 2.0 Flash (long context shines here) | Markdown + Mermaid + Vega-Lite spec emitter |
| **Critic / Verifier** | Claim-level grounding check; triggers loops; flags unverified claims | Groq Llama 3.3 70B (structured JSON output) | NLI scoring via local `cross-encoder/nli-deberta-v3-base`, citation-checker |

**Self-correction loop:** Critic emits `{verdict: pass|revise|escalate, failing_claims: [...]}`. On `revise`, Supervisor re-routes only the failing subtasks (not full restart) up to `max_iterations` (default 3). On `escalate`, human-in-the-loop interrupt fires via LangGraph's `interrupt()`.

---

## 5. Tech Stack — All Free, Zero Credit Card

Every entry below has been chosen because either (a) it's open source and self-hostable in Docker, or (b) it has a free API tier that does *not* require a credit card to start.

### 5.1 LLM Inference (free, no CC)
| Provider | Model | Free quota | Role |
|---|---|---|---|
| **Google AI Studio** | Gemini 2.0 Flash | 15 RPM, 1M TPM, 1500 RPD, **1M-token context** | Long-context reasoning, structured output |
| **Groq** | Llama 3.3 70B, Llama 3.1 8B, Mixtral | ~30 RPM, generous TPD | Fast agent steps (800 tok/s) |
| **Cerebras** | Llama 3.3 70B | ~30 RPM | Fallback if Groq throttles (~2000 tok/s) |
| **OpenRouter** | Various free models | Variable | Final fallback |

### 5.2 Embeddings & Reranking (local, free)
- `BAAI/bge-large-en-v1.5` — dense embeddings (1024-dim) via `sentence-transformers`
- `BAAI/bge-reranker-v2-m3` — cross-encoder reranker, beats Cohere rerank-3 on many benches
- `rank_bm25` — sparse retrieval for hybrid

### 5.3 Vector & Data Stores (self-hosted in Docker, free)
- **Qdrant** — vector DB; native hybrid search; better filtering than Pinecone's free tier; no quota
- **Postgres 16 + pgvector** — relational warehouse (XBRL facts, users, sessions) + backup vector store
- **Redis 7** — cache, ARQ task queue, WebSocket pubsub
- **MinIO** — S3-compatible object store for raw PDFs

### 5.4 Search & Data Sources (free)
- **Tavily API** — 1000 searches/month free, no CC
- **SEC EDGAR** — completely free, no auth (10-Ks, 10-Qs, 8-Ks, XBRL facts API)
- **yfinance** — free, unofficial (prices, OHLCV)
- **Financial Modeling Prep** — free tier (250 req/day), no CC required
- **Alpha Vantage** — free tier, no CC (rate-limited backup)

### 5.5 Document Processing (local, free)
- **Docling** (IBM open source) — best-in-class PDF + table extraction
- **PyMuPDF** — fast text extraction
- **Unstructured.io** (open source) — fallback / chunking utilities
- Custom XBRL parser via `python-edgar`

### 5.6 Orchestration & Backend
- **LangGraph** + **LangChain** — agent orchestration
- **FastAPI** + **Uvicorn** + **WebSockets**
- **Pydantic v2** for typed agent state
- **SQLAlchemy 2.0** (async) + **Alembic** migrations
- **ARQ** — Redis-native async task queue (simpler than Celery)
- **Tenacity** — retry / backoff

### 5.7 Frontend
- **React 18** + **TypeScript** + **Vite**
- **TanStack Query** + **Zustand**
- **shadcn/ui** + **Tailwind CSS** + **Radix**
- **React Flow** — live agent-DAG visualization
- **Recharts** + **Vega-Lite** (`react-vega`)
- **react-markdown** + **rehype-katex** + **shiki**

### 5.8 Observability & Eval (self-hosted, free)
- **Langfuse** — self-hosted in Docker; full LLM tracing, cost tracking, prompt management
- **RAGAS** — retrieval & generation eval (faithfulness, answer relevance, context precision/recall)
- **DeepEval** (or custom) — agent-level eval
- **OpenTelemetry** instrumentation
- **Prometheus + Grafana** (optional, Phase 8)

### 5.9 DevOps (free)
- **Docker** + **docker-compose** (local primary)
- **GitHub Actions** — CI (free for public repos)
- **ruff** + **mypy** + **pytest** + **pre-commit**
- **HuggingFace Spaces** (Docker SDK) for cloud demo — free, no CC
- **Vercel** (frontend) — free, no CC
- **Fly.io** free tier (optional, no CC required at small scale)

---

## 6. Free-Tier Budget Sanity Check

| Resource | Daily ceiling | Project burn estimate | Safety margin |
|---|---|---|---|
| Gemini 2.0 Flash | 1500 req/day | ~200 req/day in heavy dev | 7× headroom |
| Groq Llama 3.3 70B | ~14k req/day | ~500 req/day | 28× |
| Tavily Search | ~33 req/day | ~15 req/day | 2× — **need to cache aggressively** |
| SEC EDGAR | None (rate-limit: 10 req/sec) | N/A — async-throttle | Safe |

**Mitigation for Tavily ceiling:** SQLite-backed search cache, configurable freshness window (default 7 days), `--no-cache` override for evals.

---

## 7. Data & Domain

### 7.1 Corpus (Phase 2 seed)
- **Universe:** S&P 100 tickers (manageable for capstone scope; reuses partial overlap with your DJ30 FinRL universe for narrative coherence)
- **Documents:** Latest 10-K + last 4 10-Qs + last 8 8-Ks per ticker → ~1500 documents → realistically backs the "50+ industry reports" resume claim by an order of magnitude
- **Structured:** XBRL facts loaded into Postgres (revenue, segments, R&D, debt, EPS …) for the SQL Generator
- **News:** rolling 90-day window via NewsAPI free tier or RSS

### 7.2 Golden eval set (Phase 7)
- **Hand-authored:** 50 Q/A pairs across single-hop, multi-hop, comparative, and quantitative questions
- **Sources:** Each Q tagged with ground-truth document IDs and exact text spans
- Used for: retrieval precision/recall, end-to-end answer faithfulness, agent-path correctness

---

## 8. Evaluation Strategy

| Layer | Metric | Tool |
|---|---|---|
| Retrieval | Recall@k, MRR, nDCG | Custom + RAGAS |
| Reranking | nDCG@10 lift vs base retrieval | Custom (the **"28% precision lift"** quantification) |
| Generation | Faithfulness, answer relevance, context precision | RAGAS |
| Agent path | Plan optimality, tool-call correctness, self-correction trigger accuracy | Custom |
| End-to-end | Pass@1 on golden set, human eval on 10-question subset | Custom |
| Latency / cost | p50/p95 per agent step, $-per-query (synthetic, since free) | Langfuse |

**Ablation matrix:** `{retriever ∈ [BM25, dense, hybrid]} × {reranker ∈ [none, bge-rerank]} × {critic ∈ [off, on]}` = 12 cells.

---

## 9. Deployment Strategy

### Local-first (Phase 1 onward)
```
docker compose up -d
# brings up: api, worker, postgres, qdrant, redis, minio, langfuse, frontend
```
Single command demo for video / interviews.

### Cloud demo (Phase 9, optional)
- **Backend (API + worker):** HuggingFace Spaces (Docker SDK) — free, no CC
- **Frontend:** Vercel — free, no CC
- **Qdrant + Postgres:** Qdrant Cloud free tier (1GB, no CC) + Supabase free Postgres (no CC)
- **Redis:** Upstash free tier (no CC)

Public URL on the resume. Lightweight enough that even with rate limits, the live demo is rate-limited per IP and won't burn quota.

---

## 10. Repository Layout (proposed)

```
market-intel-agent/
├── apps/
│   ├── api/                 # FastAPI gateway
│   ├── worker/              # ARQ workers
│   └── web/                 # React + Vite frontend
├── packages/
│   ├── agents/              # LangGraph agents, state, tools
│   ├── retrieval/           # Hybrid + reranker
│   ├── ingestion/           # EDGAR, XBRL, PDF, news pipelines
│   ├── eval/                # RAGAS harness, golden set, ablation scripts
│   └── shared/              # Pydantic schemas, prompts, constants
├── infra/
│   ├── docker-compose.yml
│   ├── docker-compose.cloud.yml
│   └── langfuse/
├── data/                    # gitignored; raw + processed
├── notebooks/               # exploration, ablation plots
├── docs/
│   ├── PLAN.md              # this file
│   ├── PHASE_*.md           # per-phase build docs
│   ├── ARCHITECTURE.md
│   ├── EVAL.md
│   └── WRITEUP.md           # final reflection
├── .github/workflows/
└── README.md
```

Monorepo via **uv workspaces** (Python) + **pnpm workspaces** (JS).

---

## 11. Phase Outline (titles only — expanded after approval)

| # | Phase | Rough effort |
|---|---|---|
| 0 | Bootstrap & DevOps foundation | 3–4 days |
| 1 | Ingestion pipelines (EDGAR + XBRL + PDF) | 1.5 weeks |
| 2 | Retrieval stack (Qdrant + BM25 + reranker) | 1 week |
| 3 | Single-agent RAG baseline (no multi-agent yet) | 4–5 days |
| 4 | LangGraph multi-agent skeleton (Supervisor + 3 workers) | 1.5 weeks |
| 5 | Critic agent + self-correction loops | 1 week |
| 6 | FastAPI + WebSocket streaming + ARQ workers | 1 week |
| 7 | React dashboard (chat + DAG viz + citations) | 1.5 weeks |
| 8 | Observability (Langfuse), eval harness (RAGAS), ablation studies | 1.5 weeks |
| 9 | Stress test, hardening, optional cloud deploy, WRITEUP + demo video | 1 week |

Total: **~10 weeks part-time**, with built-in slack for the inevitable yak-shaves.

---

## 12. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Free LLM quota throttling mid-eval | Med | Med | Multi-provider fallback chain (Gemini → Groq → Cerebras → OpenRouter); local cache on Q/A pairs |
| Tavily 1k/month exhausted | High | Med | Aggressive caching, batch evals, use SEC-only mode for ablations |
| Qdrant local OOM on large corpus | Low | Low | Quantized vectors (int8); sharded collections per ticker |
| XBRL parsing edge cases | High | Med | Start with revenue/EPS only; expand iteratively; log + skip on parse fail |
| Self-correction infinite loop | Med | Med | Hard cap on iterations + Langfuse alerting |
| Scope creep killing capstone timeline | High | High | Phases gated by exit criteria; phase 9 = hardening, not new features |

---

## 13. Open Questions for You

These don't block writing Phase 1, but answers will sharpen the build:

1. **GitHub repo:** want me to create it under your `dhruvi002` account (matching `dynamic-portfolio-optimization`)? Public for free CI?
2. **Demo persona:** is the imagined end-user a **buy-side equity analyst** (deep single-name dives) or a **strategy consultant** (cross-industry landscape scans)? Both work; one frames the UI copy.
3. **Writeup ambition:** a polished `WRITEUP.md` is in scope. Want it formatted toward (a) a blog post / Medium article, (b) an arXiv-style short PDF, or (c) both?
4. **Anonymized telemetry:** Langfuse is self-hosted so no data leaves your machine — confirming that's the right call vs. cloud Langfuse free tier (which would also be no-CC).

---

## 14. Definition of Done (for the whole project)

- `docker compose up` brings the system live on a clean machine in under 10 minutes
- Submitting a query streams agent events to the dashboard in real-time
- Final brief includes inline citations linking to source PDFs/URLs
- Critic agent demonstrably catches a hallucinated claim in at least one golden eval case
- Ablation table reports retrieval precision lift with 95% CIs
- Langfuse dashboard shows full trace of any query with token / latency / cost breakdown
- `pytest` green; `mypy --strict` clean on `packages/`
- `WRITEUP.md` + 2-minute Loom demo + architecture diagram committed
- README has a one-paste "try it" block

---

**Awaiting your approval / edits before I expand into `PHASE_0.md` through `PHASE_9.md`.**
