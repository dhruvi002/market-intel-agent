# Session Context — Autonomous Enterprise Market Intelligence Agent
> **Purpose:** Drop this file into a new Claude Code session to restore full project context instantly.
> **Last updated:** 2026-06-06
> **Session recap:** Project conceived, planned, and repos initialized. No code written yet — Phase 0 is next.

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

---

## 4. Repositories

### Public Code Repo
- **GitHub:** https://github.com/dhruvi002/market-intel-agent (currently private — flip public in Phase 0 once skeleton has substance)
- **Local:** `/Users/dhruvishah/Documents/Projects/MarketIntelAgent/`
- **Remote:** `https://github.com/dhruvi002/market-intel-agent.git`
- **Branch:** `main`

### Private Interview Prep Repo
- **GitHub:** https://github.com/dhruvi002/prep-market-intel-agent (always private)
- **Local:** `/Users/dhruvishah/Documents/Projects/PrepMarketIntelAgent/`
- **Remote:** `https://github.com/dhruvi002/prep-market-intel-agent.git`
- **Branch:** `main`

---

## 5. Local Folder Structure (Code Repo)

```
/Users/dhruvishah/Documents/Projects/MarketIntelAgent/
├── apps/
│   ├── api/                  # FastAPI gateway (async, WebSocket)
│   ├── worker/               # ARQ async task workers (Redis-backed)
│   └── web/                  # React 18 + TypeScript + Vite frontend
├── packages/
│   ├── agents/               # LangGraph StateGraph, Supervisor + 6 workers, tools
│   ├── retrieval/            # Hybrid search (BM25 + dense) + cross-encoder reranker
│   ├── ingestion/            # EDGAR downloader, XBRL parser, PDF pipeline, news
│   ├── eval/                 # RAGAS harness, golden Q/A set, ablation scripts
│   └── shared/               # Pydantic schemas, prompts, constants
├── infra/
│   ├── docker-compose.yml    # Local: api, worker, postgres, qdrant, redis, minio, langfuse, web
│   ├── docker-compose.cloud.yml
│   └── langfuse/
├── data/                     # gitignored — raw filings, parquet, model weights
├── notebooks/                # Exploration, ablation plots
├── docs/
│   ├── PLAN.md               # Full architecture plan (the master doc)
│   ├── CONTEXT.md            # This file
│   ├── EVAL.md               # Eval results (populated in Phase 8)
│   ├── WRITEUP.md            # Final reflection (populated in Phase 9)
│   └── PHASE_*.md            # Per-phase build docs (not yet created)
├── .github/workflows/        # CI (GitHub Actions)
├── .gitignore
└── README.md
```

---

## 6. Full Tech Stack

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
- **python-edgar** + custom XBRL parser — structured financial data extraction

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
- **Recharts** + **Vega-Lite** (`react-vega`) — charts
- **react-markdown** + **rehype-katex** + **shiki** — rich content rendering

### Observability & Eval (self-hosted, free)
- **Langfuse** (self-hosted in Docker) — full LLM tracing, cost tracking, prompt management
- **RAGAS** — retrieval & generation eval (faithfulness, answer relevance, context precision/recall)
- **OpenTelemetry** instrumentation

### DevOps
- **Docker** + **docker-compose** — local primary
- **GitHub Actions** — CI (free for public repos)
- **ruff** + **mypy** + **pytest** + **pre-commit**
- Cloud (optional, Phase 9): **HuggingFace Spaces** (backend) + **Vercel** (frontend) + **Fly.io** (optional)

---

## 7. Agent Architecture

**LangGraph `StateGraph`** with typed `AgentState`:
```python
# Fields in AgentState
messages: list
plan: str
evidence: list[Evidence]
citations: list[Citation]
draft: str
critique: CritiqueResult
iteration_count: int
human_approval_required: bool
```

**6 Agents (Supervisor + 5 Workers):**

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

## 8. Phase Plan

| # | Phase | Status | Rough Effort |
|---|---|---|---|
| 0 | Bootstrap & DevOps foundation (Docker Compose, CI, pre-commit, env management) | ⬜ Next up | 3–4 days |
| 1 | Ingestion pipelines (EDGAR downloader, XBRL→Postgres ETL, PDF pipeline, MinIO storage) | ⬜ | 1.5 weeks |
| 2 | Retrieval stack (Qdrant indexing, BM25, hybrid search, bge-reranker) | ⬜ | 1 week |
| 3 | Single-agent RAG baseline (no multi-agent yet — establishes the benchmark) | ⬜ | 4–5 days |
| 4 | LangGraph multi-agent skeleton (Supervisor + 3 workers, StateGraph wiring) | ⬜ | 1.5 weeks |
| 5 | Critic agent + self-correction loops (NLI grounding, iteration cap, human-in-the-loop) | ⬜ | 1 week |
| 6 | FastAPI + WebSocket streaming + ARQ workers (session management, event bus) | ⬜ | 1 week |
| 7 | React dashboard (chat UI, React Flow DAG, citation viewer, cost panel) | ⬜ | 1.5 weeks |
| 8 | Observability (Langfuse), eval harness (RAGAS), ablation studies (12-cell matrix) | ⬜ | 1.5 weeks |
| 9 | Stress test, hardening, optional cloud deploy, WRITEUP.md, demo video, README polish | ⬜ | 1 week |

**Total:** ~10 weeks part-time.

**Phase 0 starts next** — create `docker-compose.yml`, set up all services, configure env management (uv + pnpm workspaces), pre-commit hooks, GitHub Actions CI skeleton.

---

## 9. Evaluation Strategy

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

## 10. Free-Tier Budget

| Resource | Daily ceiling | Estimated dev burn | Safety margin |
|---|---|---|---|
| Gemini 2.0 Flash | 1500 req/day | ~200 req/day heavy dev | 7× headroom |
| Groq Llama 3.3 70B | ~14k req/day | ~500 req/day | 28× headroom |
| Tavily Search | ~33 req/day | ~15 req/day | 2× — cache aggressively |
| SEC EDGAR | No limit (10 req/sec rate) | N/A | Safe with async throttle |

**Tavily mitigation:** SQLite-backed search cache, 7-day freshness window, `--no-cache` flag for evals.

---

## 11. Interview Prep Setup

### Structure
- `INTERVIEW_PREP.md` — full depth (one-liner / 30-sec / 90-sec answers, follow-ups, linked artifacts, confidence tracker)
- `CHEATSHEET.md` — auto-extracted one-liners, night-before read
- **53 questions** across 6 tiers (Q1–Q52 + adversarial follow-ups)
- **Confidence legend:** 🟢 sharp | 🟡 needs reps | 🔴 not ready

### What's done
- Q2 (motivation) — fully drafted with all three answer lengths
- All 53 question slots created with empty answer templates
- Adversarial follow-ups section included

### What's pending
- Q1 and Q3–Q52 — answers not yet drafted
- Answers will be filled in back-and-forth with Claude as the project is built
- Workflow: discuss a question → Claude fills in INTERVIEW_PREP.md directly → no manual transcription

### Important interview notes
- **Never lead with the tech stack** — lead with the *question* being answered
- **Never say "I wanted to learn LangGraph"** — sounds résumé-driven
- When asked "why not ChatGPT with browsing?" → three failure modes: XBRL parsing, no grounding check, no auditable trace

---

## 12. Definition of Done (Full Project)

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

## 13. Decisions Still Open (not yet discussed)

1. Should the public repo be initialized under a GitHub organization or stay under `dhruvi002`?
2. Demo persona framing: buy-side equity analyst (single-name deep dives) vs. strategy consultant (cross-industry scans)?
3. `WRITEUP.md` format: blog post / Medium article, arXiv-style short PDF, or both?
4. Langfuse: self-hosted (confirmed no data leaves machine) vs. cloud free tier — self-hosted is current plan.

---

## 14. How to Continue in a New Session

**Paste this file's content** at the start of the session, then say what you want to work on. The assistant will have full context on:
- Project architecture and all decisions made
- Repo locations (local + GitHub)
- What's built vs. what's pending
- Interview prep state
- Constraints (free tier, no CC, no co-author attribution in commits)

**Likely next actions:**
- Start Phase 0: `docker-compose.yml` with all 8 services, uv workspace setup, pre-commit hooks, GitHub Actions CI
- Continue interview prep: draft answers for Q1 (elevator pitch), then Q3–Q7
- Flip code repo to public once Phase 0 skeleton is committed
