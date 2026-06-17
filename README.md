# Autonomous Enterprise Market Intelligence Agent

[![CI](https://github.com/dhruvi002/market-intel-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/dhruvi002/market-intel-agent/actions/workflows/ci.yml)

A production-grade multi-agent system that answers natural-language questions
about public companies by autonomously gathering evidence from SEC filings,
financial databases, and open-web sources — streaming every agent's reasoning
step live to a React dashboard.

---

## Quick Start (one command)

```bash
# 1. Clone and copy env vars
git clone https://github.com/dhruvi002/market-intel-agent.git
cd market-intel-agent
cp .env.example .env          # fill in GEMINI_API_KEY, GROQ_API_KEY, TAVILY_API_KEY

# 2. Start all services
docker compose -f infra/docker-compose.yml --env-file .env up -d

# 3. Run migrations and bootstrap storage
make migrate
make init-minio

# 4. Ingest a ticker (downloads SEC filings → indexes → XBRL → Postgres)
make ingest ticker=NVDA
make index ticker=NVDA

# 5. Ask a question
make graph-run q="How is NVDA's data-center revenue concentration evolving?"
```

Open `http://localhost:5173` for the live dashboard.
Langfuse traces: `http://localhost:3000`.

---

## Architecture

**Supervisor–Worker** agentic graph (LangGraph) with 6 specialized agents:

| Agent | Role | LLM |
|---|---|---|
| Supervisor / Planner | Decomposes queries, routes, decides done | Gemini 2.0 Flash |
| Retrieval / RAG | Hybrid BM25 + dense + cross-encoder rerank | — (local models) |
| EDGAR Parser | Live 10-K/10-Q/8-K ingestion + XBRL extraction | Groq Llama 3.3 |
| Web Search | Open-web evidence via Tavily | Groq Llama 3.3 |
| SQL Generator | NL→SQL over Postgres XBRL warehouse | Gemini 2.0 Flash |
| Summarizer / Writer | Token-streaming draft with inline citations | Gemini 2.0 Flash |
| Critic / Verifier | NLI claim-grounding, triggers self-correction | bge-reranker (local) |

Self-correction loop: Critic issues `revise` → Supervisor re-routes failing
sub-tasks only (not full restart) → up to `max_iterations` (default 3).

```
React Dashboard (React 18 + Vite + React Flow)
        │ WebSocket (agent events) + REST
FastAPI Gateway + ARQ Worker
        │
LangGraph StateGraph ─► Qdrant · Postgres+pgvector · Redis · MinIO
        │
Gemini Flash · Groq Llama 3.3 · Cerebras (fallback)
Tavily Search · SEC EDGAR API
bge-large-en-v1.5 (embeddings) · bge-reranker-v2-m3 (rerank)
nli-deberta-v3-base (critic)
```

---

## Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph, LangChain |
| LLMs | Gemini 2.0 Flash, Groq Llama 3.3 70B, Cerebras (fallback) |
| Embeddings | `bge-large-en-v1.5` (local, 1024-dim) |
| Reranker | `bge-reranker-v2-m3` (local, beats Cohere rerank-3) |
| NLI Critic | `cross-encoder/nli-deberta-v3-base` (local) |
| Vector DB | Qdrant (self-hosted) |
| Relational DB | Postgres 16 + pgvector |
| Cache / Queue | Redis 7 + ARQ |
| Object Store | MinIO |
| Backend | FastAPI + WebSockets (async) |
| Frontend | React 18 + TypeScript + Vite + React Flow + shadcn/ui |
| Observability | Langfuse v2 (self-hosted) |
| Eval | RAGAS + custom retrieval harness (50-question golden set) |
| Infra | Docker Compose (local) · HuggingFace Spaces + Vercel (cloud) |

All LLMs and APIs: **free tier, no credit card required.**

---

## Evaluation

Ablation matrix: `{BM25 | dense | hybrid} × {no rerank | bge-rerank} × {no critic | critic}` (12 cells) evaluated on 50 hand-authored SEC Q/A pairs spanning NVDA, AMD, and AAPL.

```bash
make eval-ablation      # 6-cell retrieval ablation + EVAL.md + plots
make eval-ragas         # RAGAS generation metrics (free LLM judge)
make eval               # full suite
```

Full results and methodology: [`docs/EVAL.md`](docs/EVAL.md).

---

## Stress Test

```bash
make stress-test             # 10 concurrent sessions (default)
make stress-test sessions=20 # configurable
```

Reports p50/p95/p99 latency and success rate. Exits non-zero if success rate
drops below 80%.

---

## Cloud Deploy

Deployable entirely on free-tier services, no credit card required:

| Component | Service |
|---|---|
| Backend API + Worker | HuggingFace Spaces (Docker) |
| Frontend | Vercel |
| Vector DB | Qdrant Cloud (1 GB free) |
| Relational DB | Supabase (500 MB free) |
| Redis | Upstash (10k cmd/day free) |

```bash
cp .env.cloud.example .env.cloud   # fill in cloud credentials
make cloud-up                      # docker compose -f infra/docker-compose.cloud.yml
```

See [`infra/huggingface/Dockerfile`](infra/huggingface/Dockerfile) and
[`apps/web/vercel.json`](apps/web/vercel.json) for details.

---

## Project Phases

| Phase | Focus | Status |
|---|---|---|
| 0 | Bootstrap & DevOps foundation | ✅ Complete |
| 1 | Ingestion pipelines (EDGAR + XBRL + PDF) | ✅ Complete |
| 2 | Retrieval stack (Qdrant + BM25 + reranker) | ✅ Complete |
| 3 | Single-agent RAG baseline | ✅ Complete |
| 4 | LangGraph multi-agent skeleton | ✅ Complete |
| 5 | Critic agent + self-correction loops | ✅ Complete |
| 6 | FastAPI + WebSocket streaming + ARQ | ✅ Complete |
| 7 | React dashboard (chat + DAG viz + citations) | ✅ Complete |
| 8 | Observability (Langfuse) + eval harness (RAGAS) + ablation | ✅ Complete |
| 9 | Stress test, hardening, cloud deploy, WRITEUP | ✅ Complete |

See [`docs/PLAN.md`](docs/PLAN.md) for the full build plan and
[`docs/WRITEUP.md`](docs/WRITEUP.md) for the project reflection and
failure-mode taxonomy.

---

## Key Make Targets

```bash
make up                         # start all Docker services
make down                       # stop services
make migrate                    # run Alembic migrations
make ingest ticker=NVDA         # download + parse SEC filings
make index ticker=NVDA          # index into Qdrant + BM25
make graph-run q="..."          # run multi-agent pipeline (CLI)
make worker                     # start ARQ task worker
make eval                       # full eval suite (ablation + RAGAS)
make stress-test sessions=10    # concurrent session benchmark
make cloud-up                   # start cloud-pointed services
make lint                       # ruff lint + format check
make typecheck                  # mypy + tsc
make test                       # pytest
make help                       # all targets with descriptions
```

---

## Documentation

| File | Contents |
|---|---|
| [`docs/PLAN.md`](docs/PLAN.md) | Master architecture plan |
| [`docs/EVAL.md`](docs/EVAL.md) | Retrieval ablation methodology + live results |
| [`docs/WRITEUP.md`](docs/WRITEUP.md) | Project reflection + failure-mode taxonomy |
| `docs/PHASE_N_SUMMARY.md` | Per-phase decision log + interview prep |
