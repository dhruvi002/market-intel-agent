# Autonomous Enterprise Market Intelligence Agent

A production-grade multi-agent system that answers natural-language questions
about public companies by autonomously gathering evidence from SEC filings,
financial databases, and open-web sources — streaming every agent's reasoning
step live to a React dashboard.

**Demo:** _<!-- add Loom URL here -->_

---

## Quick Start

This is a complete bootstrap from a fresh clone. The vector index, XBRL warehouse,
and object store live in Docker volumes and a gitignored `data/` dir — **none of
that is in git** — so a fresh checkout (or a new machine) must re-run ingestion in
step 4. Your `.env` is also gitignored, so recreate it in step 1.

### 1. Clone and configure

```bash
git clone https://github.com/dhruvi002/market-intel-agent.git
cd market-intel-agent
cp .env.example .env
```

Fill in `.env`:
- `TAVILY_API_KEY` — https://app.tavily.com (web-search agent)
- At least one LLM key — `GEMINI_API_KEY`, `GROQ_API_KEY`, and/or `CEREBRAS_API_KEY`
- `LLM_PROVIDER` — leave blank for the Gemini→Groq→Cerebras fallback chain, or pin
  to one provider (e.g. `cerebras`) if the others' free quotas are exhausted.
  See [LLM configuration](#llm-configuration) below.

Install Python deps (uv) — frontend deps are already pinned via pnpm:

```bash
uv sync
```

### 2. Start infrastructure (Docker)

Start **infra only** — Postgres, Redis, Qdrant, MinIO, Langfuse. The app
(API / worker / web) runs locally in step 3, not in Docker:

```bash
make up-infra
make migrate        # Alembic schema
make init-minio     # create the filings bucket (run once)
```

### 3. Run the app — three local processes

Run each in its own terminal tab (all three stay running):

```bash
# Tab 1 — API gateway
uv run uvicorn mia_api.main:app --host 0.0.0.0 --port 8000   # wait for "Application startup complete."

# Tab 2 — ARQ worker
make worker                                                  # wait for "Worker starting queues=['default']"

# Tab 3 — frontend (Vite dev server)
make dev-web                                                 # serves http://localhost:5173
```

> **Why local, not the bundled `web` container?** The dockerized frontend's Vite
> proxy targets `localhost:8000` *inside its own container*, so it cannot reach a
> host-side API. Running the frontend with `make dev-web` keeps the proxy on the
> host where it can reach the local API. (If you prefer all-Docker, point the web
> service's `VITE_API_URL` at `http://host.docker.internal:8000`.)

### 4. Ingest data, then ask

```bash
# Index the three tickers the golden set / demo use (repeat per ticker)
make ingest ticker=NVDA && make index ticker=NVDA
make ingest ticker=AMD  && make index ticker=AMD
make ingest ticker=AAPL && make index ticker=AAPL
```

Open `http://localhost:5173`, ask a question about an **indexed** ticker
(NVDA / AMD / AAPL), e.g. *"How is NVDA's data-center revenue concentration
evolving vs AMD's?"*, and watch the agent DAG stream live. Langfuse traces:
`http://localhost:3000`.

CLI alternative (no frontend): `make graph-run q="How is NVDA's data-center revenue concentration evolving?"`

---

## LLM configuration

Every agent resolves its model through a single factory (`mia_agents.llm.get_llm`):

- **Default (no `LLM_PROVIDER`)** — a fallback chain: Gemini 2.0 Flash → Groq
  Llama 3.3 70B → Cerebras (if a key is set). LangChain retries the next provider
  on rate-limit/error.
- **Pinned (`LLM_PROVIDER=cerebras|groq|gemini`)** — all calls use that provider.
  Useful when free daily/token quotas on the others are spent.

Free-tier model availability shifts over time. Cerebras currently serves
`gpt-oss-120b` and `zai-glm-4.7` (the code defaults to `zai-glm-4.7` — faster, no
long reasoning trace); confirm what your key can access with
`curl -s https://api.cerebras.ai/v1/models -H "Authorization: Bearer $CEREBRAS_API_KEY"`.
Note free-tier providers share a request queue, so concurrent throughput is
capped (see [`docs/WRITEUP.md`](docs/WRITEUP.md) §7).

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

The **LLM** column lists the default model per role; all LLM calls route through
one configurable provider/fallback chain — see [LLM configuration](#llm-configuration).

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

# Sequential per-session latency distribution (no queue contention):
uv run python scripts/latency_bench.py --n 5
```

`stress-test` reports p50/p95/p99 latency and success rate (exits non-zero below
80%); `latency_bench` measures genuine end-to-end latency one session at a time.
On free-tier LLMs, concurrent throughput is bounded by the provider's shared
request queue, not the app — see [`docs/WRITEUP.md`](docs/WRITEUP.md) §7.

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


## Key Make Targets

```bash
make up-infra                   # start infra only (Postgres/Redis/Qdrant/MinIO/Langfuse)
make up                         # start ALL Docker services (infra + app images)
make down                       # stop services
make migrate                    # run Alembic migrations
make init-minio                 # create the MinIO filings bucket (run once)
make ingest ticker=NVDA         # download + parse SEC filings
make index ticker=NVDA          # index into Qdrant + BM25
make worker                     # start ARQ task worker (local)
make dev-web                    # start the Vite frontend dev server (local)
make graph-run q="..."          # run multi-agent pipeline (CLI, no frontend)
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
