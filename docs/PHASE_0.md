# Phase 0 — Bootstrap & DevOps Foundation

**Status:** ✅ Complete
**Duration:** ~1 session
**Unlocks:** Phase 1 (ingestion pipelines)

---

## What Was Built

### Infrastructure (`infra/`)

| File | Purpose |
|---|---|
| `docker-compose.yml` | All 8 services with healthchecks, named volumes, env-var wiring |
| `init/postgres/00_init.sql` | Enables `vector`, `pg_trgm`, `btree_gin`; creates `mia`, `xbrl`, `eval` schemas |

**Services in Compose:**

| Service | Image | Port | Notes |
|---|---|---|---|
| `postgres` | `pgvector/pgvector:pg16` | 5432 | pgvector enabled via init SQL |
| `redis` | `redis:7-alpine` | 6379 | auth-gated, AOF persistence |
| `qdrant` | `qdrant/qdrant:latest` | 6333/6334 | gRPC on 6334 |
| `minio` | `minio/minio:latest` | 9000/9001 | console on 9001 |
| `langfuse-db` | `postgres:16-alpine` | — | internal only |
| `langfuse-worker` | `langfuse/langfuse-worker:3` | — | telemetry disabled |
| `langfuse-web` | `langfuse/langfuse:3` | 3000 | self-hosted LLM observability |
| `api` | local build | 8000 | FastAPI + uvicorn |
| `worker` | local build | — | ARQ task worker |
| `web` | local build | 5173 (dev) | Vite dev server with HMR |

### Python Workspace (`pyproject.toml` + `uv`)

uv workspace with 7 members:

```
packages/
  shared/    → mia-shared    (Pydantic schemas, Settings, AgentState)
  agents/    → mia-agents    (LangGraph orchestration — Phase 4)
  retrieval/ → mia-retrieval (BM25 + dense + reranker — Phase 2)
  ingestion/ → mia-ingestion (EDGAR + XBRL + PDF — Phase 1)
  eval/      → mia-eval      (RAGAS harness — Phase 8)
apps/
  api/       → mia-api       (FastAPI gateway — Phase 6)
  worker/    → mia-worker    (ARQ workers — Phase 6)
```

Root `pyproject.toml` also configures: ruff (line-length 100, UP/B/SIM/RUF rules), mypy (strict, pydantic plugin), pytest (asyncio_mode=auto, coverage).

**`packages/shared` has real code now:**
- `config.py` — `Settings` via pydantic-settings, `get_settings()` cached singleton
- `schemas.py` — `AgentState`, `Evidence`, `Citation`, `CritiqueResult`, `AgentEvent`, `QueryRequest`, all enums

### Frontend Workspace (`pnpm-workspace.yaml`)

React 18 + TypeScript + Vite app scaffolded at `apps/web/`:

| File | Purpose |
|---|---|
| `vite.config.ts` | `/api` and `/ws` proxied to FastAPI in dev |
| `tailwind.config.ts` | CSS variables for agent node states (idle/active/done/error) |
| `src/types/agent.ts` | TypeScript mirror of `schemas.py` |
| `src/store/sessionStore.ts` | Zustand store — handles all WebSocket events, builds agent status map |
| `src/components/AgentDAG.tsx` | React Flow DAG, nodes light up as agents activate |
| `src/components/DraftViewer.tsx` | Streaming markdown with live cursor + Critic verdict badge |
| `src/components/EvidencePanel.tsx` | Scrollable list of gathered evidence chunks with source links |
| `src/components/QueryPanel.tsx` | Query textarea, submit/reset, example queries |

### DevOps

| File | Purpose |
|---|---|
| `.pre-commit-config.yaml` | ruff lint+format, mypy (shared), eslint, docker-compose-check |
| `.github/workflows/ci.yml` | Python lint → pytest (with Postgres + Redis services) → web typecheck+build → compose validate |
| `.env.example` | All env vars documented with comments and free-tier signup links |
| `Makefile` | `make up`, `down`, `logs`, `migrate`, `lint`, `test`, `format`, `typecheck`, `dev-web`, `clean`, `help` |
| `.gitignore` | Python caches, `.env`, `data/`, `node_modules/`, model weights |

---

## How to Start

```bash
# 1. Copy and fill in API keys
cp .env.example .env
# Edit .env: add GEMINI_API_KEY, GROQ_API_KEY, TAVILY_API_KEY

# 2. Install deps
make install

# 3. Start infra services
make up-infra

# 4. Install pre-commit hooks
uv run pre-commit install

# 5. Verify everything is running
make ps
make logs svc=postgres
```

After Langfuse boots at `http://localhost:3000`, create an account, create a project, then copy the public/secret keys into `.env` as `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY`.

---

## Key Decisions

**Why `pgvector/pgvector:pg16` instead of plain `postgres:16`?**
The `pgvector` image ships with the extension pre-installed; the init SQL just does `CREATE EXTENSION IF NOT EXISTS vector`. Avoids a custom Dockerfile for Postgres.

**Why Langfuse v3 self-hosted?**
Free, runs locally, no data leaves the machine. The v3 images split into `langfuse-worker` (background jobs) and `langfuse-web` (Next.js UI) — both needed.

**Why `uv sync --package <name>` in Dockerfiles?**
uv workspace-aware installs: only the specified package and its transitive deps get installed, keeping each image lean. The worker image installs `mia-worker` which transitively pulls ingestion + retrieval.

**Why Zustand over Redux/Context for the WebSocket store?**
Single-file store with no boilerplate. The `handleEvent` action is a pure reducer that the WebSocket `onmessage` calls directly — no dispatch overhead, easy to test.

---

## What Phase 1 Needs From Here

- `make up-infra` running cleanly → Postgres + MinIO + Redis available
- `mia-shared` importable → ingestion package can `from mia_shared.config import get_settings`
- MinIO bucket creation script (add to Phase 1 worker startup)
- Alembic initialized in `apps/api/` (first migration: empty, just proves the setup)

---

## Open Items (carry into Phase 1)

- [ ] `uv.lock` — run `uv lock` locally and commit
- [ ] Alembic init: `uv run alembic -c apps/api/alembic.ini init` → commit initial migration
- [ ] MinIO bucket bootstrap script (`scripts/init_minio.py`)
- [ ] Flip repo to public once Phase 1 has substantive code committed
