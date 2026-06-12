# Phase 0 — Design Decisions & Interview Prep

> **Use this doc to:** challenge design decisions, prep interview answers, and restore context in a new Claude session.
> **Phase:** 0 — Bootstrap & DevOps Foundation
> **Status:** ✅ Complete and running locally

---

## What Phase 0 Built

The full DevOps skeleton before a single line of application logic:

- `infra/docker-compose.yml` — 6 infra services running locally
- `pyproject.toml` — uv workspace with 7 Python packages
- `pnpm-workspace.yaml` — pnpm workspace with the React frontend
- `apps/web/src/` — React 18 scaffolded with Zustand store, React Flow DAG, 4 UI components
- `packages/shared/` — `AgentState`, `Evidence`, `Citation`, `CritiqueResult` schemas + `Settings`
- `Dockerfile` — for api, worker, and web (dev + prod stages)
- `.pre-commit-config.yaml` — ruff, mypy, eslint, docker-compose-check
- `.github/workflows/ci.yml` — lint → test (with real Postgres + Redis) → web build → compose validate
- `Makefile` — `make up-infra`, `lint`, `test`, `migrate`, `dev-web`, etc.

---

## Decision Log — The "Why" Behind Every Choice

### 1. Why Docker Compose instead of running services locally?

**Short answer:** Reproducibility and isolation — `make up-infra` gives you a clean, identical environment every time, with no "works on my machine" drift between dev and eventual cloud deploy.

**Deeper:** Each service (Postgres, Qdrant, Redis, MinIO, Langfuse) has specific version and configuration requirements. Managing these natively on macOS means fighting Homebrew version conflicts and manual restarts. Docker Compose lets you declare the exact version, healthchecks, startup order, and networking in one file. When we deploy to HuggingFace Spaces in Phase 9, the same compose file (adapted) works.

**Interview follow-up — "Why not Kubernetes?"**
Overkill for a single-developer capstone. Kubernetes adds complexity (pods, services, ingress, RBAC) that doesn't pay off at this scale. Compose is the right tool for local-first development; we can graduate to K8s or managed services for cloud if needed.

---

### 2. Why uv instead of pip/conda/poetry?

**Short answer:** Speed and workspace support. `uv sync` is 10–100× faster than pip, and uv workspaces let each package declare only its own dependencies while sharing a single resolved lockfile.

**Deeper:** This project has 7 Python packages that depend on each other (e.g., `mia-worker` depends on `mia-ingestion`, `mia-retrieval`, and `mia-agents`). With pip you'd manage this manually or use editable installs. uv's workspace mode handles the dependency graph automatically and produces a single `uv.lock` that pins every transitive dep — identical installs across machines and in CI.

**Interview follow-up — "Why not Poetry?"**
Poetry workspaces are less mature and slower. uv is built in Rust, understands PEP 517/518/660, and is now the de-facto standard replacing pip in production Python projects.

---

### 3. Why split into 7 Python packages instead of one monolith?

**Short answer:** Separation of concerns and lean Docker images.

**Deeper:** Each package maps to a clear bounded context:
- `mia-shared` — schemas and settings only. Every other package imports this; keeping it small means fast imports.
- `mia-ingestion` — owns all EDGAR/PDF/XBRL I/O. Heavy deps (Docling, PyMuPDF, boto3) only installed where needed.
- `mia-retrieval` — owns embeddings and search. `torch` + `sentence-transformers` are large; we only want these in the worker image, not the API image.
- `mia-agents` — owns LangGraph. Isolated so we can swap the orchestration framework without touching ingestion.

The Dockerfiles exploit this: `apps/api/Dockerfile` installs only `mia-api` + its deps (no torch, no Docling); `apps/worker/Dockerfile` installs `mia-worker` which pulls in the heavy packages. This keeps the API image small (~300MB) and the worker image larger but justified.

**Interview follow-up — "Isn't this premature optimization?"**
The boundary lines will be crossed eventually — the API needs to call the agents package. But the physical separation enforces discipline: if you catch yourself importing `mia-ingestion` directly from `mia-api`, that's a smell that the API is doing ETL work it shouldn't be.

---

### 4. Why Qdrant over Pinecone / Weaviate / pgvector-only?

**Short answer:** Native hybrid search, no quota, no credit card, and better filtering than Pinecone's free tier.

**Deeper comparison:**

| | Qdrant (self-hosted) | Pinecone (cloud) | pgvector |
|---|---|---|---|
| Cost | Free | Free tier has pod limits | Free (already in Postgres) |
| Hybrid search | Native (sparse + dense in one query) | Requires separate sparse index | Manual BM25 + merge |
| Filtering | Rich payload filters, indexed | Limited on free tier | SQL WHERE clause |
| Setup | Docker container | API key + cloud | Extension in existing DB |
| Data sovereignty | All local | Data leaves machine | All local |

For the ablation study we're running `{BM25 | dense | hybrid}` retrieval modes. Qdrant's native sparse vector support makes the hybrid mode a first-class citizen rather than a hack. pgvector is kept as a backup dense store but Qdrant is the primary retrieval engine.

**Interview follow-up — "What's the downside of self-hosting Qdrant?"**
You own ops: backups, upgrades, disk management. For a capstone this is fine; for production you'd use Qdrant Cloud or a managed vector DB.

---

### 5. Why Langfuse v2 (not v3)?

**Short answer:** v3 mandates ClickHouse for analytics storage and an S3 bucket for event blobs — that's two more services to run locally just for observability. v2 only needs Postgres and Redis, both of which we already have.

**Deeper:** Langfuse v3 was designed for multi-tenant SaaS scale (think: hundreds of teams, billions of traces). The ClickHouse requirement exists because Postgres can't efficiently serve time-series aggregate queries at that scale. For a single-developer project generating at most a few thousand traces during eval, Postgres is more than sufficient. v2 gives us everything we need: per-trace span views, cost tracking, prompt versioning, evaluation scores.

**Interview follow-up — "Would you use v3 in production?"**
Yes — if you're running a production LLM product and need the analytics at scale. For this capstone, v2 is the right engineering call: minimum viable observability with minimum infrastructure overhead.

---

### 6. Why Gemini 2.0 Flash for Supervisor and Summarizer, Groq for workers?

**Short answer:** Gemini has a 1M-token context window and structured output — perfect for long-document synthesis and plan generation. Groq runs Llama 3.3 70B at 800 tokens/second — fast enough that agent round-trips don't feel slow during development.

**Deeper role assignment:**

| Agent | LLM | Why |
|---|---|---|
| Supervisor | Gemini 2.0 Flash | Needs to see the full query + all prior evidence to replan. 1M context means it never hits limits on a complex multi-hop query. |
| Web Search | Groq Llama 3.3 70B | Stateless, fast — just needs to pick search queries and summarize snippets. 800 tok/s keeps the loop tight. |
| EDGAR Parser | Groq | Parses structured sections (MD&A, Risk Factors). Fast turnaround matters; context needs are small. |
| Retrieval | Groq | Decides which chunks are relevant. Small output (a list of chunk IDs). Speed > context length. |
| SQL Generator | Gemini Flash | Structured output (valid SQL). Gemini's JSON mode is more reliable than Groq's for constrained generation. |
| Summarizer | Gemini 2.0 Flash | Writes the final brief from 20 evidence chunks — long input, long output. 1M context is essential here. |
| Critic | Groq Llama 3.3 70B | Outputs structured JSON (`{verdict, failing_claims}`). Groq's speed matters because Critic runs on every draft iteration. |

**Interview follow-up — "What happens when Groq rate-limits?"**
`Tenacity` handles retries with exponential backoff. Cerebras (also free, ~2000 tok/s on Llama 3.3 70B) is the configured fallback in `Settings`. The agent graph doesn't know which provider is serving — it just calls the LangChain `ChatGroq` or `ChatCerebras` interface.

---

### 7. Why ARQ over Celery for background tasks?

**Short answer:** ARQ is Redis-native and async-first. Celery has async support but it was bolted on; ARQ was designed for `asyncio` from the ground up.

**Deeper:** Our workers do a lot of I/O-bound work: downloading PDFs from EDGAR, uploading to MinIO, inserting into Postgres. `asyncio` lets a single worker process handle dozens of these concurrently without threads. Celery's default concurrency model is process-based (forks), which wastes memory for I/O-bound tasks. ARQ uses a simple Redis list as the queue — no broker complexity, and we already have Redis for caching and pubsub.

**Interview follow-up — "What's ARQ's weakness?"**
No built-in task routing to different queues by default (unlike Celery's routing_key system). For a single-queue setup like ours this is fine; if we needed to prioritize eval jobs over ingestion jobs we'd need to add queue naming.

---

### 8. Why Zustand over Redux for frontend state?

**Short answer:** The WebSocket event handler is a pure reducer (`handleEvent`) with zero boilerplate. Redux would add action creators, reducers, and a store configuration file for the same functionality.

**Deeper:** The session store has one complex job: receive `AgentEvent` objects from a WebSocket and update 7 pieces of state (draft, evidence list, agent statuses, critique, etc.). With Zustand this is a single `set()` call inside `handleEvent`. With Redux you'd define an action type, an action creator, a reducer case, and then `dispatch()` at the WebSocket callsite. The extra ceremony doesn't buy anything here.

TanStack Query handles server state (fetching sessions, documents) separately — Zustand is purely for the live WebSocket session state. This separation is the right call: server cache invalidation logic belongs in TanStack Query, not in a Zustand store.

**Interview follow-up — "Why not React Context?"**
Context re-renders the entire subtree on every state change. The WebSocket fires events rapidly during an agent run — using Context would cause constant re-renders of the entire app. Zustand uses shallow equality by default and only re-renders the specific components that subscribed to the changed slice of state.

---

### 9. Why React Flow for the agent DAG?

**Short answer:** It's the standard library for interactive node-edge graphs in React, and the nodes-light-up-on-execution demo is the single highest-impact visual in a 90-second interview demo.

**Deeper:** The DAG is static (fixed topology, fixed positions), so we don't use React Flow's interactive drag-and-drop features. What we use is its rendering engine (SVG-based, hardware accelerated), the `Handle` API for edge connections, and the ability to pass arbitrary `data` to each node. The `AgentNodeData.status` field drives the CSS class (`idle | active | done | error`) which changes the border color with a CSS transition — the "light up" effect is just a CSS variable change triggered by the Zustand store.

---

### 10. Why `AgentState` as a Pydantic model in `mia-shared`?

**Short answer:** Single source of truth. The Python backend (LangGraph), the API (FastAPI serialization), and the frontend (TypeScript `types/agent.ts`) all need to agree on the shape of agent state and events. Defining it once in `mia-shared` and mirroring it in TypeScript prevents silent drift.

**Deeper:** LangGraph requires the state to be a TypedDict or dataclass-like object. We use Pydantic because it gives us validation, serialization to JSON (for the WebSocket event bus), and IDE autocompletion everywhere. The TypeScript mirror in `src/types/agent.ts` is maintained manually — it's small enough (~80 lines) that the manual sync cost is lower than setting up a code generation pipeline (openapi-typescript or similar) in Phase 0.

---

## Gotchas Encountered in Phase 0

These are real problems you'll sound credible talking about:

**1. uv workspace packages need explicit `[tool.uv.sources]`**
Simply listing `mia-shared` in a package's `dependencies` isn't enough — uv requires `mia-shared = { workspace = true }` in `[tool.uv.sources]`. Without it, uv tries to find `mia-shared` on PyPI and fails. This is intentional: it prevents accidental shadowing of a real PyPI package by a local workspace package.

**2. `python-edgar` version — latest is 3.x, not 4.x**
The package hasn't had a v4 release. Specifying `>=4.0` caused the entire uv solve to fail. Lesson: always check PyPI for the actual latest version before writing version bounds.

**3. Langfuse v3 breaks without ClickHouse**
The `langfuse/langfuse:3` image now validates for `CLICKHOUSE_URL`, `CLICKHOUSE_USER`, `CLICKHOUSE_PASSWORD`, and `LANGFUSE_S3_EVENT_UPLOAD_BUCKET` at startup and crashes with a ZodError if they're missing. This isn't documented prominently. Solution: downgrade to v2, which only needs Postgres + Redis.

**4. Port conflicts with other Docker projects**
Qdrant's default ports (6333 HTTP, 6334 gRPC) were occupied by another project (`medcomply-qdrant`). Never kill another project's containers — instead parameterize ports with env vars and set different values in `.env`. Both ports need env vars; hardcoding the gRPC port in compose caused a second bind failure.

**5. pnpm build scripts blocked by default**
pnpm v9+ blocks postinstall scripts by default for security. `esbuild` (a Vite dependency) needs a postinstall script to download its native binary. Fix: `pnpm approve-builds` to interactively approve it, which writes `allowBuilds: esbuild: true` to `pnpm-workspace.yaml`.

---

## How to Use This Doc for Interview Prep

**Pattern for answering design questions:**
1. One-sentence answer (the choice)
2. Two-sentence "why" (the problem it solves)
3. One-sentence trade-off (what you gave up)

**Example — "Why Qdrant?"**
> "We self-host Qdrant because it's the only free vector DB with native hybrid sparse+dense search — which we need for the BM25/dense/hybrid ablation study. Pinecone's free tier limits pods and doesn't support sparse vectors natively; pgvector would require implementing BM25 separately and merging results manually. The trade-off is that we own ops: backups and upgrades are our responsibility."

**Red flags to avoid:**
- "I chose X because it was popular" → say what problem it solves
- "I chose X because I wanted to learn it" → say what it does better than the alternative
- Mentioning tools without being able to explain when you'd NOT use them

**Questions to be ready for:**
- "Walk me through what happens when a user submits a query" (end-to-end, Phase 4–7 when done)
- "How does your Critic agent detect hallucinations?" (NLI entailment score — Phase 5)
- "How did you quantify the 28% retrieval improvement?" (ablation matrix — Phase 8)
- "Why not just use a single GPT-4 call with a big context?" (XBRL parsing requires structured ETL, no grounding check, no auditable trace, can't stream agent reasoning)
- "What would you change if you had a budget?" (Pinecone for managed vector DB, Langfuse cloud for managed observability, a proper CI runner with GPU for embedding model tests)
