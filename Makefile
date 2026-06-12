.PHONY: up down logs ps build pull migrate lint test typecheck install clean help \
        query query-bm25 query-dense query-groq

COMPOSE = docker compose -f infra/docker-compose.yml --env-file .env
UV      = uv run
PNPM    = pnpm

# ── Docker Compose ───────────────────────────────────────────────────────────
up: ## Start all services (detached)
	$(COMPOSE) up -d

up-infra: ## Start only infra services (postgres, redis, qdrant, minio, langfuse)
	$(COMPOSE) up -d postgres redis qdrant minio langfuse-db langfuse-web

down: ## Stop all services
	$(COMPOSE) down

down-v: ## Stop all services and remove volumes (destructive!)
	$(COMPOSE) down -v

logs: ## Tail logs for all services (or pass svc=<name>)
	$(COMPOSE) logs -f $(svc)

ps: ## Show running containers
	$(COMPOSE) ps

build: ## Build app images
	$(COMPOSE) build api worker web

pull: ## Pull latest base images
	$(COMPOSE) pull

restart: ## Restart a service: make restart svc=api
	$(COMPOSE) restart $(svc)

# ── Database ─────────────────────────────────────────────────────────────────
migrate: ## Run Alembic migrations
	$(UV) alembic -c apps/api/alembic.ini upgrade head

migrate-new: ## Create a new migration: make migrate-new msg="add users table"
	$(UV) alembic -c apps/api/alembic.ini revision --autogenerate -m "$(msg)"

migrate-history: ## Show migration history
	$(UV) alembic -c apps/api/alembic.ini history --verbose

migrate-current: ## Show current migration revision
	$(UV) alembic -c apps/api/alembic.ini current

# ── Ingestion ─────────────────────────────────────────────────────────────────
init-minio: ## Bootstrap MinIO bucket (run once after make up-infra)
	$(UV) python scripts/init_minio.py

ingest: ## Ingest filings for a ticker: make ingest ticker=NVDA
	$(UV) python scripts/ingest_ticker.py $(ticker)

# ── Retrieval ──────────────────────────────────────────────────────────────────
index: ## Index a ticker into Qdrant + BM25: make index ticker=NVDA
	$(UV) python scripts/index_ticker.py $(ticker)

index-force: ## Re-index (overwrite existing): make index-force ticker=NVDA
	$(UV) python scripts/index_ticker.py $(ticker) --force

retrieve: ## Test retrieval: make retrieve query="NVDA revenue growth"
	$(UV) python scripts/retrieve.py "$(query)"

retrieve-bm25: ## BM25-only retrieval: make retrieve-bm25 query="data center"
	$(UV) python scripts/retrieve.py "$(query)" --mode bm25

worker: ## Start the ARQ task worker
	$(UV) python -m mia_worker.main

# ── Phase 3: RAG query ────────────────────────────────────────────────────────
query: ## Run RAG agent (hybrid): make query q="NVDA data center revenue?"
	$(UV) python scripts/query.py "$(q)"

query-bm25: ## BM25-only RAG: make query-bm25 q="NVDA risk factors"
	$(UV) python scripts/query.py "$(q)" --mode bm25

query-dense: ## Dense-only RAG: make query-dense q="AMD vs NVDA margins"
	$(UV) python scripts/query.py "$(q)" --mode dense

query-groq: ## RAG with Groq Llama: make query-groq q="NVDA revenue?"
	$(UV) python scripts/query.py "$(q)" --provider groq

# ── Phase 4: Multi-agent graph ────────────────────────────────────────────────
graph-run: ## Run multi-agent graph: make graph-run q="NVDA data center revenue?"
	$(UV) python scripts/graph_run.py "$(q)"

graph-run-groq: ## Graph with Groq Llama: make graph-run-groq q="AMD margins?"
	$(UV) python scripts/graph_run.py "$(q)" --provider groq

# ── Python dev ───────────────────────────────────────────────────────────────
install: ## Install all Python + Node deps
	uv sync --dev
	$(PNPM) install

lint: ## Ruff lint + format check
	$(UV) ruff check packages/ apps/api/ apps/worker/
	$(UV) ruff format --check packages/ apps/api/ apps/worker/

format: ## Auto-fix lint + format
	$(UV) ruff check --fix packages/ apps/api/ apps/worker/
	$(UV) ruff format packages/ apps/api/ apps/worker/

typecheck: ## mypy + tsc
	$(UV) mypy packages/shared/src/
	$(PNPM) typecheck

test: ## Run pytest
	$(UV) pytest -q

test-cov: ## Run pytest with HTML coverage
	$(UV) pytest --cov=packages --cov=apps --cov-report=html

# ── Frontend dev ─────────────────────────────────────────────────────────────
dev-web: ## Run Vite dev server
	cd apps/web && $(PNPM) dev

build-web: ## Production build of the frontend
	$(PNPM) build

# ── Housekeeping ─────────────────────────────────────────────────────────────
clean: ## Remove Python caches and build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'
