# HuggingFace Spaces Dockerfile — MIA backend (API + worker in one container)
#
# HF Spaces requirements:
#   • Must listen on port 7860
#   • USER must be 1000 (non-root)
#   • Image built from this file in the repo root (Spaces copies the repo)
#
# Deploy steps:
#   1. Create a new HF Space: type=Docker, SDK=Docker, hardware=CPU Basic (free)
#   2. In Space Settings → Repository: link your GitHub repo
#   3. Set Space secrets (same keys as .env.cloud.example)
#   4. HF builds and runs this Dockerfile automatically on push
#
# The container runs the FastAPI gateway on port 7860 and spins up the ARQ
# worker in a background thread using the same process — acceptable for a demo
# instance; split into separate Spaces for production load.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    PORT=7860

# System deps for Docling / PyMuPDF
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv --no-cache-dir

# HF Spaces requires a non-root user with uid 1000
RUN useradd -m -u 1000 mia
WORKDIR /app
RUN chown mia:mia /app
USER mia

# ── Copy workspace manifests ─────────────────────────────────────────────────
COPY --chown=mia pyproject.toml ./
COPY --chown=mia packages/shared/pyproject.toml    packages/shared/
COPY --chown=mia packages/agents/pyproject.toml    packages/agents/
COPY --chown=mia packages/ingestion/pyproject.toml packages/ingestion/
COPY --chown=mia packages/retrieval/pyproject.toml packages/retrieval/
COPY --chown=mia apps/api/pyproject.toml           apps/api/
COPY --chown=mia apps/worker/pyproject.toml        apps/worker/

# Install both api and worker deps
RUN uv sync --no-dev --package mia-api --package mia-worker

# ── Copy source ──────────────────────────────────────────────────────────────
COPY --chown=mia packages/shared/src    packages/shared/src
COPY --chown=mia packages/agents/src    packages/agents/src
COPY --chown=mia packages/ingestion/src packages/ingestion/src
COPY --chown=mia packages/retrieval/src packages/retrieval/src
COPY --chown=mia apps/api/src           apps/api/src
COPY --chown=mia apps/worker/src        apps/worker/src

# ── Entrypoint script ────────────────────────────────────────────────────────
COPY --chown=mia infra/huggingface/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -sf http://localhost:7860/api/health || exit 1

CMD ["/app/entrypoint.sh"]
