FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1

RUN pip install uv --no-cache-dir

WORKDIR /app

# ── Copy workspace manifests first (layer-cache friendly) ────────────────────
COPY pyproject.toml ./
COPY packages/shared/pyproject.toml packages/shared/
COPY packages/agents/pyproject.toml  packages/agents/
COPY apps/api/pyproject.toml         apps/api/

# ── Install only what api + its deps need ────────────────────────────────────
RUN uv sync --no-dev --package mia-api

# ── Copy source ──────────────────────────────────────────────────────────────
COPY packages/shared/src packages/shared/src
COPY packages/agents/src  packages/agents/src
COPY apps/api/src         apps/api/src

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "mia_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
