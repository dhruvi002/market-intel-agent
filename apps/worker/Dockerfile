FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1

# System deps for Docling / PyMuPDF (libGL etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv --no-cache-dir

WORKDIR /app

COPY pyproject.toml ./
COPY packages/shared/pyproject.toml    packages/shared/
COPY packages/agents/pyproject.toml    packages/agents/
COPY packages/ingestion/pyproject.toml packages/ingestion/
COPY packages/retrieval/pyproject.toml packages/retrieval/
COPY apps/worker/pyproject.toml        apps/worker/

RUN uv sync --no-dev --package mia-worker

COPY packages/shared/src    packages/shared/src
COPY packages/agents/src    packages/agents/src
COPY packages/ingestion/src packages/ingestion/src
COPY packages/retrieval/src packages/retrieval/src
COPY apps/worker/src        apps/worker/src

CMD ["uv", "run", "arq", "mia_worker.main.WorkerSettings"]
