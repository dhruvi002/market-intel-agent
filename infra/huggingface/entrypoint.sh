#!/bin/bash
# HuggingFace Spaces entrypoint — starts API on port 7860 and ARQ worker.
#
# The ARQ worker runs as a background process in the same container.
# This is a demo convenience; a production deploy would use separate containers.

set -euo pipefail

echo "Starting MIA ARQ worker (background)..."
uv run arq mia_worker.main.WorkerSettings &
WORKER_PID=$!

echo "Starting MIA FastAPI server on port ${PORT:-7860}..."
exec uv run uvicorn mia_api.main:app \
    --host 0.0.0.0 \
    --port "${PORT:-7860}" \
    --workers 1 \
    --log-level info
