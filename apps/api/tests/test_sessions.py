"""Tests for apps/api/src/mia_api/main.py — sessions endpoint.

All external I/O is mocked:
  - Redis: ``AsyncMock`` for get/set operations (via patched ``_get_redis_pool``)
  - ARQ ``create_pool``: patched as ``AsyncMock`` so ``await create_pool(...)``
    returns our mock pool
  - Session helpers: ``create_session`` and ``get_session`` patched directly

Uses FastAPI's ``TestClient`` (sync ASGI wrapper) for REST endpoints.

Patch targets
-------------
  - ``mia_api.main.create_pool``          — ARQ pool (AsyncMock)
  - ``mia_api.main.create_session``       — session creation helper (AsyncMock)
  - ``mia_api.main.get_session``          — session lookup helper (AsyncMock)
  - ``mia_api.deps._get_redis_pool``      — shared Redis pool (returns mock)
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from mia_api.main import app


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_redis_mock() -> AsyncMock:
    """Shared async Redis client mock."""
    r = AsyncMock()
    r.set = AsyncMock(return_value=True)
    r.get = AsyncMock(return_value=json.dumps({"status": "queued", "query": "test"}))
    r.publish = AsyncMock(return_value=1)
    r.aclose = AsyncMock()
    return r


def _make_arq_pool_mock() -> AsyncMock:
    """ARQ pool mock — ``enqueue_job`` resolves to a fake job object."""
    pool = AsyncMock()
    pool.enqueue_job = AsyncMock(return_value=MagicMock(job_id="fake-job-id"))
    pool.aclose = AsyncMock()
    return pool


def _client_with_mocks(arq_pool=None, redis=None):
    """Return a ``TestClient`` with ARQ + Redis dependencies patched."""
    if arq_pool is None:
        arq_pool = _make_arq_pool_mock()
    if redis is None:
        redis = _make_redis_mock()
    return TestClient(app), arq_pool, redis


# ── /api/health ───────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_ok(self):
        client = TestClient(app)
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


# ── POST /api/sessions ────────────────────────────────────────────────────────

class TestCreateSession:
    def _post(self, query: str = "What is NVDA revenue for FY2024?", arq_pool=None, redis=None):
        """POST /api/sessions with all external deps patched."""
        if arq_pool is None:
            arq_pool = _make_arq_pool_mock()
        if redis is None:
            redis = _make_redis_mock()
        with (
            patch("mia_api.main.create_pool", new_callable=AsyncMock, return_value=arq_pool),
            patch("mia_api.main.create_session", new_callable=AsyncMock),
            patch("mia_api.deps._get_redis_pool", return_value=redis),
        ):
            client = TestClient(app)
            return client.post(
                "/api/sessions",
                json={"query": query, "stream": True},
            )

    def test_returns_200(self):
        resp = self._post()
        assert resp.status_code == 200

    def test_response_has_session_id(self):
        data = self._post().json()
        assert "session_id" in data

    def test_session_id_is_valid_uuid(self):
        session_id = self._post().json()["session_id"]
        UUID(session_id)  # raises if invalid

    def test_status_is_queued(self):
        assert self._post().json()["status"] == "queued"

    def test_ws_url_contains_session_id(self):
        data = self._post().json()
        assert data["session_id"] in data["ws_url"]
        assert "/ws/sessions/" in data["ws_url"]

    def test_arq_enqueue_called_once(self):
        arq_pool = _make_arq_pool_mock()
        self._post(arq_pool=arq_pool)
        arq_pool.enqueue_job.assert_called_once()

    def test_arq_enqueue_passes_run_query_and_query_text(self):
        arq_pool = _make_arq_pool_mock()
        query = "Compare MSFT and GOOGL cloud growth Q4 2024"
        self._post(query=query, arq_pool=arq_pool)

        args = arq_pool.enqueue_job.call_args[0]
        assert args[0] == "run_query"        # task name
        assert query in args                  # query string is passed

    def test_query_too_short_returns_422(self):
        # min_length=5 in QueryRequest
        with (
            patch("mia_api.main.create_pool", new_callable=AsyncMock),
            patch("mia_api.main.create_session", new_callable=AsyncMock),
            patch("mia_api.deps._get_redis_pool", return_value=_make_redis_mock()),
        ):
            client = TestClient(app)
            resp = client.post("/api/sessions", json={"query": "Hi", "stream": True})
        assert resp.status_code == 422

    def test_arq_unavailable_returns_503(self):
        with (
            patch("mia_api.main.create_pool", side_effect=ConnectionError("no Redis")),
            patch("mia_api.main.create_session", new_callable=AsyncMock),
            patch("mia_api.deps._get_redis_pool", return_value=_make_redis_mock()),
        ):
            client = TestClient(app)
            resp = client.post(
                "/api/sessions",
                json={"query": "NVDA data center revenue FY2025", "stream": True},
            )
        assert resp.status_code == 503


# ── GET /api/sessions/{id} ────────────────────────────────────────────────────

class TestGetSession:
    def test_returns_session_data(self):
        session_id = uuid4()
        session_data = {"status": "running", "query": "NVDA revenue"}

        with patch("mia_api.main.get_session", new_callable=AsyncMock, return_value=session_data):
            client = TestClient(app)
            resp = client.get(f"/api/sessions/{session_id}")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "running"
        assert body["session_id"] == str(session_id)

    def test_not_found_returns_404(self):
        session_id = uuid4()

        with patch("mia_api.main.get_session", new_callable=AsyncMock, return_value=None):
            client = TestClient(app)
            resp = client.get(f"/api/sessions/{session_id}")

        assert resp.status_code == 404
