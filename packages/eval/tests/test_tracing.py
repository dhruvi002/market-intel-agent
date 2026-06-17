"""Tests for mia_eval.tracing — Langfuse helpers degrade to no-ops when unset.

No real Langfuse server; settings are mocked and the lru_cache is cleared
between cases so each test sees a fresh handler decision.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import mia_eval.tracing as tracing


def _settings(public=None, secret=None):
    s = MagicMock()
    s.langfuse_public_key = public
    secret_mock = MagicMock()
    secret_mock.get_secret_value.return_value = secret
    s.langfuse_secret_key = secret_mock if secret else None
    s.langfuse_host = "http://localhost:3000"
    return s


def _clear_caches():
    tracing.get_langfuse_handler.cache_clear()
    tracing._get_client.cache_clear()


class TestLangfuseEnabled:
    def test_disabled_when_no_keys(self):
        with patch.object(tracing, "get_settings", return_value=_settings()):
            assert tracing.langfuse_enabled() is False

    def test_enabled_when_both_keys(self):
        with patch.object(
            tracing, "get_settings", return_value=_settings("pk", "sk")
        ):
            assert tracing.langfuse_enabled() is True

    def test_disabled_with_only_public_key(self):
        with patch.object(
            tracing, "get_settings", return_value=_settings("pk", None)
        ):
            assert tracing.langfuse_enabled() is False


class TestGetHandler:
    def setup_method(self):
        _clear_caches()

    def teardown_method(self):
        _clear_caches()

    def test_returns_none_when_disabled(self):
        with patch.object(tracing, "get_settings", return_value=_settings()):
            assert tracing.get_langfuse_handler() is None

    def test_langchain_callbacks_empty_when_disabled(self):
        with patch.object(tracing, "get_settings", return_value=_settings()):
            assert tracing.langchain_callbacks() == []


class TestObserveRun:
    def setup_method(self):
        _clear_caches()

    def teardown_method(self):
        _clear_caches()

    def test_yields_none_when_disabled(self):
        async def run():
            with patch.object(tracing, "get_settings", return_value=_settings()):
                async with tracing.observe_run("x") as trace:
                    return trace

        assert asyncio.run(run()) is None

    def test_flushes_client_on_exit_when_enabled(self):
        fake_client = MagicMock()
        fake_client.trace.return_value = "trace-obj"

        async def run():
            with patch.object(tracing, "_get_client", return_value=fake_client):
                async with tracing.observe_run("graph_run", session_id="s1") as trace:
                    assert trace == "trace-obj"

        asyncio.run(run())
        fake_client.trace.assert_called_once()
        fake_client.flush.assert_called_once()

    def test_trace_failure_does_not_break_run(self):
        fake_client = MagicMock()
        fake_client.trace.side_effect = RuntimeError("langfuse down")

        async def run():
            with patch.object(tracing, "_get_client", return_value=fake_client):
                async with tracing.observe_run("x") as trace:
                    return trace

        # Should swallow the trace() error, yield None, still flush.
        assert asyncio.run(run()) is None
        fake_client.flush.assert_called_once()
