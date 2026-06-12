"""Tests for mia_agents.llm — LLM factory.

All tests are unit tests: no real API calls.  We patch langchain_google_genai,
langchain_groq, and langchain_openai so the factory is testable without
credentials.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from mia_agents.llm import LLMProvider, _build_single, get_llm


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_settings(cerebras_key: str | None = "cerebras_key"):
    s = MagicMock()
    s.gemini_api_key.get_secret_value.return_value = "gemini_key"
    s.groq_api_key.get_secret_value.return_value = "groq_key"
    if cerebras_key is not None:
        s.cerebras_api_key.get_secret_value.return_value = cerebras_key
    else:
        s.cerebras_api_key = None
    return s


# ── LLMProvider enum ──────────────────────────────────────────────────────────

class TestLLMProvider:
    def test_values(self):
        assert LLMProvider.GEMINI == "gemini"
        assert LLMProvider.GROQ == "groq"
        assert LLMProvider.CEREBRAS == "cerebras"

    def test_from_string(self):
        assert LLMProvider("gemini") is LLMProvider.GEMINI
        assert LLMProvider("groq") is LLMProvider.GROQ

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            LLMProvider("openai")


# ── _build_single ─────────────────────────────────────────────────────────────

class TestBuildSingle:
    @patch("mia_agents.llm.ChatGoogleGenerativeAI", create=True)
    def test_gemini_uses_correct_model(self, mock_cls):
        from langchain_google_genai import ChatGoogleGenerativeAI  # noqa: F401
        with patch("langchain_google_genai.ChatGoogleGenerativeAI", mock_cls):
            settings = _make_settings()
            _build_single(LLMProvider.GEMINI, settings, temperature=0.0, max_tokens=4096)
        mock_cls.assert_called_once()
        kwargs = mock_cls.call_args.kwargs
        assert kwargs["model"] == "gemini-2.0-flash"
        assert kwargs["temperature"] == 0.0

    @patch("mia_agents.llm.ChatGroq", create=True)
    def test_groq_uses_correct_model(self, mock_cls):
        with patch("langchain_groq.ChatGroq", mock_cls):
            settings = _make_settings()
            _build_single(LLMProvider.GROQ, settings, temperature=0.0, max_tokens=512)
        mock_cls.assert_called_once()
        kwargs = mock_cls.call_args.kwargs
        assert "llama" in kwargs["model"].lower()
        assert kwargs["max_tokens"] == 512

    def test_unknown_provider_raises(self):
        settings = _make_settings()
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            _build_single("unknown_provider", settings, 0.0, 512)  # type: ignore[arg-type]

    def test_cerebras_missing_key_raises(self):
        settings = _make_settings(cerebras_key=None)
        with pytest.raises(ValueError, match="CEREBRAS_API_KEY"):
            _build_single(LLMProvider.CEREBRAS, settings, 0.0, 512)


# ── get_llm ───────────────────────────────────────────────────────────────────

class TestGetLlm:
    def test_returns_model_with_fallbacks(self):
        """get_llm() with no provider should produce a fallback-wrapped model."""
        mock_gemini = MagicMock()
        mock_groq = MagicMock()
        chained = MagicMock()
        mock_gemini.with_fallbacks.return_value = chained

        settings = _make_settings(cerebras_key=None)
        with (
            patch("mia_agents.llm.get_settings", return_value=settings),
            patch("mia_agents.llm._build_single") as mock_build,
        ):
            mock_build.side_effect = [mock_gemini, mock_groq]
            result = get_llm()

        assert result is chained
        mock_gemini.with_fallbacks.assert_called_once_with([mock_groq])

    def test_returns_model_with_cerebras_fallback(self):
        """When cerebras_api_key is set, chain includes Cerebras."""
        mock_gemini = MagicMock()
        mock_groq = MagicMock()
        mock_cerebras = MagicMock()
        chained = MagicMock()
        mock_gemini.with_fallbacks.return_value = chained

        settings = _make_settings(cerebras_key="ckey")
        with (
            patch("mia_agents.llm.get_settings", return_value=settings),
            patch("mia_agents.llm._build_single") as mock_build,
        ):
            mock_build.side_effect = [mock_gemini, mock_groq, mock_cerebras]
            result = get_llm()

        assert result is chained
        mock_gemini.with_fallbacks.assert_called_once_with([mock_groq, mock_cerebras])

    def test_pinned_provider_skips_fallbacks(self):
        """Pinning a provider should call _build_single once without with_fallbacks."""
        mock_groq = MagicMock()
        settings = _make_settings()
        with (
            patch("mia_agents.llm.get_settings", return_value=settings),
            patch("mia_agents.llm._build_single", return_value=mock_groq) as mock_build,
        ):
            result = get_llm(provider="groq")

        assert result is mock_groq
        mock_build.assert_called_once()
        mock_groq.with_fallbacks.assert_not_called()

    def test_temperature_forwarded(self):
        """temperature kwarg should be passed through to _build_single."""
        settings = _make_settings(cerebras_key=None)
        with (
            patch("mia_agents.llm.get_settings", return_value=settings),
            patch("mia_agents.llm._build_single") as mock_build,
        ):
            mock_build.return_value = MagicMock()
            get_llm(provider="gemini", temperature=0.7)

        _, call_kwargs = mock_build.call_args
        # temperature is a positional arg: (provider, settings, temperature, max_tokens)
        assert mock_build.call_args.args[2] == 0.7

    def test_max_tokens_forwarded(self):
        settings = _make_settings(cerebras_key=None)
        with (
            patch("mia_agents.llm.get_settings", return_value=settings),
            patch("mia_agents.llm._build_single") as mock_build,
        ):
            mock_build.return_value = MagicMock()
            get_llm(provider="groq", max_tokens=1024)

        assert mock_build.call_args.args[3] == 1024
