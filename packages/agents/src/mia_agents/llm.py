"""LLM provider factory with multi-provider fallback chain.

Priority order (all free-tier, no credit card required):
  1. Gemini 2.0 Flash   — Google AI Studio (15 RPM, 1M TPM, 1500 RPD, 1M-token context)
  2. Groq Llama 3.3 70B — Groq (fast inference, ~800 tok/s)
  3. Cerebras Llama 3.3  — Cerebras (~2000 tok/s, fallback if Groq throttles)

Usage::

    from mia_agents.llm import get_llm

    # Full fallback chain (recommended for production paths)
    llm = get_llm()

    # Pin a specific provider (e.g., for evals or cost control)
    llm = get_llm(provider="groq")

Design decisions:
- Returns a standard ``BaseChatModel``, so callers are provider-agnostic.
- ``with_fallbacks()`` is a LangChain primitive — it catches ``Exception``
  (including rate-limit errors) and transparently retries on the next provider.
- Cerebras uses OpenAI-compatible API, handled via ChatOpenAI with a custom
  base_url — no separate Cerebras SDK needed.
- Temperature defaults to 0.0 for RAG: we want deterministic extraction, not
  creative generation. Override for brainstorming agents.
"""

from __future__ import annotations

import logging
from enum import Enum

from langchain_core.language_models import BaseChatModel

from mia_shared.config import get_settings

logger = logging.getLogger(__name__)

_GEMINI_MODEL = "gemini-2.0-flash"
_GROQ_MODEL = "llama-3.3-70b-versatile"
_CEREBRAS_MODEL = "llama3.3-70b"
_CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"


class LLMProvider(str, Enum):
    GEMINI = "gemini"
    GROQ = "groq"
    CEREBRAS = "cerebras"


def get_llm(
    provider: LLMProvider | str | None = None,
    *,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> BaseChatModel:
    """Return a LangChain ``BaseChatModel``, optionally with a fallback chain.

    Parameters
    ----------
    provider    : pin a specific provider (``None`` → full fallback chain)
    temperature : sampling temperature (default 0.0 — deterministic RAG)
    max_tokens  : maximum output tokens

    Returns
    -------
    BaseChatModel
        Ready to call with ``.invoke()`` / ``.ainvoke()``.
    """
    settings = get_settings()

    if provider is not None:
        p = LLMProvider(provider)
        logger.debug("LLM: pinned to %s", p.value)
        return _build_single(p, settings, temperature, max_tokens)

    # Build fallback chain: Gemini → Groq → (Cerebras if key present)
    primary = _build_single(LLMProvider.GEMINI, settings, temperature, max_tokens)
    fallbacks: list[BaseChatModel] = [
        _build_single(LLMProvider.GROQ, settings, temperature, max_tokens)
    ]
    if settings.cerebras_api_key is not None:
        fallbacks.append(
            _build_single(LLMProvider.CEREBRAS, settings, temperature, max_tokens)
        )

    logger.debug(
        "LLM: fallback chain — %s → %s",
        LLMProvider.GEMINI.value,
        " → ".join(p.value for p in [LLMProvider.GROQ, LLMProvider.CEREBRAS][: len(fallbacks)]),
    )
    return primary.with_fallbacks(fallbacks)


# ── Internal builders ─────────────────────────────────────────────────────────

def _build_single(
    provider: LLMProvider,
    settings,
    temperature: float,
    max_tokens: int,
) -> BaseChatModel:
    """Instantiate a single LangChain model for *provider*."""
    if provider == LLMProvider.GEMINI:
        from langchain_google_genai import ChatGoogleGenerativeAI  # noqa: PLC0415

        return ChatGoogleGenerativeAI(
            model=_GEMINI_MODEL,
            google_api_key=settings.gemini_api_key.get_secret_value(),
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

    if provider == LLMProvider.GROQ:
        from langchain_groq import ChatGroq  # noqa: PLC0415

        return ChatGroq(
            model=_GROQ_MODEL,
            groq_api_key=settings.groq_api_key.get_secret_value(),
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if provider == LLMProvider.CEREBRAS:
        from langchain_openai import ChatOpenAI  # noqa: PLC0415

        if settings.cerebras_api_key is None:
            raise ValueError(
                "CEREBRAS_API_KEY is not set — cannot build Cerebras provider"
            )
        return ChatOpenAI(
            model=_CEREBRAS_MODEL,
            openai_api_key=settings.cerebras_api_key.get_secret_value(),
            openai_api_base=_CEREBRAS_BASE_URL,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    raise ValueError(f"Unknown LLM provider: {provider!r}")
