"""RAGAS generation-quality evaluation, wired to the free LLM + bge stack.

RAGAS scores the *generation* half of the pipeline that the custom retrieval
metrics cannot see:

- **faithfulness**       — are the answer's claims grounded in the retrieved
                           context? (the quantitative twin of the NLI critic)
- **answer_relevancy**   — does the answer actually address the question?
- **context_precision**  — are the retrieved chunks that matter ranked highly?
- **context_recall**     — did retrieval surface everything the reference answer
                           needs? (requires a ground-truth reference)

Cost discipline ($0 budget)
---------------------------
RAGAS calls an LLM as judge.  We bind it to the **same free Gemini/Groq stack**
via ``get_llm`` (LangChain ``BaseChatModel``) and to the **local bge embedder**
so no paid OpenAI key is ever required.  All heavy imports (ragas, datasets,
langchain wrappers) are local to the functions so importing this module — and
collecting tests — never drags them in.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from mia_eval.golden import GoldenQA

logger = logging.getLogger(__name__)


def _patch_missing_vertexai() -> None:
    """Shim the dead Vertex import in ``ragas`` so it imports cleanly.

    ``ragas`` (0.4.x) hard-imports ``ChatVertexAI`` from
    ``langchain_community.chat_models.vertexai`` at module load, but that
    submodule was removed in ``langchain_community`` >= 0.4 (Vertex moved to the
    ``langchain-google-vertexai`` package). This project never uses Vertex (the
    LLM stack is Gemini / Groq / Cerebras), so we register a minimal placeholder
    module. ``ChatVertexAI`` is only referenced by ragas in ``isinstance`` type
    checks, which correctly evaluate ``False`` for our ChatOpenAI-based models.

    No-op if the real module is present (e.g. on a future compatible install).
    """
    import sys
    import types

    mod_name = "langchain_community.chat_models.vertexai"
    if mod_name in sys.modules:
        return
    try:
        __import__(mod_name)
        return  # real module exists — nothing to patch
    except ModuleNotFoundError:
        pass

    stub = types.ModuleType(mod_name)

    class ChatVertexAI:  # placeholder; never instantiated
        """Stub for the removed langchain_community Vertex chat model."""

    stub.ChatVertexAI = ChatVertexAI
    sys.modules[mod_name] = stub
    logger.debug("Patched missing %s with a stub for ragas compatibility", mod_name)


@dataclass(slots=True)
class RagasSample:
    """One row fed to RAGAS: question, generated answer, contexts, reference."""

    question: str
    answer: str
    contexts: list[str]
    reference: str = ""


@dataclass(slots=True)
class RagasScores:
    """Mean RAGAS scores across the evaluated samples."""

    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    n: int = 0
    per_metric_raw: dict[str, list[float]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float | int]:
        return {
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
            "n": self.n,
        }


def build_samples_from_responses(
    golden: Sequence[GoldenQA],
    answers: Sequence[str],
    contexts: Sequence[Sequence[str]],
) -> list[RagasSample]:
    """Zip golden questions with generated answers + retrieved contexts.

    Lengths must match; this is a pure helper so it is unit-testable without
    RAGAS installed.
    """
    if not (len(golden) == len(answers) == len(contexts)):
        raise ValueError(
            f"length mismatch: golden={len(golden)} answers={len(answers)} "
            f"contexts={len(contexts)}"
        )
    return [
        RagasSample(
            question=qa.question,
            answer=ans,
            contexts=list(ctx),
            reference=qa.answer,
        )
        for qa, ans, ctx in zip(golden, answers, contexts)
    ]


def _build_ragas_llm(llm: BaseChatModel | None):
    """Wrap our free LangChain chat model for RAGAS."""
    _patch_missing_vertexai()
    from langchain_core.language_models import BaseChatModel
    from ragas.llms import LangchainLLMWrapper

    if llm is None:
        from mia_agents.llm import get_llm

        llm = get_llm()  # full free fallback chain / LLM_PROVIDER pin
    assert isinstance(llm, BaseChatModel)

    # Give the judge headroom. RAGAS prompts emit long JSON, and reasoning-style
    # models (e.g. Cerebras zai-glm-4.7) can hit the default 4096-token cap
    # mid-output, which RAGAS raises as LLMDidNotFinishException. Bump the
    # completion budget wherever the model exposes it.
    for attr in ("max_tokens", "max_completion_tokens"):
        cur = getattr(llm, attr, None)
        if isinstance(cur, int) and cur < 8192:
            try:
                setattr(llm, attr, 8192)
            except Exception:
                pass

    # bypass_n=True → RAGAS issues n separate single-completion requests instead
    # of one request with n>1. Cerebras's OpenAI-compatible API rejects n>1
    # ("'n' > 1 is not currently supported"); this routes around it.
    return LangchainLLMWrapper(llm, bypass_n=True)


def _build_ragas_embeddings():
    """Wrap the project's bge embedder for RAGAS (answer-relevancy needs embeddings).

    Reuses ``mia_retrieval``'s ``Embedder`` (the same BAAI/bge-large-en-v1.5
    singleton the retriever uses) through a thin ``langchain_core.embeddings``
    adapter. This keeps the eval's embeddings identical to retrieval's and
    avoids depending on the ``langchain_huggingface`` integration package.
    """
    from langchain_core.embeddings import Embeddings
    from mia_retrieval.embedder import get_embedder
    from ragas.embeddings import LangchainEmbeddingsWrapper

    embedder = get_embedder()

    class _BGEEmbeddings(Embeddings):
        """Adapt mia_retrieval.Embedder to the langchain Embeddings interface."""

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [vec.tolist() for vec in embedder.embed(list(texts))]

        def embed_query(self, text: str) -> list[float]:
            return embedder.embed_query(text).tolist()

    return LangchainEmbeddingsWrapper(_BGEEmbeddings())


def evaluate_generation(
    samples: Sequence[RagasSample],
    *,
    llm: BaseChatModel | None = None,
    metrics: Sequence[str] | None = None,
    max_workers: int = 1,
) -> RagasScores:
    """Run RAGAS over *samples* and return mean scores.

    Parameters
    ----------
    samples     : list of :class:`RagasSample` (use
                  :func:`build_samples_from_responses`)
    llm         : LangChain chat model judge (defaults to the free fallback
                  chain / ``LLM_PROVIDER`` pin)
    metrics     : subset of
                  ``{"faithfulness", "answer_relevancy", "context_precision",
                  "context_recall"}``; defaults to all four
    max_workers : RAGAS judge concurrency. Defaults to **1** (serial) because
                  free-tier inference providers (e.g. Cerebras) share a request
                  queue and return ``429 queue_exceeded`` under bursty
                  concurrent load. Raise only on a paid/self-hosted endpoint.

    Returns
    -------
    RagasScores

    Notes
    -----
    Heavy imports are local. Requires ``ragas``, ``datasets`` and
    ``langchain-huggingface`` to be installed (declared in pyproject).
    """
    if not samples:
        return RagasScores()

    _patch_missing_vertexai()
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )
    from ragas.run_config import RunConfig

    metric_map = {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "context_recall": context_recall,
    }
    selected = list(metrics) if metrics else list(metric_map)
    chosen = [metric_map[m] for m in selected if m in metric_map]

    ds = Dataset.from_dict(
        {
            "question": [s.question for s in samples],
            "answer": [s.answer for s in samples],
            "contexts": [s.contexts for s in samples],
            "reference": [s.reference for s in samples],
        }
    )

    ragas_llm = _build_ragas_llm(llm)
    ragas_emb = _build_ragas_embeddings()

    # Serialize judge calls (max_workers=1) with a generous per-call timeout and
    # retry budget so transient free-tier 429s recover instead of failing the run.
    run_config = RunConfig(max_workers=max_workers, timeout=300, max_retries=10)

    logger.info(
        "Running RAGAS on %d samples, metrics=%s, max_workers=%d",
        len(samples), selected, max_workers,
    )
    result = evaluate(
        ds, metrics=chosen, llm=ragas_llm, embeddings=ragas_emb, run_config=run_config
    )

    df = result.to_pandas()
    scores = RagasScores(n=len(samples))
    for name in metric_map:
        if name in df.columns:
            col = df[name].astype(float)
            setattr(scores, name, float(col.mean()))
            scores.per_metric_raw[name] = [float(x) for x in col.tolist()]
    return scores
