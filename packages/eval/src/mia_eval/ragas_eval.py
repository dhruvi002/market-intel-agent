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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from mia_eval.golden import GoldenQA

logger = logging.getLogger(__name__)


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
    golden: "Sequence[GoldenQA]",
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


def _build_ragas_llm(llm: "BaseChatModel | None"):
    """Wrap our free LangChain chat model for RAGAS."""
    from langchain_core.language_models import BaseChatModel  # noqa: PLC0415
    from ragas.llms import LangchainLLMWrapper  # noqa: PLC0415

    if llm is None:
        from mia_agents.llm import get_llm  # noqa: PLC0415

        llm = get_llm()  # full free fallback chain
    assert isinstance(llm, BaseChatModel)
    return LangchainLLMWrapper(llm)


def _build_ragas_embeddings():
    """Wrap the local bge embedder for RAGAS (answer-relevancy needs embeddings)."""
    from langchain_huggingface import HuggingFaceEmbeddings  # noqa: PLC0415
    from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: PLC0415

    from mia_shared.config import get_settings  # noqa: PLC0415

    cfg = get_settings()
    hf = HuggingFaceEmbeddings(model_name=cfg.embedding_model)
    return LangchainEmbeddingsWrapper(hf)


def evaluate_generation(
    samples: Sequence[RagasSample],
    *,
    llm: "BaseChatModel | None" = None,
    metrics: Sequence[str] | None = None,
) -> RagasScores:
    """Run RAGAS over *samples* and return mean scores.

    Parameters
    ----------
    samples : list of :class:`RagasSample` (use
              :func:`build_samples_from_responses`)
    llm     : LangChain chat model judge (defaults to the free fallback chain)
    metrics : subset of
              ``{"faithfulness", "answer_relevancy", "context_precision",
              "context_recall"}``; defaults to all four

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

    from datasets import Dataset  # noqa: PLC0415
    from ragas import evaluate  # noqa: PLC0415
    from ragas.metrics import (  # noqa: PLC0415
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

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

    logger.info("Running RAGAS on %d samples, metrics=%s", len(samples), selected)
    result = evaluate(
        ds, metrics=chosen, llm=ragas_llm, embeddings=ragas_emb
    )

    df = result.to_pandas()
    scores = RagasScores(n=len(samples))
    for name in metric_map:
        if name in df.columns:
            col = df[name].astype(float)
            setattr(scores, name, float(col.mean()))
            scores.per_metric_raw[name] = [float(x) for x in col.tolist()]
    return scores
