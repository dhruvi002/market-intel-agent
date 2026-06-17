"""Golden Q/A set: schema + loader.

The golden set is the ground truth that everything in Phase 8 is measured
against.  Each entry pairs a natural-language question with:

- a reference answer (used by RAGAS faithfulness / answer-relevancy),
- the set of *relevant document ids* (used by the custom retrieval metrics —
  Recall@k / MRR / nDCG), and
- optional exact text spans (the literal sentences that ground the answer,
  used for span-level checks and for explaining failures in the writeup).

Design decisions
----------------
- **JSONL on disk, Pydantic in memory.**  JSONL is diff-friendly (one Q per
  line), trivially appendable, and survives merge conflicts far better than a
  single big JSON array.  Pydantic validation catches typos (e.g. an unknown
  ``question_type``) at load time rather than mid-eval.
- ``relevant_doc_ids`` is the **chunk / document identifier** as it appears in
  ``Evidence.metadata["doc_id"]`` (falling back to ``Evidence.id``).  Retrieval
  metrics compare the ranked evidence's ids against this set.
- ``QuestionType`` keeps the four capstone categories (single-hop, multi-hop,
  comparative, quantitative) typed so per-category breakdowns are cheap.
- No LLM, no network — pure data.  Safe to import in fast unit tests.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

# Resolve to packages/eval/src/mia_eval/data/golden_set.jsonl regardless of cwd.
DEFAULT_GOLDEN_PATH = Path(__file__).resolve().parent / "data" / "golden_set.jsonl"


class QuestionType(str, Enum):
    """The four capstone question categories."""

    SINGLE_HOP = "single_hop"
    MULTI_HOP = "multi_hop"
    COMPARATIVE = "comparative"
    QUANTITATIVE = "quantitative"


class GoldenQA(BaseModel):
    """One hand-authored question with ground-truth annotations."""

    id: str = Field(..., description="Stable identifier, e.g. 'q001'")
    question: str = Field(..., min_length=5)
    answer: str = Field(..., description="Reference answer for generation eval")
    question_type: QuestionType
    tickers: list[str] = Field(default_factory=list, description="In-scope tickers")
    relevant_doc_ids: list[str] = Field(
        default_factory=list,
        description="Ground-truth chunk/document ids that should be retrieved",
    )
    relevant_spans: list[str] = Field(
        default_factory=list,
        description="Exact sentences that ground the answer (optional)",
    )

    @property
    def relevant_set(self) -> set[str]:
        """Relevant doc ids as a set (the form retrieval metrics consume)."""
        return set(self.relevant_doc_ids)


def load_golden_set(path: Path | str | None = None) -> list[GoldenQA]:
    """Load and validate the golden Q/A set from a JSONL file.

    Blank lines and lines beginning with ``#`` are ignored, so the file can
    carry section comments.  Raises on the first malformed record (fail fast —
    a silently-dropped golden question corrupts every downstream metric).

    Parameters
    ----------
    path : JSONL path; defaults to the packaged ``data/golden_set.jsonl``.

    Returns
    -------
    list[GoldenQA]
    """
    p = Path(path) if path is not None else DEFAULT_GOLDEN_PATH
    if not p.exists():
        raise FileNotFoundError(f"Golden set not found at {p}")

    items: list[GoldenQA] = []
    seen_ids: set[str] = set()
    with p.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:  # pragma: no cover - message path
                raise ValueError(f"{p}:{lineno} — invalid JSON: {exc}") from exc
            qa = GoldenQA.model_validate(record)
            if qa.id in seen_ids:
                raise ValueError(f"{p}:{lineno} — duplicate golden id {qa.id!r}")
            seen_ids.add(qa.id)
            items.append(qa)

    if not items:
        raise ValueError(f"Golden set at {p} is empty")
    return items
