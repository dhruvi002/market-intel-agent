"""Tests for mia_eval.ablation — matrix enumeration + runner (mocked Retriever)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from mia_eval.ablation import (
    AblationCell,
    ablation_grid,
    baseline_vs_best,
    run_ablation,
)
from mia_eval.golden import GoldenQA, QuestionType
from mia_shared.schemas import Evidence


def _qa(qid: str, relevant: list[str]) -> GoldenQA:
    return GoldenQA(
        id=qid,
        question="valid question text",
        answer="a",
        question_type=QuestionType.SINGLE_HOP,
        relevant_doc_ids=relevant,
    )


def _ev(doc_id: str) -> Evidence:
    return Evidence(source_type="rag_chunk", text="t", metadata={"doc_id": doc_id})


# ── grid enumeration ─────────────────────────────────────────────────────────

class TestAblationGrid:
    def test_full_grid_is_12_cells(self):
        grid = ablation_grid(with_critic=True)
        assert len(grid) == 12
        assert len(set(grid)) == 12

    def test_no_critic_is_6_cells(self):
        grid = ablation_grid(with_critic=False)
        assert len(grid) == 6
        assert all(critic is False for _, _, critic in grid)

    def test_covers_all_modes(self):
        modes = {m for m, _, _ in ablation_grid()}
        assert modes == {"bm25", "dense", "hybrid"}


# ── AblationCell ─────────────────────────────────────────────────────────────

class TestAblationCell:
    def test_label_composition(self):
        assert AblationCell("hybrid", True, True).label == "hybrid+rerank+critic"
        assert AblationCell("bm25", False, False).label == "bm25"
        assert AblationCell("dense", True, False).label == "dense+rerank"

    def test_as_row_drops_raw_arrays(self):
        cell = AblationCell("bm25", False, False, ndcg_raw=[0.1], precision_raw=[0.2])
        row = cell.as_row()
        assert "ndcg_raw" not in row and "precision_raw" not in row
        assert row["label"] == "bm25"


# ── run_ablation (mocked retriever) ──────────────────────────────────────────

class TestRunAblation:
    def test_retrieval_only_six_cells_and_cache(self):
        golden = [_qa("q1", ["a"])]
        retriever = AsyncMock()
        retriever.retrieve = AsyncMock(return_value=[_ev("a")])

        cells = asyncio.run(run_ablation(retriever, golden, k=10, with_critic=False))
        assert len(cells) == 6
        # 6 cells but retrieval depends only on (mode, rerank) = 6 unique combos,
        # and with_critic=False so each is called exactly once per question.
        assert retriever.retrieve.await_count == 6

    def test_critic_dimension_reuses_retrieval_cache(self):
        golden = [_qa("q1", ["a"])]
        retriever = AsyncMock()
        retriever.retrieve = AsyncMock(return_value=[_ev("a")])

        cells = asyncio.run(run_ablation(retriever, golden, k=10, with_critic=True))
        assert len(cells) == 12
        # 12 cells but only 6 unique (mode, rerank) keys → 6 retrieval passes,
        # not 12 (the critic dimension does not re-run retrieval).
        assert retriever.retrieve.await_count == 6

    def test_e2e_runner_populates_pass_at_1(self):
        golden = [_qa("q1", ["a"])]
        retriever = AsyncMock()
        retriever.retrieve = AsyncMock(return_value=[_ev("a")])

        async def fake_e2e(g, mode, rerank, critic):
            return (0.9 if critic else 0.7, 1.5)

        cells = asyncio.run(
            run_ablation(
                retriever, golden, with_critic=True, with_e2e=True, e2e_runner=fake_e2e
            )
        )
        critic_cells = [c for c in cells if c.critic]
        assert all(c.pass_at_1 == 0.9 for c in critic_cells)
        assert all(c.mean_iterations == 1.5 for c in cells)


# ── baseline_vs_best ─────────────────────────────────────────────────────────

class TestBaselineVsBest:
    def test_lift_computation(self):
        cells = [
            AblationCell("bm25", False, False, ndcg=0.50, ndcg_raw=[0.5]),
            AblationCell("hybrid", True, False, ndcg=0.64, ndcg_raw=[0.64]),
        ]
        lift = baseline_vs_best(cells, metric="ndcg")
        assert lift["baseline_label"] == "bm25"
        assert lift["best_label"] == "hybrid+rerank"
        assert lift["absolute_lift"] == pytest.approx(0.14)
        assert lift["relative_lift_pct"] == pytest.approx(28.0)

    def test_missing_baseline_raises(self):
        cells = [AblationCell("hybrid", True, False, ndcg=0.6)]
        with pytest.raises(ValueError, match="baseline"):
            baseline_vs_best(cells, metric="ndcg", baseline_mode="bm25")

    def test_ignores_critic_cells_for_retrieval_metric(self):
        cells = [
            AblationCell("bm25", False, False, ndcg=0.5, ndcg_raw=[0.5]),
            AblationCell("bm25", False, True, ndcg=0.5, ndcg_raw=[0.5]),
            AblationCell("hybrid", True, False, ndcg=0.6, ndcg_raw=[0.6]),
            AblationCell("hybrid", True, True, ndcg=0.6, ndcg_raw=[0.6]),
        ]
        lift = baseline_vs_best(cells, metric="ndcg")
        assert lift["best_label"] == "hybrid+rerank"  # not the +critic variant
