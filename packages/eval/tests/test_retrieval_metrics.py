"""Tests for mia_eval.retrieval_metrics.

Primitive metrics are checked against hand-computed values; the async
``evaluate_retrieval`` runner is exercised with a mocked Retriever.
"""

from __future__ import annotations

import asyncio
import math
from unittest.mock import AsyncMock

import pytest

from mia_eval.golden import GoldenQA, QuestionType
from mia_eval.retrieval_metrics import (
    RetrievalMetrics,
    evaluate_retrieval,
    evidence_doc_id,
    ndcg_at_k,
    precision_at_k,
    reciprocal_rank,
    recall_at_k,
    score_ranking,
)
from mia_shared.schemas import Evidence


# ── recall@k ─────────────────────────────────────────────────────────────────

class TestRecall:
    def test_perfect_recall(self):
        assert recall_at_k(["a", "b", "c"], {"a", "b"}, k=3) == 1.0

    def test_partial_recall(self):
        assert recall_at_k(["a", "x", "y"], {"a", "b"}, k=3) == 0.5

    def test_cutoff_excludes_late_hit(self):
        # relevant 'b' is at rank 3 but k=2 → not counted
        assert recall_at_k(["x", "a", "b"], {"a", "b"}, k=2) == 0.5

    def test_empty_relevant_is_zero(self):
        assert recall_at_k(["a"], set(), k=1) == 0.0


# ── precision@k ──────────────────────────────────────────────────────────────

class TestPrecision:
    def test_all_relevant(self):
        assert precision_at_k(["a", "b"], {"a", "b"}, k=2) == 1.0

    def test_half_relevant(self):
        assert precision_at_k(["a", "x"], {"a"}, k=2) == 0.5

    def test_fewer_results_than_k(self):
        # only 1 retrieved, k=5 → divide by retrieved count, not k
        assert precision_at_k(["a"], {"a"}, k=5) == 1.0

    def test_k_zero(self):
        assert precision_at_k(["a"], {"a"}, k=0) == 0.0

    def test_empty_ranking(self):
        assert precision_at_k([], {"a"}, k=5) == 0.0


# ── MRR ──────────────────────────────────────────────────────────────────────

class TestReciprocalRank:
    def test_first_position(self):
        assert reciprocal_rank(["a", "b"], {"a"}) == 1.0

    def test_third_position(self):
        assert reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)

    def test_no_hit(self):
        assert reciprocal_rank(["x", "y"], {"a"}) == 0.0


# ── nDCG@k ───────────────────────────────────────────────────────────────────

class TestNDCG:
    def test_ideal_ranking_is_one(self):
        assert ndcg_at_k(["a", "b"], {"a", "b"}, k=2) == pytest.approx(1.0)

    def test_reversed_from_ideal_known_value(self):
        # one relevant doc at rank 2: DCG = 1/log2(3); IDCG = 1/log2(2)=1
        expected = (1 / math.log2(3)) / 1.0
        assert ndcg_at_k(["x", "a"], {"a"}, k=2) == pytest.approx(expected)

    def test_empty_relevant_is_zero(self):
        assert ndcg_at_k(["a"], set(), k=1) == 0.0

    def test_partial_ideal_normalisation(self):
        # 2 relevant docs, both retrieved but at ranks 1 and 3, k=3
        # DCG = 1/log2(2) + 1/log2(4) = 1 + 0.5 = 1.5
        # IDCG = 1/log2(2) + 1/log2(3) = 1 + 0.6309 = 1.6309
        dcg = 1 + 0.5
        idcg = 1 + 1 / math.log2(3)
        assert ndcg_at_k(["a", "x", "b"], {"a", "b"}, k=3) == pytest.approx(dcg / idcg)


# ── score_ranking bundle ─────────────────────────────────────────────────────

class TestScoreRanking:
    def test_bundle_fields(self):
        m = score_ranking(["a", "x", "b"], {"a", "b"}, k=3)
        assert isinstance(m, RetrievalMetrics)
        assert m.k == 3
        assert m.num_relevant == 2
        assert m.num_retrieved == 3
        assert m.recall == 1.0
        assert m.mrr == 1.0

    def test_as_dict_roundtrip(self):
        m = score_ranking(["a"], {"a"}, k=1)
        d = m.as_dict()
        assert d["recall"] == 1.0 and d["k"] == 1


# ── evidence_doc_id ──────────────────────────────────────────────────────────

class TestEvidenceDocId:
    def test_prefers_doc_id_metadata(self):
        ev = Evidence(source_type="rag_chunk", text="t", metadata={"doc_id": "DOC1"})
        assert evidence_doc_id(ev) == "DOC1"

    def test_falls_back_to_chunk_id(self):
        ev = Evidence(source_type="rag_chunk", text="t", metadata={"chunk_id": "C9"})
        assert evidence_doc_id(ev) == "C9"

    def test_falls_back_to_uuid(self):
        ev = Evidence(source_type="rag_chunk", text="t")
        assert evidence_doc_id(ev) == str(ev.id)


# ── evaluate_retrieval (mocked Retriever) ────────────────────────────────────

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


class TestEvaluateRetrieval:
    def test_aggregates_over_golden(self):
        golden = [_qa("q1", ["a", "b"]), _qa("q2", ["c"])]

        retriever = AsyncMock()
        # q1 gets a perfect ranking; q2 misses entirely (differentiate by call order).
        retriever.retrieve = AsyncMock(side_effect=[[_ev("a"), _ev("b")], [_ev("z")]])

        result = asyncio.run(
            evaluate_retrieval(retriever, golden, mode="hybrid", rerank=True, k=10)
        )
        assert len(result.per_query) == 2
        assert result.per_query[0].recall == 1.0
        assert result.per_query[1].recall == 0.0
        assert result.mean_recall == 0.5
        assert result.summary()["mode"] == "hybrid"

    def test_passes_mode_and_rerank_through(self):
        golden = [_qa("q1", ["a"])]
        retriever = AsyncMock()
        retriever.retrieve = AsyncMock(return_value=[_ev("a")])

        asyncio.run(
            evaluate_retrieval(retriever, golden, mode="bm25", rerank=False, k=5)
        )
        _, kwargs = retriever.retrieve.call_args
        assert kwargs["rerank"] is False
        assert kwargs["top_k"] == 5
        assert kwargs["mode"].value == "bm25"
