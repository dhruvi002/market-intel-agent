"""Tests for mia_eval.golden — golden Q/A schema + JSONL loader.

No network, no models — pure data validation.
"""

from __future__ import annotations

import pytest

from mia_eval.golden import (
    DEFAULT_GOLDEN_PATH,
    GoldenQA,
    QuestionType,
    load_golden_set,
)


# ── Packaged golden set ─────────────────────────────────────────────────────────

class TestPackagedGoldenSet:
    def test_default_set_loads(self):
        items = load_golden_set()
        assert len(items) >= 12
        assert all(isinstance(q, GoldenQA) for q in items)

    def test_ids_unique(self):
        items = load_golden_set()
        ids = [q.id for q in items]
        assert len(ids) == len(set(ids))

    def test_all_four_question_types_present(self):
        items = load_golden_set()
        types = {q.question_type for q in items}
        assert types == set(QuestionType)

    def test_most_questions_have_relevant_docs(self):
        """43/50 questions map to >=1 relevant text chunk; the remaining 7 target
        XBRL-only figures with no matching text span (see docs/EVAL.md). The
        golden set is intentionally aligned this way, so we assert the bulk are
        covered rather than every single one."""
        items = load_golden_set()
        with_docs = [q for q in items if q.relevant_doc_ids]
        assert len(with_docs) >= 40

    def test_default_path_points_at_packaged_file(self):
        assert DEFAULT_GOLDEN_PATH.name == "golden_set.jsonl"
        assert DEFAULT_GOLDEN_PATH.exists()


# ── Model behaviour ──────────────────────────────────────────────────────────

class TestGoldenQAModel:
    def test_relevant_set_property(self):
        qa = GoldenQA(
            id="x1",
            question="What is NVDA revenue?",
            answer="$60.9B",
            question_type=QuestionType.QUANTITATIVE,
            relevant_doc_ids=["a", "b", "a"],
        )
        assert qa.relevant_set == {"a", "b"}

    def test_short_question_rejected(self):
        with pytest.raises(Exception):
            GoldenQA(
                id="x2",
                question="hi",  # < 5 chars
                answer="...",
                question_type=QuestionType.SINGLE_HOP,
            )

    def test_unknown_question_type_rejected(self):
        with pytest.raises(Exception):
            GoldenQA.model_validate(
                {
                    "id": "x3",
                    "question": "valid question text",
                    "answer": "a",
                    "question_type": "bogus_type",
                }
            )

    def test_defaults(self):
        qa = GoldenQA(
            id="x4",
            question="valid question text",
            answer="a",
            question_type=QuestionType.MULTI_HOP,
        )
        assert qa.tickers == []
        assert qa.relevant_doc_ids == []
        assert qa.relevant_spans == []


# ── Loader edge cases ────────────────────────────────────────────────────────

class TestLoader:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_golden_set(tmp_path / "nope.jsonl")

    def test_comments_and_blanks_ignored(self, tmp_path):
        p = tmp_path / "g.jsonl"
        p.write_text(
            "# a comment\n"
            "\n"
            '{"id":"q1","question":"valid question","answer":"a",'
            '"question_type":"single_hop","relevant_doc_ids":["d1"]}\n',
            encoding="utf-8",
        )
        items = load_golden_set(p)
        assert len(items) == 1
        assert items[0].id == "q1"

    def test_duplicate_id_raises(self, tmp_path):
        p = tmp_path / "g.jsonl"
        line = (
            '{"id":"dup","question":"valid question","answer":"a",'
            '"question_type":"single_hop"}\n'
        )
        p.write_text(line + line, encoding="utf-8")
        with pytest.raises(ValueError, match="duplicate"):
            load_golden_set(p)

    def test_empty_file_raises(self, tmp_path):
        p = tmp_path / "g.jsonl"
        p.write_text("# only comments\n\n", encoding="utf-8")
        with pytest.raises(ValueError, match="empty"):
            load_golden_set(p)

    def test_malformed_json_raises(self, tmp_path):
        p = tmp_path / "g.jsonl"
        p.write_text("{not valid json\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_golden_set(p)
