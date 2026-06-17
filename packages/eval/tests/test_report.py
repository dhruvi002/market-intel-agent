"""Tests for mia_eval.report — markdown rendering + EVAL.md injection.

Plot functions (matplotlib/seaborn) are not exercised here; the pure string +
file-IO paths are.
"""

from __future__ import annotations

from mia_eval.ablation import AblationCell
from mia_eval.report import (
    lift_line,
    results_to_markdown,
    write_eval_report,
)
from mia_eval.report import _REPORT_BEGIN, _REPORT_END


def _cells():
    return [
        AblationCell("bm25", False, False, n=15, recall=0.4, precision=0.3, mrr=0.5, ndcg=0.50),
        AblationCell("hybrid", True, False, n=15, recall=0.7, precision=0.6, mrr=0.8, ndcg=0.64),
    ]


class TestMarkdown:
    def test_header_and_rows(self):
        md = results_to_markdown(_cells())
        assert "| Config |" in md
        assert "`bm25`" in md
        assert "`hybrid+rerank`" in md
        # 2 data rows + header + separator = 4 lines
        assert len(md.strip().splitlines()) == 4

    def test_none_metrics_render_dash(self):
        cell = AblationCell("bm25", False, False, n=1, pass_at_1=None)
        md = results_to_markdown([cell])
        assert "—" in md


class TestLiftLine:
    def test_includes_percentage(self):
        lift = {
            "relative_lift_pct": 28.0,
            "baseline_label": "bm25",
            "best_label": "hybrid+rerank",
            "metric": "ndcg",
        }
        line = lift_line(lift)
        assert "28.0%" in line and "hybrid+rerank" in line

    def test_appends_ci_when_given(self):
        from mia_eval.stats import MeanCI

        lift = {"relative_lift_pct": 28.0, "baseline_label": "b", "best_label": "x", "metric": "ndcg"}
        ci = MeanCI(mean=0.14, low=0.09, high=0.19, confidence=0.95, n=15)
        line = lift_line(lift, ci)
        assert "95% CI" in line and "0.090" in line


class TestWriteEvalReport:
    def test_creates_file_with_markers(self, tmp_path):
        path = tmp_path / "EVAL.md"
        write_eval_report(_cells(), path)
        text = path.read_text(encoding="utf-8")
        assert _REPORT_BEGIN in text and _REPORT_END in text
        assert "`hybrid+rerank`" in text

    def test_replaces_block_preserves_prose(self, tmp_path):
        path = tmp_path / "EVAL.md"
        path.write_text(
            f"# Methodology\n\nKeep me.\n\n{_REPORT_BEGIN}\nOLD\n{_REPORT_END}\n\nFooter.\n",
            encoding="utf-8",
        )
        write_eval_report(_cells(), path)
        text = path.read_text(encoding="utf-8")
        assert "Keep me." in text
        assert "Footer." in text
        assert "OLD" not in text
        assert text.count(_REPORT_BEGIN) == 1

    def test_appends_when_no_markers(self, tmp_path):
        path = tmp_path / "EVAL.md"
        path.write_text("# Existing prose\n", encoding="utf-8")
        write_eval_report(_cells(), path)
        text = path.read_text(encoding="utf-8")
        assert "# Existing prose" in text
        assert _REPORT_BEGIN in text
