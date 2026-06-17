"""Reporting: ablation results → markdown tables + matplotlib/seaborn plots.

Turns the structured eval outputs into the artifacts the writeup and README
reference:

- ``results_to_markdown`` — a GitHub-flavoured markdown table of the ablation
  cells (pure string formatting, no deps — unit-testable).
- ``plot_retrieval_lift`` — bar chart of nDCG@k per retrieval config with the
  baseline highlighted (the "28% lift" visual).
- ``plot_ablation_heatmap`` — mode × (rerank/critic) heatmap of a chosen metric.
- ``write_eval_report`` — renders the markdown summary + lift line into
  ``docs/EVAL.md`` between sentinel markers, leaving the prose intact.

matplotlib / seaborn / pandas are imported lazily inside the plotting functions
so the markdown path needs none of them.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from mia_eval.ablation import AblationCell
    from mia_eval.stats import MeanCI

logger = logging.getLogger(__name__)

_REPORT_BEGIN = "<!-- EVAL:RESULTS:BEGIN -->"
_REPORT_END = "<!-- EVAL:RESULTS:END -->"


def _fmt(x: float | None, places: int = 3) -> str:
    return "—" if x is None else f"{x:.{places}f}"


def results_to_markdown(cells: "Sequence[AblationCell]") -> str:
    """Render ablation cells as a markdown table (no external deps)."""
    header = (
        "| Config | n | Recall@k | Precision@k | MRR | nDCG@k | Pass@1 | "
        "Mean iters |\n"
        "|---|---|---|---|---|---|---|---|"
    )
    rows = []
    for c in cells:
        rows.append(
            f"| `{c.label}` | {c.n} | {_fmt(c.recall)} | {_fmt(c.precision)} | "
            f"{_fmt(c.mrr)} | {_fmt(c.ndcg)} | {_fmt(c.pass_at_1)} | "
            f"{_fmt(c.mean_iterations, 2)} |"
        )
    return header + "\n" + "\n".join(rows)


def lift_line(lift: dict[str, object], ci: "MeanCI | None" = None) -> str:
    """One-line summary of the headline retrieval lift, with optional CI."""
    rel = lift.get("relative_lift_pct", 0.0)
    base_label = lift.get("baseline_label", "baseline")
    best_label = lift.get("best_label", "best")
    metric = lift.get("metric", "ndcg")
    ci_str = f" (95% CI on lift: [{ci.low:.3f}, {ci.high:.3f}])" if ci else ""
    return (
        f"**Headline:** `{best_label}` improves {metric} by "
        f"**{rel:.1f}%** over the `{base_label}` baseline"
        f"{ci_str}."
    )


def plot_retrieval_lift(
    cells: "Sequence[AblationCell]",
    out_path: Path | str,
    *,
    metric: str = "ndcg",
    baseline_mode: str = "bm25",
) -> Path:
    """Bar chart of *metric* per retrieval config; baseline highlighted."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")  # headless — no display needed in CI/eval
    import matplotlib.pyplot as plt  # noqa: PLC0415

    retrieval_cells = [c for c in cells if not c.critic] or list(cells)
    labels = [c.label.replace("+critic", "") for c in retrieval_cells]
    values = [getattr(c, metric) for c in retrieval_cells]
    colors = [
        "#888888" if (c.mode == baseline_mode and not c.rerank) else "#76b900"
        for c in retrieval_cells
    ]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, values, color=colors)
    ax.set_ylabel(f"{metric}@k")
    ax.set_title(f"Retrieval ablation — {metric}@k by configuration")
    ax.set_ylim(0, max(values) * 1.15 if values else 1)
    for i, v in enumerate(values):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Wrote retrieval-lift plot → %s", out)
    return out


def plot_ablation_heatmap(
    cells: "Sequence[AblationCell]",
    out_path: Path | str,
    *,
    metric: str = "ndcg",
) -> Path:
    """Heatmap of *metric*: rows = retrieval mode, cols = rerank×critic combo."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    import seaborn as sns  # noqa: PLC0415

    modes = sorted({c.mode for c in cells})
    combos = sorted({(c.rerank, c.critic) for c in cells})

    def combo_label(rerank: bool, critic: bool) -> str:
        parts = []
        parts.append("rerank" if rerank else "no-rerank")
        parts.append("critic" if critic else "no-critic")
        return "\n".join(parts)

    grid = np.full((len(modes), len(combos)), np.nan)
    lookup = {(c.mode, c.rerank, c.critic): getattr(c, metric) for c in cells}
    for i, mode in enumerate(modes):
        for j, (rr, cr) in enumerate(combos):
            if (mode, rr, cr) in lookup:
                grid[i, j] = lookup[(mode, rr, cr)]

    fig, ax = plt.subplots(figsize=(1.6 * len(combos) + 2, 1.2 * len(modes) + 2))
    sns.heatmap(
        grid,
        annot=True,
        fmt=".3f",
        cmap="YlGn",
        xticklabels=[combo_label(rr, cr) for rr, cr in combos],
        yticklabels=modes,
        cbar_kws={"label": f"{metric}@k"},
        ax=ax,
    )
    ax.set_title(f"Ablation heatmap — {metric}@k")
    plt.tight_layout()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Wrote ablation heatmap → %s", out)
    return out


def write_eval_report(
    cells: "Sequence[AblationCell]",
    eval_md_path: Path | str,
    *,
    lift: dict[str, object] | None = None,
    lift_ci: "MeanCI | None" = None,
) -> Path:
    """Inject the results table + lift line into ``docs/EVAL.md``.

    Replaces the content between the sentinel markers; if the markers are
    absent (or the file does not exist) the block is appended.  This keeps the
    hand-written methodology prose in EVAL.md intact across re-runs.
    """
    table = results_to_markdown(cells)
    block_parts = ["### Latest ablation run", ""]
    if lift is not None:
        block_parts += [lift_line(lift, lift_ci), ""]
    block_parts += [table, ""]
    block = (
        f"{_REPORT_BEGIN}\n" + "\n".join(block_parts) + f"{_REPORT_END}\n"
    )

    path = Path(eval_md_path)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""

    if _REPORT_BEGIN in existing and _REPORT_END in existing:
        pre = existing.split(_REPORT_BEGIN)[0]
        post = existing.split(_REPORT_END)[1]
        new_content = pre + block + post
    else:
        sep = "\n\n" if existing and not existing.endswith("\n\n") else ""
        new_content = existing + sep + block

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(new_content, encoding="utf-8")
    logger.info("Updated eval report → %s", path)
    return path
