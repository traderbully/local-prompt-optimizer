"""Cross-model comparison reporting for Strategy B and C runs.

Writes:
  - ``tasks/<name>/comparison/summary.json`` — machine-readable.
  - ``tasks/<name>/comparison/report.md`` — human-readable.

The layout is identical across strategies so downstream consumers (UI,
MCP ``lpo_get_comparison``) don't branch on strategy.
"""

from __future__ import annotations

from pathlib import Path

from lpo.core.history import atomic_write_json, atomic_write_text
from lpo.core.multi_engine import MultiResult


def comparison_dir(task_root: Path) -> Path:
    return task_root / "comparison"


def write_comparison_report(task_root: Path, task_name: str, result: MultiResult) -> tuple[Path, Path]:
    """Write ``summary.json`` and ``report.md``. Returns their paths."""
    comp = comparison_dir(task_root)
    comp.mkdir(parents=True, exist_ok=True)

    summary = {
        "task_name": task_name,
        "strategy": result.strategy,
        "total_cost_usd": round(result.total_cost_usd, 6),
        "per_model": [
            {
                "slug": r.slug,
                "model_id": r.model_id,
                "best_score": round(r.best_score, 2),
                "iterations": r.iterations,
                "stop_reason": r.stop_reason.value,
                "cost_usd": round(r.cost_usd, 6),
            }
            for r in result.per_model
        ],
        "winner_recommendation": _pick_winner(result),
    }
    if result.strategy == "unified_portable":
        summary["shared_best_score"] = (
            round(result.shared_best_score, 2) if result.shared_best_score is not None else None
        )
        summary["shared_stop_reason"] = (
            result.shared_stop_reason.value if result.shared_stop_reason is not None else None
        )

    summary_path = comp / "summary.json"
    atomic_write_json(summary_path, summary)

    report_path = comp / "report.md"
    atomic_write_text(report_path, _render_report(task_name, result, summary["winner_recommendation"]))
    return summary_path, report_path


def _pick_winner(result: MultiResult) -> dict[str, str | float]:
    """Return the recommended winner.

    Strategy B: the per-model run with the highest best_score.
    Strategy C: the shared prompt is the winner by construction; we surface
    the model with the highest score at that shared prompt for reference.
    """
    if not result.per_model:
        return {"slug": "", "best_score": 0.0, "reason": "no runs completed"}
    best = max(result.per_model, key=lambda r: r.best_score)
    if result.strategy == "unified_portable":
        return {
            "slug": "(shared prompt)",
            "best_score": round(result.shared_best_score or 0.0, 2),
            "reason": "Strategy C uses one shared prompt for all targets",
            "strongest_target": best.slug,
            "strongest_target_score": round(best.best_score, 2),
        }
    return {
        "slug": best.slug,
        "best_score": round(best.best_score, 2),
        "reason": "highest best_score across independent runs",
    }


def _render_report(task_name: str, result: MultiResult, winner: dict) -> str:
    header = f"# Cross-model comparison — {task_name}\n\n"
    header += f"- **Strategy:** `{result.strategy}`\n"
    header += f"- **Targets:** {len(result.per_model)}\n"
    header += f"- **Total cost:** ${result.total_cost_usd:.4f}\n\n"

    # Scores table.
    rows = []
    for r in sorted(result.per_model, key=lambda x: x.best_score, reverse=True):
        rows.append(
            f"| `{r.slug}` | `{r.model_id}` | {r.best_score:.2f} | {r.iterations} "
            f"| `{r.stop_reason.value}` | ${r.cost_usd:.4f} |"
        )
    table = (
        "## Per-model results\n\n"
        "| Slug | Model | Best score | Iterations | Stop reason | Cost |\n"
        "|---|---|---|---|---|---|\n"
        + "\n".join(rows)
        + "\n\n"
    )

    winner_block = "## Winner\n\n"
    if result.strategy == "parallel_independent":
        winner_block += (
            f"- **{winner['slug']}** with best_score={winner['best_score']:.2f}\n"
            f"  ({winner['reason']})\n\n"
        )
    else:  # unified_portable
        winner_block += (
            f"- **One shared prompt** — combined score {winner['best_score']:.2f}.\n"
            f"  Strongest target on this prompt: **{winner.get('strongest_target', '?')}** "
            f"({winner.get('strongest_target_score', 0):.2f}).\n\n"
        )

    prompt_block = ""
    if result.strategy == "unified_portable" and result.shared_prompt is not None:
        prompt_block = (
            "## Shared winning prompt\n\n"
            f"```\n{result.shared_prompt.rstrip()}\n```\n\n"
        )
    else:
        prompt_block += "## Winning prompts (per model)\n\n"
        for r in sorted(result.per_model, key=lambda x: x.best_score, reverse=True):
            prompt_block += f"### `{r.slug}`  (score {r.best_score:.2f})\n\n"
            prompt_block += f"```\n{r.best_prompt.rstrip()}\n```\n\n"

    return header + table + winner_block + prompt_block
