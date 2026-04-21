r"""Re-score a stored iteration's outputs against the current scoring code
without regenerating anything from the model.

Usage (from repo root):

    python scripts\rescore_iteration.py <task_dir> <slug> <iter_index>

Example:

    python scripts\rescore_iteration.py tasks\windows_file_ops_reliability \
        gemma-4-26b-local 9

Loads:
  - <task_dir>/eval_set.jsonl, gold_standard.jsonl, metric.yaml
  - <task_dir>/runs/<slug>/history/iter_XXXX/outputs.jsonl  (raw model outputs)

Runs the current DeterministicScorer against the stored outputs and prints:
  - stored aggregate + stored per-example scores (the numbers the ratchet
    saw at the time)
  - re-scored aggregate + per-example scores (under the current scoring code)
  - per-example delta

This is the offline equivalent of re-running an iteration against a changed
metric — useful for validating a metric patch before committing to a full
re-run of the optimization loop.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

# Allow `python scripts/rescore_iteration.py ...` to import the lpo package
# without installing in editable mode.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lpo.core.task import TaskBundle  # noqa: E402
from lpo.scoring.aggregation import aggregate_scores  # noqa: E402
from lpo.scoring.base import ScoringContext  # noqa: E402
from lpo.scoring.factory import build_scorer  # noqa: E402


def _load_outputs(iter_dir: Path) -> tuple[dict[str, str], dict[str, float]]:
    """Return (id -> raw model output, id -> stored score) from an
    iteration's outputs.jsonl."""
    path = iter_dir / "outputs.jsonl"
    outputs: dict[str, str] = {}
    stored: dict[str, float] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            outputs[row["id"]] = row["output"]
            if "score" in row:
                stored[row["id"]] = float(row["score"])
    return outputs, stored


async def main() -> int:
    if len(sys.argv) != 4:
        print(__doc__)
        return 2
    task_dir = Path(sys.argv[1]).resolve()
    slug = sys.argv[2]
    iter_index = int(sys.argv[3])

    task = TaskBundle.load(task_dir)
    iter_dir = task_dir / "runs" / slug / "history" / f"iter_{iter_index:04d}"
    if not iter_dir.exists():
        print(f"iteration dir not found: {iter_dir}", file=sys.stderr)
        return 1

    outputs, stored = _load_outputs(iter_dir)

    scorer = build_scorer(task.metric)
    ctx = ScoringContext(task_name=task.config.task_name, iteration_index=iter_index)

    score_map = await scorer.score_iteration(
        outputs=outputs,
        eval_records=task.eval_records,
        gold_standard=task.gold_standard,
        context=ctx,
    )
    agg = aggregate_scores(task.eval_records, score_map)

    # Reconstruct the stored aggregate the same way (weighted mean over
    # eval records' weight) so the comparison is apples-to-apples.
    stored_wsum = 0.0
    stored_tw = 0.0
    for rec in task.eval_records:
        if rec.id in stored:
            stored_wsum += stored[rec.id] * rec.weight
            stored_tw += rec.weight
    stored_agg = stored_wsum / stored_tw if stored_tw else 0.0

    print(f"Task:       {task.config.task_name}")
    print(f"Slug:       {slug}")
    print(f"Iteration:  {iter_index}")
    print()
    print(f"Stored aggregate:     {stored_agg:6.2f}")
    print(f"Re-scored aggregate:  {agg.aggregate:6.2f}")
    print(f"Delta:                {agg.aggregate - stored_agg:+6.2f}")
    print()
    print(f"{'id':<8} {'scenario':<30} {'stored':>7} {'new':>7} {'delta':>7}  {'rationale'}")
    print("-" * 100)
    for rec in task.eval_records:
        old = stored.get(rec.id, 0.0)
        res = score_map.get(rec.id)
        new = res.aggregate if res else 0.0
        delta = new - old
        rationale = (res.rationale if res else "").replace("\n", " ")[:70]
        marker = " *" if abs(delta) > 0.01 else ""
        print(
            f"{rec.id:<8} {(rec.scenario or ''):<30} "
            f"{old:7.2f} {new:7.2f} {delta:+7.2f}{marker}  {rationale}"
        )

    print()
    print(f"Failed (score < 70): {', '.join(agg.failed_ids) if agg.failed_ids else '(none)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
