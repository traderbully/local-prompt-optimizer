"""Filesystem readers for task bundles and persisted run artifacts.

Pure I/O — no engine imports beyond :class:`lpo.core.task.TaskBundle`. These
functions back the read-only REST endpoints. Live run state (WebSocket push,
in-memory iteration queues) lives in :mod:`lpo.server.runs`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lpo.core.history import RunPaths
from lpo.core.multi_engine import UNIFIED_SLUG
from lpo.core.task import TaskBundle
from lpo.server.schemas import (
    ComparisonView,
    EvalOutputRow,
    IterationDetail,
    IterationSummary,
    RunState,
    TargetSummary,
    TaskDetail,
    TaskSummary,
)


# ---------------------------------------------------------------------------
# Task-level readers
# ---------------------------------------------------------------------------


def _is_task_dir(p: Path) -> bool:
    return p.is_dir() and (p / "config.yaml").exists() and (p / "eval_set.jsonl").exists()


def list_task_dirs(tasks_root: Path) -> list[Path]:
    if not tasks_root.exists():
        return []
    return sorted(p for p in tasks_root.iterdir() if _is_task_dir(p))


def _task_summary(task: TaskBundle) -> TaskSummary:
    # For Strategy C the canonical "run dir" is ``runs/_unified``; for others
    # it's ``runs/<slug>/``. `has_runs` is True if any of these exist.
    run_dirs: list[Path] = []
    if task.config.target_strategy == "unified_portable":
        run_dirs.append(task.root / "runs" / UNIFIED_SLUG)
    else:
        run_dirs += [task.root / "runs" / m.slug for m in task.config.target_models]

    return TaskSummary(
        name=task.config.task_name,
        path=str(task.root),
        strategy=task.config.target_strategy,
        mode=task.config.mode,
        output_type=task.config.output_type,
        targets=[
            TargetSummary(slug=m.slug, provider=m.provider, model_id=m.model_id)
            for m in task.config.target_models
        ],
        has_runs=any(p.exists() and any(p.iterdir()) for p in run_dirs if p.exists()),
        has_comparison=(task.root / "comparison" / "summary.json").exists(),
    )


def read_task_summary(task_path: Path) -> TaskSummary:
    return _task_summary(TaskBundle.load(task_path))


def read_all_tasks(tasks_root: Path) -> list[TaskSummary]:
    out: list[TaskSummary] = []
    for d in list_task_dirs(tasks_root):
        try:
            out.append(_task_summary(TaskBundle.load(d)))
        except Exception:  # noqa: BLE001 — surface a stub rather than 500 the whole list
            out.append(
                TaskSummary(
                    name=d.name,
                    path=str(d),
                    strategy="unknown",
                    mode="unknown",
                    output_type="unknown",
                    targets=[],
                    has_runs=False,
                    has_comparison=False,
                )
            )
    return out


def read_task_detail(task_path: Path) -> TaskDetail:
    task = TaskBundle.load(task_path)
    cfg_yaml = (task_path / "config.yaml").read_text(encoding="utf-8")
    metric_yaml = (task_path / "metric.yaml").read_text(encoding="utf-8")
    return TaskDetail(
        summary=_task_summary(task),
        task_md=task.task_md,
        seed_prompt=task.seed_prompt,
        config_yaml=cfg_yaml,
        metric_yaml=metric_yaml,
        eval_records=[r.model_dump() for r in task.eval_records],
        gold_count=len(task.gold_standard),
        metric_type=task.metric.type,
    )


# ---------------------------------------------------------------------------
# Iteration readers
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            out.append(json.loads(s))
        except json.JSONDecodeError:
            continue  # best-effort; partial writes at crash time
    return out


def _iteration_summary_from_disk(iter_dir: Path, index: int) -> IterationSummary:
    scores = _read_json(iter_dir / "scores.json")
    decision = _read_json(iter_dir / "decision.json")

    per_model: dict[str, float] | None = None
    if "per_model" in scores and isinstance(scores["per_model"], dict):
        per_model = {
            slug: float(entry.get("aggregate", 0.0))
            for slug, entry in scores["per_model"].items()
            if isinstance(entry, dict)
        }

    per_scenario_key = (
        "combined_per_scenario" if "combined_per_scenario" in scores else "per_scenario"
    )
    return IterationSummary(
        index=index,
        aggregate_score=float(scores.get("aggregate", 0.0)),
        decision=str(decision.get("decision", "unknown")),
        delta=float(decision.get("delta", 0.0)),
        cost_usd=float(decision.get("cost_usd", 0.0)),
        timestamp=str(decision.get("timestamp", "")),
        failed_ids=list(scores.get("failed_ids") or []),
        per_scenario={
            k: float(v) for k, v in (scores.get(per_scenario_key) or {}).items()
        },
        per_model=per_model,
    )


def _iteration_dirs(history_root: Path) -> list[tuple[int, Path]]:
    if not history_root.exists():
        return []
    out: list[tuple[int, Path]] = []
    for d in sorted(history_root.iterdir()):
        if not d.is_dir() or not d.name.startswith("iter_"):
            continue
        try:
            idx = int(d.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        out.append((idx, d))
    return sorted(out, key=lambda t: t[0])


def read_run_state(task_path: Path, slug: str) -> RunState:
    """Reconstruct one model's run state from its ``runs/<slug>/`` directory."""
    paths = RunPaths(task_root=task_path, model_slug=slug)
    if not paths.run_root.exists():
        return RunState(slug=slug, exists=False)

    iters = [_iteration_summary_from_disk(d, idx) for idx, d in _iteration_dirs(paths.history_root)]
    best_score = max((it.aggregate_score for it in iters), default=None)
    best_prompt = (
        paths.best_prompt.read_text(encoding="utf-8")
        if paths.best_prompt.exists() else None
    )
    return RunState(
        slug=slug,
        exists=True,
        best_score=best_score,
        best_prompt=best_prompt,
        iteration_count=len(iters),
        latest_iteration=iters[-1].index if iters else None,
        iterations=iters,
        winner_ready=(paths.winner_root / "prompt.txt").exists(),
    )


def read_iteration_detail(task_path: Path, slug: str, index: int) -> IterationDetail:
    paths = RunPaths(task_root=task_path, model_slug=slug)
    iter_dir = paths.iteration_dir(index)
    if not iter_dir.exists():
        raise FileNotFoundError(f"No iteration {index} for {slug}")

    summary = _iteration_summary_from_disk(iter_dir, index)
    prompt_path = iter_dir / "prompt.txt"
    prompt = prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else ""

    outputs_raw = _read_jsonl(iter_dir / "outputs.jsonl")
    outputs: list[EvalOutputRow] = []
    for row in outputs_raw:
        # Best-effort; unknown extras pass through via Config.extra="allow".
        try:
            outputs.append(EvalOutputRow.model_validate(row))
        except Exception:
            outputs.append(
                EvalOutputRow(
                    id=str(row.get("id", "?")),
                    input=row.get("input"),
                    output=row.get("output"),
                    score=row.get("score"),
                    rationale=row.get("rationale"),
                    scenario=row.get("scenario"),
                    model_slug=row.get("model_slug"),
                )
            )

    analysis_path = iter_dir / "overseer_analysis.md"
    analysis = analysis_path.read_text(encoding="utf-8") if analysis_path.exists() else None

    return IterationDetail(
        summary=summary,
        prompt=prompt,
        outputs=outputs,
        overseer_analysis_md=analysis,
        scores_full=_read_json(iter_dir / "scores.json"),
        decision_full=_read_json(iter_dir / "decision.json"),
    )


# ---------------------------------------------------------------------------
# Comparison + winners
# ---------------------------------------------------------------------------


def read_comparison(task_path: Path) -> ComparisonView:
    summary_path = task_path / "comparison" / "summary.json"
    report_path = task_path / "comparison" / "report.md"
    if not summary_path.exists():
        return ComparisonView(present=False)
    return ComparisonView(
        present=True,
        summary=_read_json(summary_path),
        report_md=report_path.read_text(encoding="utf-8") if report_path.exists() else None,
    )


def read_winner(task_path: Path, slug: str) -> dict[str, Any]:
    paths = RunPaths(task_root=task_path, model_slug=slug)
    prompt_path = paths.winner_root / "prompt.txt"
    report_path = paths.winner_root / "report.md"
    return {
        "slug": slug,
        "present": prompt_path.exists(),
        "prompt": prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else None,
        "report_md": report_path.read_text(encoding="utf-8") if report_path.exists() else None,
    }


def resolve_task_path(tasks_root: Path, task_name: str) -> Path:
    """Map a task_name (as it appears in :class:`TaskSummary`) to its directory.

    The on-disk directory name can differ from ``task_name`` (e.g.
    ``tasks/example_json_extract/`` holds ``task_name: ebay_listing_generator``).
    We resolve by scanning the configs on the fly — cheap enough at UI scale.
    """
    # Fast path: directory name == task_name
    direct = tasks_root / task_name
    if _is_task_dir(direct):
        try:
            task = TaskBundle.load(direct)
            if task.config.task_name == task_name or direct.name == task_name:
                return direct
        except Exception:  # noqa: BLE001
            pass
    for d in list_task_dirs(tasks_root):
        try:
            task = TaskBundle.load(d)
        except Exception:  # noqa: BLE001
            continue
        if task.config.task_name == task_name or d.name == task_name:
            return d
    raise FileNotFoundError(f"No task named {task_name!r} under {tasks_root}")
