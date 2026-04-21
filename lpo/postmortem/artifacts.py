"""Postmortem artifact loader.

Reads the full run history for a given ``(task_root, slug)`` off disk into
typed :class:`RunHistoryBundle` + :class:`IterationArtifact` objects the
Postmortem Analyst can reason over.

The on-disk layout we parse is the one written by
:class:`lpo.core.history.IterationRecord.write` and
:func:`lpo.core.multi_engine._write_unified_iteration`:

    <task_root>/
    ├── task.md
    ├── eval_set.jsonl
    ├── gold_standard.jsonl
    ├── metric.yaml
    ├── config.yaml
    ├── prompt_seed.txt
    └── runs/<slug>/
        ├── prompt.txt
        ├── prompt.txt.best
        ├── winner/
        │   ├── prompt.txt
        │   └── report.md
        └── history/
            ├── iter_0001/
            │   ├── prompt.txt
            │   ├── outputs.jsonl
            │   ├── scores.json
            │   ├── decision.json
            │   └── overseer_analysis.md      (iter 2+)
            └── iter_0002/ ...

Both single-target and unified (Strategy C) layouts are supported — the
unified ``scores.json`` has a ``per_model`` block which we expose as an
optional field on :class:`IterationScores` so the Analyst can reason about
cross-model breakdowns when appropriate.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lpo.core.task import TaskBundle


_ITER_DIR_RE = re.compile(r"^iter_(\d+)$")


# ---------------------------------------------------------------------------
# Typed artifact classes
# ---------------------------------------------------------------------------


@dataclass
class IterationScores:
    """Parsed ``scores.json`` for one iteration."""

    aggregate: float
    per_example: dict[str, float]
    per_scenario: dict[str, float]
    failed_ids: list[str]
    # Only populated for Strategy-C unified runs. Maps slug -> its own
    # per-example / per-scenario / aggregate dict.
    per_model: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class IterationDecision:
    """Parsed ``decision.json`` for one iteration."""

    decision: str
    delta: float
    timings: dict[str, float]
    cost_usd: float
    timestamp: str
    notes: str


@dataclass
class IterationArtifact:
    """Everything the Postmortem Analyst needs to know about one iteration."""

    index: int
    prompt: str
    outputs: list[dict[str, Any]]
    scores: IterationScores
    decision: IterationDecision
    overseer_analysis: str | None = None


@dataclass
class RunHistoryBundle:
    """Fully-loaded run history for a single target slug.

    ``task`` is already a :class:`~lpo.core.task.TaskBundle` so the Analyst
    sees the task description, eval set, gold standard, metric, and seed
    prompt as first-class objects — not raw files — consistent with the
    rest of the codebase.
    """

    task: TaskBundle
    slug: str
    run_root: Path
    iterations: list[IterationArtifact]
    winner_prompt: str | None
    winner_report: str | None

    @property
    def best_iteration(self) -> IterationArtifact | None:
        """Return the iteration with the highest aggregate score, or None
        if the run produced no iterations. Ties resolve to the earliest
        (matching the engine's ratchet semantics — first-to-reach wins)."""
        if not self.iterations:
            return None
        # Sort by (-score, index) so highest score wins; lower index wins ties.
        return min(
            self.iterations,
            key=lambda it: (-it.scores.aggregate, it.index),
        )

    @property
    def total_iterations(self) -> int:
        return len(self.iterations)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_run_history(task_root: Path | str, slug: str) -> RunHistoryBundle:
    """Load the complete run history for ``(task_root, slug)`` off disk.

    Raises :class:`FileNotFoundError` if the task root doesn't exist or has
    no ``runs/<slug>/`` directory. Missing optional artifacts (a run with
    zero iterations, or no winner yet) produce empty lists / ``None``
    rather than errors — the Analyst can still reason over a partial run.
    """
    task_root = Path(task_root).resolve()
    task = TaskBundle.load(task_root)

    run_root = task_root / "runs" / slug
    if not run_root.exists():
        raise FileNotFoundError(
            f"No run directory for slug {slug!r} under {task_root}. "
            f"Expected {run_root}."
        )

    iterations = _load_iterations(run_root / "history")
    winner_prompt, winner_report = _load_winner(run_root / "winner")

    return RunHistoryBundle(
        task=task,
        slug=slug,
        run_root=run_root,
        iterations=iterations,
        winner_prompt=winner_prompt,
        winner_report=winner_report,
    )


def _load_iterations(history_root: Path) -> list[IterationArtifact]:
    if not history_root.exists():
        return []
    entries: list[IterationArtifact] = []
    for child in sorted(history_root.iterdir()):
        if not child.is_dir():
            continue
        m = _ITER_DIR_RE.match(child.name)
        if not m:
            continue
        index = int(m.group(1))
        entries.append(_load_single_iteration(child, index))
    return entries


def _load_single_iteration(iter_dir: Path, index: int) -> IterationArtifact:
    prompt = _read_text(iter_dir / "prompt.txt", default="")
    outputs = _read_jsonl(iter_dir / "outputs.jsonl")
    scores = _parse_scores(_read_json(iter_dir / "scores.json", default={}))
    decision = _parse_decision(_read_json(iter_dir / "decision.json", default={}))
    overseer = iter_dir / "overseer_analysis.md"
    analysis = overseer.read_text(encoding="utf-8") if overseer.exists() else None
    return IterationArtifact(
        index=index,
        prompt=prompt,
        outputs=outputs,
        scores=scores,
        decision=decision,
        overseer_analysis=analysis,
    )


def _parse_scores(raw: dict[str, Any]) -> IterationScores:
    # Single-engine layout: per_example at top level. Unified layout uses
    # combined_per_example + per_model. We unify these so the Analyst sees
    # a consistent shape.
    per_example = raw.get("per_example") or raw.get("combined_per_example") or {}
    per_scenario = raw.get("per_scenario") or raw.get("combined_per_scenario") or {}
    return IterationScores(
        aggregate=float(raw.get("aggregate", 0.0)),
        per_example={k: float(v) for k, v in per_example.items()},
        per_scenario={k: float(v) for k, v in per_scenario.items()},
        failed_ids=list(raw.get("failed_ids", [])),
        per_model=dict(raw.get("per_model") or {}),
    )


def _parse_decision(raw: dict[str, Any]) -> IterationDecision:
    return IterationDecision(
        decision=str(raw.get("decision", "unknown")),
        delta=float(raw.get("delta", 0.0)),
        timings=dict(raw.get("timings") or {}),
        cost_usd=float(raw.get("cost_usd", 0.0)),
        timestamp=str(raw.get("timestamp", "")),
        notes=str(raw.get("notes", "")),
    )


def _load_winner(winner_root: Path) -> tuple[str | None, str | None]:
    if not winner_root.exists():
        return None, None
    prompt_path = winner_root / "prompt.txt"
    report_path = winner_root / "report.md"
    prompt = prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else None
    report = report_path.read_text(encoding="utf-8") if report_path.exists() else None
    return prompt, report


def _read_text(path: Path, *, default: str = "") -> str:
    if not path.exists():
        return default
    return path.read_text(encoding="utf-8")


def _read_json(path: Path, *, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # skip malformed lines; loader must be tolerant of partial writes
    return out
