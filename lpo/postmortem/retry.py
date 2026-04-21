"""Focused validation retry.

Runs the patched prompt against the full eval set exactly once. For v1
this is deliberately a single scoring pass with no further mutation —
the design doc (STAGE_8_DESIGN.md §6) allows Overseer-driven refinement
in iterations 2-3 but that adds complexity that's only useful when there
is real stochasticity. LPO target clients run with ``temperature=0.0`` by
default, so a single deterministic pass IS the measurement; repeating it
doesn't produce new information.

The ``max_retry_iterations`` config field is preserved for forward
compatibility; when a future change adds Overseer refinement, this
module expands to a short loop, and the decision gate's ``retry_iterations_run``
field will reflect the actual count.

Artifacts are written to ``<task_root>/runs/<slug>/postmortem/retry/``:

    retry/
    ├── prompt.txt               # the patched prompt under test
    └── iter_0001/
        ├── prompt.txt
        ├── outputs.jsonl
        ├── scores.json
        └── decision.json

Reuses :class:`IterationRunner` for the generate + score phase and
:func:`build_target_context` for the client + scorer wiring — the design
doc's "inherits, does not replicate" invariant for the ratchet.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lpo.config.schema import TargetModelConfig
from lpo.core.cost import CostTracker
from lpo.core.iteration import IterationResult, IterationRunner
from lpo.core.task import TaskBundle
from lpo.postmortem.schemas import PostmortemConfig

log = logging.getLogger("lpo.postmortem.retry")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class FocusedRetryResult:
    """What :func:`run_focused_retry` returns."""

    patched_prompt: str
    iterations: list[IterationResult]
    retry_root: Path
    best_score: float
    cost_before_retry_usd: float
    cost_after_retry_usd: float

    @property
    def cost_usd(self) -> float:
        return max(0.0, self.cost_after_retry_usd - self.cost_before_retry_usd)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def run_focused_retry(
    task: TaskBundle,
    *,
    slug: str,
    target_cfg: TargetModelConfig,
    patched_prompt: str,
    cfg: PostmortemConfig,
    cost: CostTracker,
    runner_factory: Any | None = None,
) -> FocusedRetryResult:
    """Score ``patched_prompt`` against the task's full eval set.

    The ``runner_factory`` parameter is an injection seam for tests. When
    ``None`` (the production path) we build an :class:`IterationRunner`
    the same way :func:`lpo.core.multi_engine._run_all_targets` does,
    using :func:`lpo.core.target_factory.build_target_context`. Tests pass
    a stub factory that returns a pre-built runner without touching the
    network. The factory signature is
    ``(task, target_cfg, cost) -> tuple[IterationRunner, cleanup_coroutine]``.
    """
    retry_root = task.root / "runs" / slug / "postmortem" / "retry"
    retry_root.mkdir(parents=True, exist_ok=True)
    (retry_root / "prompt.txt").write_text(patched_prompt, encoding="utf-8")

    cost_before = cost.total_usd
    runner, cleanup = await _build_runner(
        task=task,
        target_cfg=target_cfg,
        cost=cost,
        runner_factory=runner_factory,
    )
    try:
        iterations: list[IterationResult] = []
        for i in range(1, cfg.max_retry_iterations + 1):
            # v1: only iter 1 runs; Overseer refinement at i>=2 is
            # deliberately unimplemented until we have a user case that
            # needs it. Breaking here keeps artifacts consistent with the
            # design (single-iteration measurement == the patch's effect).
            result = await runner.run(
                iteration_index=i,
                prompt=patched_prompt,
                eval_records=task.eval_records,
                gold_standard=task.gold_standard,
                task_name=task.config.task_name,
            )
            _write_retry_iteration(retry_root / f"iter_{i:04d}", result, cost.total_usd)
            iterations.append(result)
            # v1 short-circuit: measuring the same prompt twice under
            # temperature=0 adds cost without adding signal. Stop after
            # one iteration; preserve the field for future multi-iter
            # refinement loops.
            break

        # Check postmortem-specific cost cap AFTER the call so the
        # operator always sees the measurement for their $ spent.
        cost_after = cost.total_usd
        if cost_after - cost_before > cfg.cost_cap_usd:
            log.warning(
                "Focused retry cost ($%.4f) exceeded postmortem.cost_cap_usd "
                "($%.2f). Retry completed, but consider lowering the retry "
                "scope or raising the cap.",
                cost_after - cost_before, cfg.cost_cap_usd,
            )

        best = max((it.aggregated.aggregate for it in iterations), default=0.0)
        return FocusedRetryResult(
            patched_prompt=patched_prompt,
            iterations=iterations,
            retry_root=retry_root,
            best_score=best,
            cost_before_retry_usd=cost_before,
            cost_after_retry_usd=cost_after,
        )
    finally:
        if cleanup is not None:
            await cleanup()


# ---------------------------------------------------------------------------
# Runner construction — injection seam lets tests stub the whole thing.
# ---------------------------------------------------------------------------


async def _build_runner(
    *,
    task: TaskBundle,
    target_cfg: TargetModelConfig,
    cost: CostTracker,
    runner_factory: Any | None,
) -> tuple[IterationRunner, Any]:
    if runner_factory is not None:
        built = runner_factory(task, target_cfg, cost)
        # Tests may return a bare runner OR (runner, cleanup). Accept both.
        if isinstance(built, tuple):
            return built[0], built[1]
        return built, None

    # Production path — reuse the same wiring the main engine uses.
    from lpo.core.target_factory import build_target_context

    ctx = build_target_context(task, target_cfg, cost, mutator_mode="null")
    runner = IterationRunner(
        client=ctx.client,
        target_cfg=ctx.cfg,
        scorer=ctx.scorer,
        concurrency=task.config.eval_concurrency,
    )

    async def cleanup() -> None:
        for fn in ctx.cleanups:
            try:
                await fn()
            except Exception:  # pragma: no cover — best-effort close
                log.exception("cleanup failed for postmortem retry context")

    return runner, cleanup


# ---------------------------------------------------------------------------
# Artifact writing — matches the engine's on-disk shape.
# ---------------------------------------------------------------------------


def _write_retry_iteration(
    iter_dir: Path,
    result: IterationResult,
    total_cost_usd: float,
) -> None:
    """Write the same four files the engine writes per iteration so the
    Analyst can later read postmortem retry iterations with the same
    loader used for the main ratchet history."""
    iter_dir.mkdir(parents=True, exist_ok=True)
    rec = result.record
    (iter_dir / "prompt.txt").write_text(rec.prompt, encoding="utf-8")
    # outputs.jsonl — one line per eval example.
    with (iter_dir / "outputs.jsonl").open("w", encoding="utf-8") as fh:
        for row in rec.outputs:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    (iter_dir / "scores.json").write_text(
        json.dumps({
            "aggregate": rec.aggregate_score,
            "per_example": rec.per_example,
            "per_scenario": rec.per_scenario,
            "failed_ids": rec.failed_ids,
        }, ensure_ascii=False),
        encoding="utf-8",
    )
    (iter_dir / "decision.json").write_text(
        json.dumps({
            "decision": "postmortem_retry",
            "delta": 0.0,
            "timings": rec.timings,
            "cost_usd": total_cost_usd,
            "timestamp": rec.timestamp,
            "notes": "focused validation retry (Stage 8)",
        }, ensure_ascii=False),
        encoding="utf-8",
    )
