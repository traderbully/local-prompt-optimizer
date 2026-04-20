"""Multi-target orchestration — Strategies B and C. See `LPO_SDP.md` §3.4.

The single-target :class:`RatchetEngine` is the inner loop. This module
supplies the two outer-loop orchestrators:

* :func:`run_parallel_independent` — Strategy B. One full optimization per
  target, each isolated (own Overseer conversation, own history directory
  under ``runs/<slug>/``, own winning prompt). Execution is either sequential
  (default, safe for single-GPU local hosts) or parallel (for users with
  multiple endpoints or a beefy rig).

* :func:`run_unified_portable` — Strategy C. A single shared prompt is
  evaluated against every target per iteration; per-model aggregates are
  combined via ``min`` / ``mean`` / ``weighted_mean`` and the ratchet decision
  uses that combined score. The Overseer sees per-model breakdowns in its
  feedback so it can target the weakest link without over-specializing.

Strategy A (single target) keeps the simple :mod:`lpo.core.engine` path in
the CLI — but is trivially equivalent to ``run_parallel_independent`` with
one target.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from lpo.config.schema import RunConfig, TargetModelConfig
from lpo.core.cost import CostTracker
from lpo.core.engine import EngineResult, RatchetEngine, StopReason, UserGate
from lpo.core.history import (
    IterationRecord,
    RunPaths,
    append_jsonl,
    atomic_write_json,
    atomic_write_text,
)
from lpo.core.iteration import IterationRunner
from lpo.core.target_factory import TargetContext, build_target_context
from lpo.core.task import TaskBundle

log = logging.getLogger("lpo.multi_engine")

IterationCallback = Callable[[str, IterationRecord], None]
"""(target_slug, record) -> None. Fires once per iteration across both strategies.
For Strategy C, ``target_slug`` is ``"_unified"`` and ``record`` is the aggregate."""


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class PerModelResult:
    slug: str
    model_id: str
    best_score: float
    best_prompt: str
    iterations: int
    stop_reason: StopReason
    # Subset of the cost that can be attributed to this model's run. For
    # Strategy B this is the delta between CostTracker totals before and
    # after the model's run. For Strategy C all models share cost equally
    # (one Overseer conversation, one set of judge calls) and this is the
    # per-model prorated share.
    cost_usd: float
    history: list[IterationRecord] = field(default_factory=list)


@dataclass
class MultiResult:
    strategy: str
    per_model: list[PerModelResult]
    # Strategy-C-only: the single shared winning prompt (duplicated across
    # PerModelResult.best_prompt for convenience).
    shared_prompt: str | None = None
    shared_best_score: float | None = None
    shared_stop_reason: StopReason | None = None
    total_cost_usd: float = 0.0


# ---------------------------------------------------------------------------
# Strategy B — parallel_independent
# ---------------------------------------------------------------------------


UserGateFactory = Callable[[str], "UserGate | None"]
"""``(target_slug) -> UserGate``. Multi-engine callers supply a factory so
each RatchetEngine gets its own gate closure routed to the correct model tab
in the UI (Strategy B) or a single shared gate (Strategy C uses slug
``"_unified"``)."""


async def run_parallel_independent(
    task: TaskBundle,
    *,
    cost: CostTracker,
    mutator_mode: str = "auto",
    iteration_callback: IterationCallback | None = None,
    user_gate_factory: UserGateFactory | None = None,
    initial_mode: str | None = None,
) -> MultiResult:
    """Run one full optimization per target model, independently.

    Respects ``RunConfig.parallel_execution``:
      - ``sequential`` — one target at a time. Default; safe for single-GPU
        local hosts because LM Studio serves one model at a time.
      - ``parallel`` — all targets concurrently via ``asyncio.gather``.
    """
    cfg = task.config
    targets = list(cfg.target_models)

    async def _run_one(target: TargetModelConfig) -> PerModelResult:
        cost_before = cost.total_usd
        ctx = build_target_context(task, target, cost, mutator_mode=mutator_mode)
        try:
            cb = (lambda rec: iteration_callback(target.slug, rec)) if iteration_callback else None
            gate = user_gate_factory(target.slug) if user_gate_factory else None
            engine = RatchetEngine(
                task=task,
                target_cfg=target,
                client=ctx.client,
                scorer=ctx.scorer,
                mutator=ctx.mutator,
                cost_tracker=cost,
                iteration_callback=cb,
                user_gate=gate,
                initial_mode=initial_mode,
            )
            log.info("[%s] starting Strategy B run", target.slug)
            result: EngineResult = await engine.run()
            log.info(
                "[%s] done: best=%.2f iters=%d stop=%s",
                target.slug,
                result.best_score,
                len(result.iterations),
                result.stop_reason.value,
            )
        finally:
            await ctx.aclose()
        return PerModelResult(
            slug=target.slug,
            model_id=target.model_id,
            best_score=result.best_score,
            best_prompt=result.best_prompt,
            iterations=len(result.iterations),
            stop_reason=result.stop_reason,
            cost_usd=round(cost.total_usd - cost_before, 6),
            history=result.iterations,
        )

    if cfg.parallel_execution == "parallel":
        per_model = await asyncio.gather(*[_run_one(t) for t in targets])
    else:
        per_model = []
        for t in targets:
            per_model.append(await _run_one(t))

    return MultiResult(
        strategy="parallel_independent",
        per_model=list(per_model),
        total_cost_usd=cost.total_usd,
    )


# ---------------------------------------------------------------------------
# Strategy C — unified_portable
# ---------------------------------------------------------------------------


UNIFIED_SLUG = "_unified"


def combine_scores(
    per_model_scores: dict[str, float],
    method: str,
    weights: dict[str, float] | None = None,
) -> float:
    """Aggregate per-model aggregate scores into a single ratchet signal.

    ``weights`` is only consulted when ``method == "weighted_mean"``.
    """
    if not per_model_scores:
        return 0.0
    vals = list(per_model_scores.values())
    if method == "min":
        return min(vals)
    if method == "mean":
        return sum(vals) / len(vals)
    if method == "weighted_mean":
        if not weights:
            raise ValueError("weighted_mean requires weights")
        num = sum(per_model_scores[s] * weights[s] for s in per_model_scores)
        den = sum(weights[s] for s in per_model_scores) or 1.0
        return num / den
    raise ValueError(f"Unknown unified_aggregation method: {method!r}")


def _unified_paths(task_root: Path) -> RunPaths:
    """Strategy C uses a reserved slug directory for the shared prompt."""
    return RunPaths(task_root=task_root, model_slug=UNIFIED_SLUG)


async def run_unified_portable(
    task: TaskBundle,
    *,
    cost: CostTracker,
    mutator_mode: str = "auto",
    iteration_callback: IterationCallback | None = None,
    user_gate_factory: UserGateFactory | None = None,
    initial_mode: str | None = None,
) -> MultiResult:
    """Optimize a single prompt that must work across all target models.

    Per-iteration flow:
      1. Run the current prompt against every target (sequential or parallel
         by ``parallel_execution``).
      2. Record per-model aggregates + per-example details.
      3. Combine aggregates via ``unified_aggregation`` → ratchet signal.
      4. Standard ratchet decision; Overseer sees per-model breakdowns in the
         aggregated :class:`IterationRecord`.
    """
    cfg = task.config
    method = cfg.unified_aggregation
    weights: dict[str, float] | None = None
    if method == "weighted_mean":
        weights = {m.slug: float(m.weight or 0.0) for m in cfg.target_models}

    # Build a TargetContext per model. The overseer+mutator lives with the
    # FIRST context — there is only one Overseer conversation in Strategy C.
    contexts: list[TargetContext] = []
    for i, t in enumerate(cfg.target_models):
        # Only the first model gets the real mutator; others get Null (their
        # mutator is never invoked anyway, we drive mutation from contexts[0]).
        ctx = build_target_context(
            task, t, cost,
            mutator_mode=mutator_mode if i == 0 else "null",
        )
        contexts.append(ctx)
    primary = contexts[0]

    # Strategy C has one Overseer conversation; route the gate through the
    # reserved unified slug so a single supervised-signal stream drives it.
    gate = user_gate_factory(UNIFIED_SLUG) if user_gate_factory else None
    current_mode = (initial_mode or cfg.mode or "autonomous").lower()

    paths = _unified_paths(task.root)
    paths.ensure()

    try:
        # Seed / resume logic — same rules as single-engine.
        current_prompt = _initial_unified_prompt(task, paths)
        atomic_write_text(paths.current_prompt, current_prompt)

        best_prompt = current_prompt
        best_score = float("-inf")
        history: list[IterationRecord] = []
        per_model_best: dict[str, float] = {c.slug: 0.0 for c in contexts}
        iter_since_improve = 0
        stop_reason = StopReason.MAX_ITERATIONS

        for i in range(1, cfg.stop_conditions.max_iterations + 1):
            t0 = time.perf_counter()

            # Run the same prompt across every target.
            per_model = await _run_all_targets(
                contexts,
                prompt=current_prompt,
                iteration_index=i,
                task=task,
                parallel=cfg.parallel_execution == "parallel",
            )

            per_model_aggs = {slug: r.aggregate_score for slug, r in per_model.items()}
            combined = combine_scores(per_model_aggs, method, weights)

            # Build the combined iteration record. Scenario breakdown is
            # aggregated too, so the overseer can still reason about scenarios.
            agg_record = _combine_iteration_records(
                iteration_index=i,
                prompt=current_prompt,
                per_model=per_model,
                combined_score=combined,
                method=method,
            )

            delta = combined - (best_score if best_score != float("-inf") else 0.0)
            if combined > best_score:
                agg_record.decision = "accepted" if history else "initial"
                agg_record.delta = delta
                best_prompt = current_prompt
                best_score = combined
                per_model_best = dict(per_model_aggs)
                atomic_write_text(paths.best_prompt, best_prompt)
                iter_since_improve = 0
            else:
                agg_record.decision = "rejected"
                agg_record.delta = combined - best_score
                iter_since_improve += 1
                current_prompt = best_prompt
                atomic_write_text(paths.current_prompt, current_prompt)

            agg_record.timings["total_s"] = round(time.perf_counter() - t0, 3)
            agg_record.cost_usd = cost.total_usd
            _write_unified_iteration(paths, agg_record, per_model)
            append_jsonl(
                paths.log_file,
                {
                    "iter": i,
                    "score": agg_record.aggregate_score,
                    "decision": agg_record.decision,
                    "delta": agg_record.delta,
                    "per_model": per_model_aggs,
                    "method": method,
                    "cost_usd": agg_record.cost_usd,
                },
            )
            history.append(agg_record)
            if iteration_callback is not None:
                try:
                    iteration_callback(UNIFIED_SLUG, agg_record)
                except Exception:  # pragma: no cover
                    log.exception("iteration_callback raised")

            log.info(
                "iter=%d combined(%s)=%.2f decision=%s best=%.2f cost=$%.4f per_model=%s",
                i,
                method,
                combined,
                agg_record.decision,
                best_score,
                cost.total_usd,
                ", ".join(f"{s}={v:.1f}" for s, v in per_model_aggs.items()),
            )

            # Stop conditions.
            if best_score >= cfg.stop_conditions.target_score:
                stop_reason = StopReason.TARGET_SCORE
                break
            if iter_since_improve >= cfg.stop_conditions.plateau_patience:
                stop_reason = StopReason.PLATEAU
                break
            if cost.over_cap(cfg.stop_conditions.cost_cap_usd):
                stop_reason = StopReason.COST_CAP
                break
            if i >= cfg.stop_conditions.max_iterations:
                stop_reason = StopReason.MAX_ITERATIONS
                break

            # User gate (Supervised / Manual). Same semantics as single-engine.
            pending_feedback = ""
            if gate is not None and current_mode != "autonomous":
                decision = await gate(agg_record, current_mode)
                if decision.mode is not None and decision.mode != current_mode:
                    log.info("mode switch: %s -> %s", current_mode, decision.mode)
                    current_mode = decision.mode.lower()
                if decision.stop:
                    stop_reason = StopReason.USER_STOP
                    break
                pending_feedback = decision.feedback

            # Single Overseer call drives the next prompt. It sees the
            # combined record, which carries per-model breakdowns in its
            # ``notes`` field (see _combine_iteration_records).
            proposal = await primary.mutator.propose(
                current_prompt=current_prompt,
                best_prompt=best_prompt,
                history=history,
                user_feedback=pending_feedback,
            )
            _persist_unified_analysis(paths.iteration_dir(i), proposal)

            if cost.over_cap(cfg.stop_conditions.cost_cap_usd):
                stop_reason = StopReason.COST_CAP
                break
            if proposal.new_prompt.strip() == current_prompt.strip():
                stop_reason = StopReason.MUTATOR_NOOP
                break
            current_prompt = proposal.new_prompt
            atomic_write_text(paths.current_prompt, current_prompt)

        # Winner artifacts for Strategy C.
        paths.winner_root.mkdir(parents=True, exist_ok=True)
        atomic_write_text(paths.winner_root / "prompt.txt", best_prompt)
        atomic_write_text(
            paths.winner_root / "report.md",
            _render_unified_report(
                task_name=task.config.task_name,
                best_score=best_score if best_score != float("-inf") else 0.0,
                iterations=len(history),
                stop_reason=stop_reason,
                best_prompt=best_prompt,
                per_model_best=per_model_best,
                method=method,
            ),
        )

    finally:
        for ctx in contexts:
            await ctx.aclose()

    # Per-model result objects for the comparison report.
    per_model_results = [
        PerModelResult(
            slug=c.slug,
            model_id=c.cfg.model_id,
            best_score=per_model_best.get(c.slug, 0.0),
            best_prompt=best_prompt,
            iterations=len(history),
            stop_reason=stop_reason,
            cost_usd=round(cost.total_usd / max(1, len(contexts)), 6),
        )
        for c in contexts
    ]

    return MultiResult(
        strategy="unified_portable",
        per_model=per_model_results,
        shared_prompt=best_prompt,
        shared_best_score=best_score if best_score != float("-inf") else 0.0,
        shared_stop_reason=stop_reason,
        total_cost_usd=cost.total_usd,
    )


# ---------------------------------------------------------------------------
# Strategy C helpers
# ---------------------------------------------------------------------------


async def _run_all_targets(
    contexts: list[TargetContext],
    *,
    prompt: str,
    iteration_index: int,
    task: TaskBundle,
    parallel: bool,
) -> dict[str, IterationRecord]:
    """Run the given prompt against every target in one iteration."""

    async def _one(ctx: TargetContext) -> tuple[str, IterationRecord]:
        runner = IterationRunner(
            client=ctx.client,
            target_cfg=ctx.cfg,
            scorer=ctx.scorer,
            concurrency=task.config.eval_concurrency,
        )
        r = await runner.run(
            iteration_index=iteration_index,
            prompt=prompt,
            eval_records=task.eval_records,
            gold_standard=task.gold_standard,
            task_name=task.config.task_name,
        )
        return ctx.slug, r.record

    if parallel:
        pairs = await asyncio.gather(*[_one(c) for c in contexts])
    else:
        pairs = []
        for c in contexts:
            pairs.append(await _one(c))
    return dict(pairs)


def _combine_iteration_records(
    *,
    iteration_index: int,
    prompt: str,
    per_model: dict[str, IterationRecord],
    combined_score: float,
    method: str,
) -> IterationRecord:
    """Fold N per-model records into the single :class:`IterationRecord` the
    ratchet loop and the Overseer consume.

    * ``per_example`` averages per-example scores across models (an eye-ball
      signal only; the authoritative per-model numbers live in ``outputs``).
    * ``per_scenario`` averages scenario scores across models.
    * ``outputs`` is a flat list with a ``model_slug`` field so per-row origin
      is preserved.
    * ``notes`` carries a pretty-printed per-model breakdown that the Overseer
      renders inline — this is the cross-model signal SDP §5.3 describes.
    """
    all_outputs: list[dict] = []
    per_example_sum: dict[str, float] = {}
    per_example_n: dict[str, int] = {}
    per_scenario_sum: dict[str, float] = {}
    per_scenario_n: dict[str, int] = {}
    failed_ids_set: set[str] = set()
    timings: dict[str, float] = {}

    for slug, rec in per_model.items():
        for row in rec.outputs:
            row_with_slug = dict(row)
            row_with_slug["model_slug"] = slug
            all_outputs.append(row_with_slug)
        for eid, sc in rec.per_example.items():
            per_example_sum[eid] = per_example_sum.get(eid, 0.0) + sc
            per_example_n[eid] = per_example_n.get(eid, 0) + 1
        for scen, sc in rec.per_scenario.items():
            per_scenario_sum[scen] = per_scenario_sum.get(scen, 0.0) + sc
            per_scenario_n[scen] = per_scenario_n.get(scen, 0) + 1
        failed_ids_set.update(rec.failed_ids)
        for k, v in rec.timings.items():
            timings[f"{slug}.{k}"] = v

    per_example_mean = {
        eid: per_example_sum[eid] / per_example_n[eid] for eid in per_example_sum
    }
    per_scenario_mean = {
        scen: per_scenario_sum[scen] / per_scenario_n[scen] for scen in per_scenario_sum
    }

    # Render per-model breakdown that the Overseer will see.
    per_model_lines = []
    for slug, rec in per_model.items():
        scen_bits = ", ".join(f"{k}={v:.1f}" for k, v in sorted(rec.per_scenario.items()))
        per_model_lines.append(
            f"- {slug}: aggregate={rec.aggregate_score:.2f}"
            + (f"  scenarios: {scen_bits}" if scen_bits else "")
            + (f"  failed: {', '.join(rec.failed_ids)}" if rec.failed_ids else "")
        )
    weakest = min(per_model.values(), key=lambda r: r.aggregate_score)
    weakest_slug = next(s for s, r in per_model.items() if r is weakest)
    notes = (
        "Per-model results (Strategy C):\n"
        + "\n".join(per_model_lines)
        + f"\n\nAggregation method: {method}"
        + f"  →  combined={combined_score:.2f}"
        + f"\nWeakest link: {weakest_slug} ({weakest.aggregate_score:.2f})"
    )

    return IterationRecord(
        index=iteration_index,
        prompt=prompt,
        aggregate_score=combined_score,
        per_example=per_example_mean,
        per_scenario=per_scenario_mean,
        failed_ids=sorted(failed_ids_set),
        outputs=all_outputs,
        decision="pending",
        timings=timings,
        notes=notes,
    )


def _write_unified_iteration(
    paths: RunPaths,
    record: IterationRecord,
    per_model: dict[str, IterationRecord],
) -> None:
    """Persist a unified iteration with per-model breakdown + shared shape."""
    d = paths.iteration_dir(record.index)
    d.mkdir(parents=True, exist_ok=True)
    atomic_write_text(d / "prompt.txt", record.prompt)

    # Flat outputs.jsonl with model_slug field (preserves SDP layout).
    out_path = d / "outputs.jsonl"
    if out_path.exists():
        out_path.unlink()
    for row in record.outputs:
        append_jsonl(out_path, row)

    atomic_write_json(
        d / "scores.json",
        {
            "aggregate": record.aggregate_score,
            "aggregation_method": record.notes.split("Aggregation method: ")[-1].split(" ")[0]
            if "Aggregation method:" in record.notes
            else "",
            "per_model": {
                slug: {
                    "aggregate": r.aggregate_score,
                    "per_example": r.per_example,
                    "per_scenario": r.per_scenario,
                    "failed_ids": r.failed_ids,
                }
                for slug, r in per_model.items()
            },
            "combined_per_example": record.per_example,
            "combined_per_scenario": record.per_scenario,
            "failed_ids": record.failed_ids,
        },
    )
    atomic_write_json(
        d / "decision.json",
        {
            "decision": record.decision,
            "delta": record.delta,
            "timings": record.timings,
            "cost_usd": record.cost_usd,
            "timestamp": record.timestamp,
            "notes": record.notes,
        },
    )


def _initial_unified_prompt(task: TaskBundle, paths: RunPaths) -> str:
    for p in (paths.best_prompt, paths.current_prompt):
        if p.exists():
            text = p.read_text(encoding="utf-8")
            if text.strip():
                return text
    if task.seed_prompt.strip():
        return task.seed_prompt
    raise FileNotFoundError(
        f"No prompt found for unified run. Create '{task.root / 'prompt_seed.txt'}'."
    )


def _persist_unified_analysis(iteration_dir: Path, proposal) -> None:
    if not (proposal.analysis or proposal.rationale):
        return
    body = (
        "# Overseer analysis (Strategy C — unified)\n\n"
        f"{proposal.analysis.strip() or '(no analysis provided)'}\n\n"
        "## Hypothesis for next iteration\n\n"
        f"{proposal.rationale.strip() or '(no hypothesis provided)'}\n\n"
        "## Proposed next prompt\n\n"
        f"```\n{proposal.new_prompt.rstrip()}\n```\n"
    )
    atomic_write_text(iteration_dir / "overseer_analysis.md", body)


def _render_unified_report(
    *,
    task_name: str,
    best_score: float,
    iterations: int,
    stop_reason: StopReason,
    best_prompt: str,
    per_model_best: dict[str, float],
    method: str,
) -> str:
    rows = "\n".join(
        f"| `{slug}` | {score:.2f} |" for slug, score in sorted(per_model_best.items())
    )
    return (
        f"# Unified portable run — {task_name}\n\n"
        f"- **Combined best score ({method}):** {best_score:.2f}\n"
        f"- **Iterations:** {iterations}\n"
        f"- **Stop reason:** {stop_reason.value}\n\n"
        "## Per-model scores (at winner)\n\n"
        "| Model | Score |\n|---|---|\n"
        f"{rows}\n\n"
        f"## Shared winning prompt\n\n```\n{best_prompt}\n```\n"
    )


# ---------------------------------------------------------------------------
# Strategy dispatcher
# ---------------------------------------------------------------------------


async def run_multi(
    task: TaskBundle,
    *,
    cost: CostTracker,
    mutator_mode: str = "auto",
    iteration_callback: IterationCallback | None = None,
    user_gate_factory: UserGateFactory | None = None,
    initial_mode: str | None = None,
) -> MultiResult:
    """Dispatch by ``task.config.target_strategy``."""
    strategy = task.config.target_strategy
    if strategy == "parallel_independent":
        return await run_parallel_independent(
            task, cost=cost, mutator_mode=mutator_mode,
            iteration_callback=iteration_callback,
            user_gate_factory=user_gate_factory,
            initial_mode=initial_mode,
        )
    if strategy == "unified_portable":
        return await run_unified_portable(
            task, cost=cost, mutator_mode=mutator_mode,
            iteration_callback=iteration_callback,
            user_gate_factory=user_gate_factory,
            initial_mode=initial_mode,
        )
    raise ValueError(
        f"run_multi() is for Strategy B/C only; got strategy={strategy!r}. "
        "Use RatchetEngine directly for single-target runs."
    )


def validate_runtime(cfg: RunConfig) -> None:
    """Defence-in-depth runtime validation that complements Pydantic validators
    in the schema. Raises :class:`ValueError` with a user-facing message."""
    if cfg.target_strategy == "unified_portable" and cfg.unified_aggregation == "weighted_mean":
        for m in cfg.target_models:
            if m.weight is None or m.weight <= 0:
                raise ValueError(
                    f"unified_portable + weighted_mean requires positive 'weight' on every "
                    f"target_model. Missing or non-positive on slug={m.slug!r}."
                )
