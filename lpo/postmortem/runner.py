"""Top-level postmortem orchestrator.

This is the function the MCP tool and the CLI call. It:

1. Loads the completed run (:func:`load_run_history`).
2. Calls the Analyst to produce a validated :class:`PostmortemPlan`.
3. In ``propose_only`` mode: writes ``diagnosis.json`` + ``proposal.md``
   and returns ``Decision(outcome='abstained')``. No retry, no disk
   modification to the winner.
4. In ``autonomous`` mode: selects auto-applicable interventions
   (confidence-filtered), applies them to the current best prompt,
   runs :func:`run_focused_retry`, computes deltas, emits a
   :class:`Decision`, and — when ``accepted`` or ``partial`` — promotes
   the retry's prompt to the run's winner with ``source: "postmortem"``
   provenance (leaving the original winner in ``winner.pre_postmortem/``
   for rollback).

All postmortem artifacts land under ``runs/<slug>/postmortem/``:

    postmortem/
    ├── diagnosis.json            # the Analyst's structured findings
    ├── proposal.md               # human-readable intervention list
    ├── proposal.json             # machine-readable interventions
    ├── retry/                    # focused-retry artifacts (iter_0001/, prompt.txt)
    ├── decision.json             # the Decision object
    └── report.md                 # combined operator-facing report
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from lpo.config.schema import TargetModelConfig
from lpo.core.cost import CostTracker
from lpo.postmortem.analyst import AnalystClient, AnalystResult, run_analyst
from lpo.postmortem.artifacts import (
    IterationScores,
    RunHistoryBundle,
    load_run_history,
)
from lpo.postmortem.gate import (
    compute_deltas,
    decide_abstain,
    decide_on_retry,
    thresholds_snapshot,
)
from lpo.postmortem.patches import PatchSelection, select_and_apply
from lpo.postmortem.retry import FocusedRetryResult, run_focused_retry
from lpo.postmortem.schemas import (
    Decision,
    PostmortemConfig,
    PostmortemPlan,
)

log = logging.getLogger("lpo.postmortem.runner")


# ---------------------------------------------------------------------------
# Result aggregate
# ---------------------------------------------------------------------------


PostmortemMode = Literal["autonomous", "propose_only"]


@dataclass
class PostmortemResult:
    """Everything :func:`run_postmortem` produces, in one object.

    Distinct from :class:`Decision` because the result also carries the
    raw plan and retry object that downstream tools (CLI --verbose, MCP
    tool) may want to surface without re-reading from disk.
    """

    decision: Decision
    plan: PostmortemPlan
    retry: FocusedRetryResult | None
    postmortem_root: Path
    mode: PostmortemMode
    analyst_retries: int
    analyst_model_id: str
    total_cost_usd: float


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def run_postmortem(
    task_root: Path | str,
    *,
    slug: str,
    analyst_client: AnalystClient,
    cfg: PostmortemConfig | None = None,
    mode: PostmortemMode = "autonomous",
    target_cfg: TargetModelConfig | None = None,
    cost: CostTracker | None = None,
    retry_runner_factory: Any | None = None,
    allow_on_cost_cap: bool = False,
    analyst_model_id: str | None = None,
) -> PostmortemResult:
    """Run the full Stage-8 postmortem phase for ``(task_root, slug)``.

    Arguments:
        task_root: The task bundle directory.
        slug: Target slug whose run should be analyzed.
        analyst_client: Frontier-model client matching :class:`AnalystClient`.
        cfg: Tuning. Defaults to the PostmortemConfig carried on the
            loaded ``TaskBundle.config.postmortem``.
        mode: ``autonomous`` runs the full pipeline; ``propose_only``
            stops after the Analyst and writes diagnosis+proposal only.
        target_cfg: Which :class:`TargetModelConfig` to use for the
            focused retry. Defaults to the slug's config on the task
            bundle. Required if the bundle has no matching slug.
        cost: Shared :class:`CostTracker`. One is created when None;
            pass the same instance the MCP tool exposes so the
            ``$postmortem.cost_cap_usd`` budget applies correctly.
        retry_runner_factory: Test seam for injecting a stub
            :class:`IterationRunner`. Production callers leave this None.
        allow_on_cost_cap: Whether to run even if the main ratchet
            terminated on ``cost_cap``. Defaults to False per design §9.
        analyst_model_id: Overrides the config default for the
            Analyst's model identity stamp.

    Never raises on a *semantic* failure (rejected proposal, abstained
    decision); those are represented by :class:`PostmortemResult`. It
    may raise on infrastructure failures (missing task, Analyst
    unrecoverable, focused retry exception).
    """
    bundle = load_run_history(task_root, slug)
    task = bundle.task
    cfg = cfg or task.config.postmortem
    cost = cost or CostTracker()

    postmortem_root = bundle.run_root / "postmortem"
    postmortem_root.mkdir(parents=True, exist_ok=True)

    # --- Analyst ---------------------------------------------------------
    analyst_result = await run_analyst(
        bundle,
        cfg=cfg,
        client=analyst_client,
        client_model_id=analyst_model_id,
    )
    plan = analyst_result.plan

    _write_diagnosis(postmortem_root, plan)
    _write_proposal(postmortem_root, plan)

    # --- Propose-only short-circuit -------------------------------------
    if mode == "propose_only":
        decision = decide_abstain(
            plan,
            rationale=(
                "mode=propose_only — Analyst produced a diagnosis + proposal "
                "but no focused retry was run. Operator may invoke the "
                "postmortem again with mode=autonomous to validate and "
                "auto-commit the auto-applicable interventions."
            ),
            cfg=cfg,
        )
        _write_decision(postmortem_root, decision)
        _write_report(
            postmortem_root,
            plan=plan,
            decision=decision,
            retry=None,
            analyst_result=analyst_result,
            mode=mode,
        )
        return PostmortemResult(
            decision=decision,
            plan=plan,
            retry=None,
            postmortem_root=postmortem_root,
            mode=mode,
            analyst_retries=analyst_result.retries,
            analyst_model_id=analyst_result.model_id,
            total_cost_usd=cost.total_usd,
        )

    # --- Select + apply auto-interventions ------------------------------
    current_best = bundle.winner_prompt or _fallback_best(bundle)
    selection: PatchSelection = select_and_apply(
        plan, current_best_prompt=current_best, cfg=cfg
    )

    if selection.patched_prompt is None:
        # No intervention cleared the confidence floor. All findings go
        # into the report-only bucket.
        rationale_bits = [
            "No auto-applicable intervention cleared its confidence floor.",
        ]
        if selection.skipped_low_confidence_ids:
            rationale_bits.append(
                "Skipped (below floor): "
                + ", ".join(selection.skipped_low_confidence_ids)
            )
        if selection.report_only_intervention_ids:
            rationale_bits.append(
                "Always report-only per Apr 21 review: "
                + ", ".join(selection.report_only_intervention_ids)
            )
        decision = decide_abstain(
            plan,
            rationale=" ".join(rationale_bits),
            cfg=cfg,
        )
        _write_decision(postmortem_root, decision)
        _write_report(
            postmortem_root,
            plan=plan,
            decision=decision,
            retry=None,
            analyst_result=analyst_result,
            mode=mode,
        )
        return PostmortemResult(
            decision=decision,
            plan=plan,
            retry=None,
            postmortem_root=postmortem_root,
            mode=mode,
            analyst_retries=analyst_result.retries,
            analyst_model_id=analyst_result.model_id,
            total_cost_usd=cost.total_usd,
        )

    # --- Focused retry --------------------------------------------------
    target = target_cfg or _resolve_target_cfg(task, slug)
    retry_result = await run_focused_retry(
        task,
        slug=slug,
        target_cfg=target,
        patched_prompt=selection.patched_prompt,
        cfg=cfg,
        cost=cost,
        runner_factory=retry_runner_factory,
    )

    # --- Decision gate --------------------------------------------------
    pre_scores = _pre_scores(bundle)
    post_scores = retry_result.iterations[0].record if retry_result.iterations else None
    if post_scores is None:
        # Retry produced no iterations — treat as rejected.
        decision = Decision(
            outcome="rejected",
            deltas=_zero_deltas(pre_scores),
            auto_applied_intervention_ids=selection.applied_intervention_ids,
            report_only_intervention_ids=selection.report_only_intervention_ids,
            rationale="Focused retry produced no iteration record. No score delta to measure.",
            thresholds_snapshot=thresholds_snapshot(cfg),
        )
    else:
        post_iteration_scores = IterationScores(
            aggregate=post_scores.aggregate_score,
            per_example=post_scores.per_example,
            per_scenario=post_scores.per_scenario,
            failed_ids=post_scores.failed_ids,
        )
        deltas = compute_deltas(
            pre=pre_scores,
            post=post_iteration_scores,
            cfg=cfg,
            retry_iterations_run=len(retry_result.iterations),
        )
        decision = decide_on_retry(
            plan,
            deltas=deltas,
            applied_ids=selection.applied_intervention_ids,
            cfg=cfg,
        )

    # --- Promote retry winner on accept/partial -------------------------
    if decision.outcome in ("accepted", "partial"):
        _promote_retry_winner(bundle.run_root, retry_result, plan)

    _write_decision(postmortem_root, decision)
    _write_report(
        postmortem_root,
        plan=plan,
        decision=decision,
        retry=retry_result,
        analyst_result=analyst_result,
        mode=mode,
    )

    return PostmortemResult(
        decision=decision,
        plan=plan,
        retry=retry_result,
        postmortem_root=postmortem_root,
        mode=mode,
        analyst_retries=analyst_result.retries,
        analyst_model_id=analyst_result.model_id,
        total_cost_usd=cost.total_usd,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_target_cfg(task, slug: str) -> TargetModelConfig:
    for m in task.config.target_models:
        if m.slug == slug:
            return m
    raise ValueError(
        f"No target_model with slug={slug!r} in {task.root / 'config.yaml'}. "
        "Pass target_cfg explicitly if the bundle's config doesn't carry it."
    )


def _fallback_best(bundle: RunHistoryBundle) -> str:
    """When no winner/ exists yet, use the best iteration's prompt."""
    best = bundle.best_iteration
    if best is not None and best.prompt.strip():
        return best.prompt
    return bundle.task.seed_prompt


def _pre_scores(bundle: RunHistoryBundle) -> IterationScores:
    """Score-block representing the pre-postmortem state. Uses the best
    iteration's scores (what the ratchet's winner represents)."""
    best = bundle.best_iteration
    if best is not None:
        return best.scores
    # Shouldn't happen in practice — a run with no iterations can't
    # produce a postmortem target — but guard defensively.
    return IterationScores(aggregate=0.0, per_example={}, per_scenario={}, failed_ids=[])


def _zero_deltas(pre: IterationScores):
    from lpo.postmortem.schemas import DecisionDeltas
    return DecisionDeltas(
        global_delta=0.0,
        remediation_delta=0.0,
        max_scenario_regression=0.0,
        pre_best_score=pre.aggregate,
        post_best_score=pre.aggregate,
        retry_iterations_run=0,
    )


# ---------------------------------------------------------------------------
# Artifact writers
# ---------------------------------------------------------------------------


def _write_diagnosis(postmortem_root: Path, plan: PostmortemPlan) -> None:
    (postmortem_root / "diagnosis.json").write_text(
        plan.diagnosis.model_dump_json(indent=2),
        encoding="utf-8",
    )


def _write_proposal(postmortem_root: Path, plan: PostmortemPlan) -> None:
    (postmortem_root / "proposal.json").write_text(
        plan.proposal.model_dump_json(indent=2),
        encoding="utf-8",
    )
    (postmortem_root / "proposal.md").write_text(
        _render_proposal_md(plan),
        encoding="utf-8",
    )


def _write_decision(postmortem_root: Path, decision: Decision) -> None:
    (postmortem_root / "decision.json").write_text(
        decision.model_dump_json(indent=2),
        encoding="utf-8",
    )


def _write_report(
    postmortem_root: Path,
    *,
    plan: PostmortemPlan,
    decision: Decision,
    retry: FocusedRetryResult | None,
    analyst_result: AnalystResult,
    mode: PostmortemMode,
) -> None:
    (postmortem_root / "report.md").write_text(
        _render_report_md(
            plan=plan,
            decision=decision,
            retry=retry,
            analyst_result=analyst_result,
            mode=mode,
        ),
        encoding="utf-8",
    )


def _render_proposal_md(plan: PostmortemPlan) -> str:
    lines = [
        "# Postmortem proposal",
        "",
        f"**Rationale.** {plan.proposal.rationale}",
        "",
        "## Interventions",
        "",
    ]
    for i in plan.proposal.interventions:
        auto = "auto-applicable" if i.is_auto_applicable else "report-only (human approval required)"
        lines.extend([
            f"### {i.id} — {i.type} ({auto})",
            "",
            f"- **Fixes:** {', '.join(i.fixes)}",
            f"- **Confidence:** {i.confidence:.2f}",
            f"- **Summary:** {i.summary}",
        ])
        if i.expected_impact is not None:
            glob = i.expected_impact.global_
            remed = i.expected_impact.remediation
            if glob is not None or remed is not None:
                parts = []
                if glob is not None:
                    parts.append(f"global {glob[0]:+.1f} to {glob[1]:+.1f}")
                if remed is not None:
                    parts.append(f"remediation {remed[0]:+.1f} to {remed[1]:+.1f}")
                lines.append(f"- **Expected impact:** {'; '.join(parts)}")
        lines.extend([
            "",
            "```json",
            json.dumps(i.patch, indent=2, ensure_ascii=False),
            "```",
            "",
        ])
    if plan.proposal.human_review_summary:
        lines.extend([
            "## Human review summary",
            "",
            plan.proposal.human_review_summary,
            "",
        ])
    return "\n".join(lines)


def _render_report_md(
    *,
    plan: PostmortemPlan,
    decision: Decision,
    retry: FocusedRetryResult | None,
    analyst_result: AnalystResult,
    mode: PostmortemMode,
) -> str:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = [
        f"# Postmortem report — {plan.diagnosis.task_id} / {plan.diagnosis.slug}",
        "",
        f"- **Generated:** {now}",
        f"- **Mode:** {mode}",
        f"- **Analyst model:** {analyst_result.model_id}  (retries: {analyst_result.retries})",
        f"- **Outcome:** **{decision.outcome}**",
        "",
        "## Decision rationale",
        "",
        "```",
        decision.rationale,
        "```",
        "",
        "## Findings",
        "",
    ]
    for f in plan.diagnosis.findings:
        lines.extend([
            f"### {f.id} — {f.type} (severity: {f.severity}, confidence: {f.confidence:.2f})",
            "",
            f"{f.summary}",
            "",
            f"**Evidence.** iterations={f.evidence.iterations}, "
            f"scenarios={f.evidence.scenarios or 'n/a'}, "
            f"examples={f.evidence.example_ids}",
            "",
            f"**Root cause hypothesis.** {f.root_cause_hypothesis}",
            "",
        ])
        if f.differential_evidence:
            lines.extend([f"**Differential evidence.** {f.differential_evidence}", ""])

    lines.extend(["## Interventions", ""])
    for i in plan.proposal.interventions:
        auto = "auto-applicable" if i.is_auto_applicable else "report-only"
        lines.append(
            f"- **{i.id}** ({i.type}, {auto}, confidence={i.confidence:.2f}) "
            f"— fixes {', '.join(i.fixes)} — {i.summary}"
        )

    if decision.auto_applied_intervention_ids:
        lines.extend([
            "",
            "## Auto-applied",
            "",
            "The following interventions were applied to `prompt.txt.best`:",
            "",
        ])
        for iid in decision.auto_applied_intervention_ids:
            lines.append(f"- {iid}")

    if decision.report_only_intervention_ids:
        lines.extend([
            "",
            "## Awaiting human review",
            "",
            "Per Apr 21 review, these intervention types are never auto-committed. "
            "Inspect and apply manually if appropriate:",
            "",
        ])
        for iid in decision.report_only_intervention_ids:
            lines.append(f"- {iid}")

    if retry is not None:
        lines.extend([
            "",
            "## Focused retry",
            "",
            f"- **Iterations run:** {len(retry.iterations)}",
            f"- **Post-retry best score:** {retry.best_score:.2f}",
            f"- **Retry cost:** ${retry.cost_usd:.4f}",
        ])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Winner promotion — accept / partial only
# ---------------------------------------------------------------------------


def _promote_retry_winner(
    run_root: Path,
    retry: FocusedRetryResult,
    plan: PostmortemPlan,
) -> None:
    """Move the old winner aside and install the retry's patched prompt
    as the new one, with provenance metadata so a future postmortem can
    detect stacking (and so rollback is a directory rename away)."""
    old_winner = run_root / "winner"
    rollback = run_root / "winner.pre_postmortem"
    if old_winner.exists():
        # Blow away any stale rollback from a previous postmortem; we
        # only preserve the immediate pre-postmortem state.
        if rollback.exists():
            shutil.rmtree(rollback)
        old_winner.rename(rollback)

    old_winner.mkdir(parents=True, exist_ok=True)
    (old_winner / "prompt.txt").write_text(retry.patched_prompt, encoding="utf-8")
    (old_winner / "provenance.json").write_text(
        json.dumps({
            "source": "postmortem",
            "auto_applied_intervention_ids": [
                i.id for i in plan.proposal.interventions
                if i.is_auto_applicable
            ],
            "finding_ids": [f.id for f in plan.diagnosis.findings],
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "rollback_directory": "winner.pre_postmortem",
        }, indent=2),
        encoding="utf-8",
    )
    # Also update the run's current-best prompt so subsequent main-ratchet
    # runs start from the postmortem-patched version.
    best_prompt_path = run_root / "prompt.txt.best"
    best_prompt_path.write_text(retry.patched_prompt, encoding="utf-8")
