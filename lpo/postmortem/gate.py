"""Decision gate: compare pre- and post-retry scores against the thresholds
defined in :class:`PostmortemConfig` and emit a :class:`Decision`.

Implements the AND-semantics thresholds from STAGE_8_DESIGN.md §7:

    global_delta           >= cfg.accept_threshold_global
    remediation_delta      >= cfg.accept_threshold_remediation
    max_scenario_regression <= cfg.regression_tolerance

All three must hold for ``accepted``. The gate treats ``metric_patch``
and ``eval_addition`` interventions as report-only (Apr 21 review):
they never participate in the threshold check — their IDs pass through
to ``Decision.report_only_intervention_ids`` for human follow-up, and
their presence alongside an accepted auto-applied intervention yields
``outcome='partial'`` rather than ``'accepted'``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from lpo.postmortem.artifacts import IterationScores
from lpo.postmortem.schemas import (
    Decision,
    DecisionDeltas,
    PostmortemConfig,
    PostmortemPlan,
)

log = logging.getLogger("lpo.postmortem.gate")


# ---------------------------------------------------------------------------
# Delta computation
# ---------------------------------------------------------------------------


@dataclass
class _DeltaInputs:
    """Compact view of the scores we compare. Lets tests feed either
    :class:`IterationScores` objects or hand-rolled dicts."""

    aggregate: float
    per_scenario: dict[str, float]
    per_example: dict[str, float]


def _to_inputs(scores: IterationScores | _DeltaInputs) -> _DeltaInputs:
    if isinstance(scores, _DeltaInputs):
        return scores
    return _DeltaInputs(
        aggregate=float(scores.aggregate),
        per_scenario=dict(scores.per_scenario),
        per_example=dict(scores.per_example),
    )


def compute_deltas(
    *,
    pre: IterationScores | _DeltaInputs,
    post: IterationScores | _DeltaInputs,
    cfg: PostmortemConfig,
    retry_iterations_run: int,
) -> DecisionDeltas:
    """Compute the three headline measurements.

    * **global_delta** — post.aggregate - pre.aggregate. The headline
      ratchet score.
    * **remediation_delta** — mean improvement across scenarios that
      scored <= ``cfg.failure_threshold`` pre-retry. Requires at least
      one such scenario; when there are none, falls back to ``global_delta``
      so the gate still has a meaningful signal (no failing scenarios to
      fix means any improvement is pure win).
    * **max_scenario_regression** — largest drop (0 or positive) on
      scenarios that scored >= ``cfg.success_threshold`` pre-retry. The
      Goodhart-guard: detects interventions that improve the failing
      subset at the cost of breaking what already worked.
    """
    pre_i = _to_inputs(pre)
    post_i = _to_inputs(post)

    global_delta = post_i.aggregate - pre_i.aggregate
    remediation_delta = _remediation_delta(pre_i, post_i, cfg.failure_threshold)
    # If no scenarios were failing pre-retry, the remediation axis is
    # vacuous; treat global_delta as the remediation signal.
    if remediation_delta is None:
        remediation_delta = global_delta
    max_regression = _max_scenario_regression(pre_i, post_i, cfg.success_threshold)

    return DecisionDeltas(
        global_delta=round(global_delta, 4),
        remediation_delta=round(remediation_delta, 4),
        max_scenario_regression=round(max_regression, 4),
        pre_best_score=round(pre_i.aggregate, 4),
        post_best_score=round(post_i.aggregate, 4),
        retry_iterations_run=retry_iterations_run,
    )


def _remediation_delta(
    pre: _DeltaInputs,
    post: _DeltaInputs,
    failure_threshold: float,
) -> float | None:
    """Mean improvement on scenarios that scored <= failure_threshold
    pre-retry. Scenarios missing from the post-retry scores count as
    score=0 (the prompt didn't produce anything we can score). Returns
    None when no scenarios qualified."""
    qualifying = [s for s, v in pre.per_scenario.items() if v <= failure_threshold]
    if not qualifying:
        return None
    deltas = []
    for s in qualifying:
        pre_v = pre.per_scenario.get(s, 0.0)
        post_v = post.per_scenario.get(s, 0.0)
        deltas.append(post_v - pre_v)
    return sum(deltas) / len(deltas)


def _max_scenario_regression(
    pre: _DeltaInputs,
    post: _DeltaInputs,
    success_threshold: float,
) -> float:
    """Largest drop (expressed as a positive number) on scenarios that
    scored >= success_threshold pre-retry. Returns 0.0 when there are no
    qualifying scenarios or no regressions."""
    qualifying = [s for s, v in pre.per_scenario.items() if v >= success_threshold]
    if not qualifying:
        return 0.0
    drops = []
    for s in qualifying:
        pre_v = pre.per_scenario.get(s, 0.0)
        post_v = post.per_scenario.get(s, 0.0)
        drops.append(max(0.0, pre_v - post_v))
    return max(drops) if drops else 0.0


# ---------------------------------------------------------------------------
# Decision emission
# ---------------------------------------------------------------------------


def thresholds_snapshot(cfg: PostmortemConfig) -> dict[str, float]:
    """Capture the threshold values used for this decision so the
    artifact on disk is self-describing even if config.yaml changes later."""
    return {
        "accept_threshold_global": cfg.accept_threshold_global,
        "accept_threshold_remediation": cfg.accept_threshold_remediation,
        "regression_tolerance": cfg.regression_tolerance,
        "failure_threshold": cfg.failure_threshold,
        "success_threshold": cfg.success_threshold,
        "min_confidence_prompt_patch": cfg.min_confidence_prompt_patch,
        "min_confidence_seed_reset": cfg.min_confidence_seed_reset,
    }


def decide_abstain(
    plan: PostmortemPlan,
    *,
    rationale: str,
    cfg: PostmortemConfig,
) -> Decision:
    """Emit an ``abstained`` decision when no focused retry was run.

    This happens when the plan contains no auto-applicable interventions
    (only ``metric_patch`` / ``eval_addition`` / ``model_swap_suggestion``)
    or when all auto-applicable interventions fell below their confidence
    floor.
    """
    return Decision(
        outcome="abstained",
        deltas=None,
        auto_applied_intervention_ids=[],
        report_only_intervention_ids=[i.id for i in plan.proposal.interventions],
        rationale=rationale,
        thresholds_snapshot=thresholds_snapshot(cfg),
    )


def decide_on_retry(
    plan: PostmortemPlan,
    *,
    deltas: DecisionDeltas,
    applied_ids: list[str],
    cfg: PostmortemConfig,
) -> Decision:
    """Emit an ``accepted`` / ``rejected`` / ``partial`` decision given
    the measured deltas. Never emits ``abstained`` — the caller should
    route through :func:`decide_abstain` for that path.
    """
    if not applied_ids:
        raise ValueError(
            "decide_on_retry called with no applied interventions. "
            "Use decide_abstain for the retry-skipped path."
        )

    passed_global = deltas.global_delta >= cfg.accept_threshold_global
    passed_remediation = deltas.remediation_delta >= cfg.accept_threshold_remediation
    passed_regression = deltas.max_scenario_regression <= cfg.regression_tolerance
    all_passed = passed_global and passed_remediation and passed_regression

    report_only_ids = [
        i.id for i in plan.proposal.interventions if not i.is_auto_applicable
    ]

    if all_passed:
        # 'partial' when there are report-only interventions that require
        # human follow-up alongside the accepted auto-applied change;
        # 'accepted' when the proposal was all-or-nothing auto-applicable.
        outcome = "partial" if report_only_ids else "accepted"
        rationale = _passed_rationale(deltas, cfg, outcome, report_only_ids)
    else:
        outcome = "rejected"
        rationale = _failed_rationale(
            deltas=deltas,
            cfg=cfg,
            passed_global=passed_global,
            passed_remediation=passed_remediation,
            passed_regression=passed_regression,
        )

    return Decision(
        outcome=outcome,
        deltas=deltas,
        auto_applied_intervention_ids=applied_ids,
        report_only_intervention_ids=report_only_ids,
        rationale=rationale,
        thresholds_snapshot=thresholds_snapshot(cfg),
    )


# ---------------------------------------------------------------------------
# Rationale composition — the string that ends up in decision.json and
# gets quoted into report.md. Operators will read this; it has to be
# actionable, not marketing.
# ---------------------------------------------------------------------------


def _passed_rationale(
    deltas: DecisionDeltas,
    cfg: PostmortemConfig,
    outcome: str,
    report_only_ids: list[str],
) -> str:
    lines = [
        f"Decision: {outcome}. All three thresholds met.",
        f"  global_delta           = {deltas.global_delta:+.2f} (>= {cfg.accept_threshold_global:.2f})",
        f"  remediation_delta      = {deltas.remediation_delta:+.2f} (>= {cfg.accept_threshold_remediation:.2f})",
        f"  max_scenario_regression = {deltas.max_scenario_regression:.2f} (<= {cfg.regression_tolerance:.2f})",
        f"  pre_best={deltas.pre_best_score:.2f}  post_best={deltas.post_best_score:.2f}",
    ]
    if report_only_ids:
        lines.append(
            f"Additional report-only interventions ({', '.join(report_only_ids)}) "
            "require human review. They were NOT auto-applied per Apr 21 review "
            "(metric patches, eval additions, model-swap suggestions are advisory)."
        )
    return "\n".join(lines)


def _failed_rationale(
    *,
    deltas: DecisionDeltas,
    cfg: PostmortemConfig,
    passed_global: bool,
    passed_remediation: bool,
    passed_regression: bool,
) -> str:
    failures = []
    if not passed_global:
        failures.append(
            f"global_delta = {deltas.global_delta:+.2f} < {cfg.accept_threshold_global:.2f}"
        )
    if not passed_remediation:
        failures.append(
            f"remediation_delta = {deltas.remediation_delta:+.2f} < {cfg.accept_threshold_remediation:.2f}"
        )
    if not passed_regression:
        failures.append(
            f"max_scenario_regression = {deltas.max_scenario_regression:.2f} > {cfg.regression_tolerance:.2f}"
        )
    lines = [
        "Decision: rejected. One or more thresholds not met (AND-semantics):",
        *[f"  {f}" for f in failures],
        f"  pre_best={deltas.pre_best_score:.2f}  post_best={deltas.post_best_score:.2f}",
        "",
        "The original winner is unchanged. proposal.md and diagnosis.json "
        "remain on disk; the operator may inspect and apply manually.",
    ]
    return "\n".join(lines)
