"""Decision-gate tests: delta computation + outcome selection."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from lpo.postmortem.artifacts import IterationScores
from lpo.postmortem.gate import (
    compute_deltas,
    decide_abstain,
    decide_on_retry,
    thresholds_snapshot,
)
from lpo.postmortem.schemas import (
    Diagnosis,
    Evidence,
    Finding,
    Intervention,
    PostmortemConfig,
    PostmortemPlan,
    Proposal,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _scores(aggregate: float, per_scenario: dict[str, float]) -> IterationScores:
    # per_example mirrors per_scenario for the tests' purposes.
    return IterationScores(
        aggregate=aggregate,
        per_example={f"ex_{k}": v for k, v in per_scenario.items()},
        per_scenario=per_scenario,
        failed_ids=[],
    )


def _finding() -> Finding:
    return Finding(
        id="F1",
        type="scenario_blindspot",
        severity="high",
        confidence=0.9,
        summary="s",
        evidence=Evidence(
            iterations=[1],
            example_ids=["ex001"],
            score_breakdown={"ex001": {"iter_1": 0.0}},
        ),
        root_cause_hypothesis="h",
    )


def _plan(*, with_report_only: bool = False) -> PostmortemPlan:
    interventions = [
        Intervention(
            id="I1",
            type="prompt_patch",
            fixes=["F1"],
            confidence=0.8,
            summary="s",
            patch={"mode": "append", "content": "- rule"},
        ),
    ]
    if with_report_only:
        interventions.append(
            Intervention(
                id="I2",
                type="metric_patch",
                fixes=["F1"],
                confidence=0.9,
                summary="s",
                patch={"rationale": "r"},
            )
        )
    return PostmortemPlan(
        diagnosis=Diagnosis(
            findings=[_finding()],
            analyst_model_id="x",
            task_id="t",
            slug="s",
        ),
        proposal=Proposal(interventions=interventions, rationale="r"),
    )


# ---------------------------------------------------------------------------
# compute_deltas — the three headline measurements.
# ---------------------------------------------------------------------------


class TestComputeDeltas:
    def test_global_delta_and_remediation_delta_match_definition(self):
        cfg = PostmortemConfig()  # failure 20, success 70
        pre = _scores(40.0, {"easy": 80.0, "hard": 0.0})
        post = _scores(60.0, {"easy": 80.0, "hard": 40.0})
        deltas = compute_deltas(pre=pre, post=post, cfg=cfg, retry_iterations_run=1)
        assert deltas.global_delta == pytest.approx(20.0)
        # 'hard' was the only failing scenario pre; improved 0 -> 40.
        assert deltas.remediation_delta == pytest.approx(40.0)
        # 'easy' was passing and stayed the same — no regression.
        assert deltas.max_scenario_regression == pytest.approx(0.0)
        assert deltas.retry_iterations_run == 1
        assert deltas.pre_best_score == pytest.approx(40.0)
        assert deltas.post_best_score == pytest.approx(60.0)

    def test_remediation_averages_across_multiple_failing_scenarios(self):
        cfg = PostmortemConfig()
        pre = _scores(20.0, {"a": 10.0, "b": 5.0, "c": 90.0})
        post = _scores(60.0, {"a": 50.0, "b": 25.0, "c": 90.0})
        deltas = compute_deltas(pre=pre, post=post, cfg=cfg, retry_iterations_run=1)
        # a improved by 40, b improved by 20 — mean = 30.
        assert deltas.remediation_delta == pytest.approx(30.0)

    def test_regression_detected_on_previously_passing_scenario(self):
        # 'easy' was passing (>= 70) pre; dropped to 55 post. Regression = 25.
        cfg = PostmortemConfig()
        pre = _scores(80.0, {"easy": 90.0, "hard": 0.0})
        post = _scores(60.0, {"easy": 55.0, "hard": 30.0})
        deltas = compute_deltas(pre=pre, post=post, cfg=cfg, retry_iterations_run=1)
        assert deltas.max_scenario_regression == pytest.approx(35.0)

    def test_remediation_falls_back_to_global_when_no_failing_scenarios(self):
        # Every scenario pre >= failure_threshold, so the remediation axis
        # is vacuous. The gate falls back to global_delta so it still has
        # a signal to compare against accept_threshold_remediation (which
        # is a safety net — an accept-worthy improvement on already-passing
        # scenarios is still accept-worthy).
        cfg = PostmortemConfig()
        pre = _scores(60.0, {"a": 60.0, "b": 70.0})
        post = _scores(80.0, {"a": 80.0, "b": 80.0})
        deltas = compute_deltas(pre=pre, post=post, cfg=cfg, retry_iterations_run=1)
        assert deltas.remediation_delta == pytest.approx(deltas.global_delta)

    def test_no_previously_passing_scenarios_zero_regression(self):
        cfg = PostmortemConfig()
        pre = _scores(10.0, {"a": 0.0, "b": 10.0})
        post = _scores(5.0, {"a": 0.0, "b": 5.0})
        deltas = compute_deltas(pre=pre, post=post, cfg=cfg, retry_iterations_run=1)
        # b dropped 5 but was never >= success_threshold; doesn't count.
        assert deltas.max_scenario_regression == 0.0


# ---------------------------------------------------------------------------
# decide_abstain
# ---------------------------------------------------------------------------


class TestDecideAbstain:
    def test_abstain_returns_no_auto_applied(self):
        cfg = PostmortemConfig()
        decision = decide_abstain(_plan(), rationale="no candidates", cfg=cfg)
        assert decision.outcome == "abstained"
        assert decision.deltas is None
        assert decision.auto_applied_intervention_ids == []
        assert decision.rationale == "no candidates"

    def test_abstain_lists_all_proposal_interventions_as_report_only(self):
        cfg = PostmortemConfig()
        decision = decide_abstain(_plan(with_report_only=True), rationale="r", cfg=cfg)
        # Both I1 (prompt_patch, wasn't applied) and I2 (metric_patch, never
        # applicable) are surfaced for human follow-up.
        assert set(decision.report_only_intervention_ids) == {"I1", "I2"}

    def test_thresholds_snapshot_captured(self):
        cfg = PostmortemConfig(accept_threshold_global=7.5)
        decision = decide_abstain(_plan(), rationale="r", cfg=cfg)
        assert decision.thresholds_snapshot["accept_threshold_global"] == 7.5


# ---------------------------------------------------------------------------
# decide_on_retry — AND-semantics on thresholds
# ---------------------------------------------------------------------------


class TestDecideOnRetry:
    def _deltas(self, *, g=10.0, r=20.0, regression=1.0, pre=50.0, post=60.0):
        from lpo.postmortem.schemas import DecisionDeltas
        return DecisionDeltas(
            global_delta=g,
            remediation_delta=r,
            max_scenario_regression=regression,
            pre_best_score=pre,
            post_best_score=post,
            retry_iterations_run=1,
        )

    def test_all_thresholds_pass_accepted_without_report_only(self):
        cfg = PostmortemConfig()  # global>=5, remediation>=15, regression<=3
        decision = decide_on_retry(
            _plan(),
            deltas=self._deltas(),
            applied_ids=["I1"],
            cfg=cfg,
        )
        assert decision.outcome == "accepted"
        assert decision.auto_applied_intervention_ids == ["I1"]
        assert "accepted" in decision.rationale.lower()

    def test_all_thresholds_pass_with_report_only_yields_partial(self):
        cfg = PostmortemConfig()
        decision = decide_on_retry(
            _plan(with_report_only=True),
            deltas=self._deltas(),
            applied_ids=["I1"],
            cfg=cfg,
        )
        assert decision.outcome == "partial"
        assert "I2" in decision.report_only_intervention_ids

    def test_global_below_threshold_rejected(self):
        cfg = PostmortemConfig()
        decision = decide_on_retry(
            _plan(),
            deltas=self._deltas(g=3.0),  # below 5
            applied_ids=["I1"],
            cfg=cfg,
        )
        assert decision.outcome == "rejected"
        assert "global_delta" in decision.rationale

    def test_remediation_below_threshold_rejected(self):
        cfg = PostmortemConfig()
        decision = decide_on_retry(
            _plan(),
            deltas=self._deltas(r=10.0),  # below 15
            applied_ids=["I1"],
            cfg=cfg,
        )
        assert decision.outcome == "rejected"
        assert "remediation_delta" in decision.rationale

    def test_regression_over_tolerance_rejected(self):
        cfg = PostmortemConfig()
        decision = decide_on_retry(
            _plan(),
            deltas=self._deltas(regression=10.0),  # above 3
            applied_ids=["I1"],
            cfg=cfg,
        )
        assert decision.outcome == "rejected"
        assert "max_scenario_regression" in decision.rationale

    def test_requires_at_least_one_applied_intervention(self):
        # The Decision schema enforces this; the gate enforces it earlier
        # with a clearer error so the orchestrator has a single-call path.
        cfg = PostmortemConfig()
        with pytest.raises(ValueError):
            decide_on_retry(_plan(), deltas=self._deltas(), applied_ids=[], cfg=cfg)

    def test_rejected_rationale_lists_every_failed_threshold(self):
        # All three failed simultaneously — rationale should mention all of them
        # so the operator can see the full picture at a glance.
        cfg = PostmortemConfig()
        decision = decide_on_retry(
            _plan(),
            deltas=self._deltas(g=0.0, r=0.0, regression=100.0),
            applied_ids=["I1"],
            cfg=cfg,
        )
        assert decision.outcome == "rejected"
        for token in ("global_delta", "remediation_delta", "max_scenario_regression"):
            assert token in decision.rationale


# ---------------------------------------------------------------------------
# thresholds_snapshot — round-trip completeness.
# ---------------------------------------------------------------------------


def test_thresholds_snapshot_includes_every_gate_field():
    cfg = PostmortemConfig()
    snap = thresholds_snapshot(cfg)
    for key in (
        "accept_threshold_global",
        "accept_threshold_remediation",
        "regression_tolerance",
        "failure_threshold",
        "success_threshold",
        "min_confidence_prompt_patch",
        "min_confidence_seed_reset",
    ):
        assert key in snap
