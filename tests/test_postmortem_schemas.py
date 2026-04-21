"""Stage 8 — Postmortem schema tests.

The Apr 21 design review made two invariants non-negotiable: every finding
must cite concrete evidence, and every intervention must reference the
finding IDs it addresses. These tests pin those invariants to the type
layer so a future refactor can't silently relax them.

We also cover the metric-patch safety rule (only prompt_patch and
seed_reset are auto-applicable), the differential-evidence requirement
for the two most hallucination-prone finding types, and the
cross-referential check that ties Intervention.fixes back to real
Findings at the PostmortemPlan level.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from lpo.postmortem.schemas import (
    Decision,
    DecisionDeltas,
    Diagnosis,
    Evidence,
    ExpectedImpact,
    Finding,
    Intervention,
    PostmortemConfig,
    PostmortemPlan,
    Proposal,
)


# ---------------------------------------------------------------------------
# Builders — reused across tests to keep them readable.
# ---------------------------------------------------------------------------


def _evidence(
    iterations: list[int] | None = None,
    example_ids: list[str] | None = None,
    scenarios: list[str] | None = None,
    breakdown: dict[str, dict[str, float]] | None = None,
) -> Evidence:
    iterations = iterations if iterations is not None else [1, 2]
    example_ids = example_ids if example_ids is not None else ["ex001"]
    scenarios = scenarios if scenarios is not None else []
    breakdown = breakdown if breakdown is not None else {
        "ex001": {"iter_1": 0.0, "iter_2": 0.0}
    }
    return Evidence(
        iterations=iterations,
        example_ids=example_ids,
        scenarios=scenarios,
        score_breakdown=breakdown,
    )


def _finding(
    id_: str = "F1",
    type_: str = "scenario_blindspot",
    evidence: Evidence | None = None,
    differential: str | None = None,
) -> Finding:
    kwargs: dict = dict(
        id=id_,
        type=type_,
        severity="high",
        confidence=0.9,
        summary="A summary",
        evidence=evidence or _evidence(),
        root_cause_hypothesis="Because reasons.",
    )
    if differential is not None:
        kwargs["differential_evidence"] = differential
    return Finding(**kwargs)


def _intervention(
    id_: str = "I1",
    type_: str = "prompt_patch",
    fixes: list[str] | None = None,
    patch: dict | None = None,
) -> Intervention:
    if fixes is None:
        fixes = ["F1"]
    if patch is None:
        if type_ == "prompt_patch":
            patch = {"mode": "append", "content": "- New rule."}
        elif type_ == "seed_reset":
            patch = {"new_seed": "New prompt body."}
        elif type_ == "metric_patch":
            patch = {"rationale": "Metric needs a content-escaping rule."}
        elif type_ == "eval_addition":
            patch = {"new_examples": [{"input": "x", "expected_output": "y"}]}
        elif type_ == "model_swap_suggestion":
            patch = {"suggested_models": ["anthropic/claude-opus-4.6"], "rationale": "r"}
    return Intervention(
        id=id_,
        type=type_,
        fixes=fixes,
        confidence=0.8,
        summary="A summary",
        patch=patch,
    )


# ---------------------------------------------------------------------------
# Evidence invariant — the April 21 review's single biggest requirement.
# ---------------------------------------------------------------------------


class TestEvidenceInvariant:
    def test_empty_iterations_rejected(self):
        with pytest.raises(ValidationError):
            Evidence(
                iterations=[],
                example_ids=["ex001"],
                score_breakdown={"ex001": {"iter_1": 0.0}},
            )

    def test_empty_example_ids_rejected(self):
        with pytest.raises(ValidationError):
            Evidence(
                iterations=[1],
                example_ids=[],
                score_breakdown={"ex001": {"iter_1": 0.0}},
            )

    def test_empty_score_breakdown_rejected(self):
        with pytest.raises(ValidationError):
            Evidence(
                iterations=[1],
                example_ids=["ex001"],
                score_breakdown={},
            )

    def test_score_breakdown_must_reference_listed_examples(self):
        # Citing ex999 in the breakdown but not in example_ids is the exact
        # free-floating-narrative pattern the review targeted.
        with pytest.raises(ValidationError) as exc:
            Evidence(
                iterations=[1],
                example_ids=["ex001"],
                score_breakdown={"ex999": {"iter_1": 0.0}},
            )
        assert "example_ids" in str(exc.value)

    def test_score_breakdown_labels_must_match_iterations(self):
        # Iter label referring to an iteration not listed in iterations[].
        with pytest.raises(ValidationError) as exc:
            Evidence(
                iterations=[1, 2],
                example_ids=["ex001"],
                score_breakdown={"ex001": {"iter_1": 0.0, "iter_5": 0.0}},
            )
        assert "iter_" in str(exc.value)

    def test_zero_or_negative_iterations_rejected(self):
        with pytest.raises(ValidationError):
            Evidence(
                iterations=[0],
                example_ids=["ex001"],
                score_breakdown={"ex001": {"iter_0": 0.0}},
            )

    def test_happy_path(self):
        # Well-formed evidence must construct without complaint.
        ev = Evidence(
            iterations=[1, 2, 3],
            example_ids=["ex001", "ex008"],
            scenarios=["content_special_chars"],
            score_breakdown={
                "ex001": {"iter_1": 0.0, "iter_2": 0.0, "iter_3": 0.0},
                "ex008": {"iter_1": 0.0, "iter_3": 0.0},
            },
        )
        assert ev.iterations == [1, 2, 3]
        assert set(ev.example_ids) == {"ex001", "ex008"}


# ---------------------------------------------------------------------------
# Finding — differential-evidence rule for hallucination-prone types.
# ---------------------------------------------------------------------------


class TestFinding:
    def test_id_shape_enforced(self):
        with pytest.raises(ValidationError):
            _finding(id_="finding-1")

    def test_metric_mismatch_requires_differential_evidence(self):
        # These two types are the most prone to the Analyst blaming the
        # measurement/model when the real problem is the prompt. The schema
        # makes the claim cost something.
        with pytest.raises(ValidationError) as exc:
            _finding(type_="metric_mismatch", differential=None)
        assert "differential_evidence" in str(exc.value)

    def test_model_fit_issue_requires_differential_evidence(self):
        with pytest.raises(ValidationError):
            _finding(type_="model_fit_issue", differential=None)

    def test_other_types_dont_require_differential_evidence(self):
        # A scenario_blindspot without differential_evidence is fine —
        # the evidence invariant alone is sufficient to make the claim
        # debuggable.
        f = _finding(type_="scenario_blindspot", differential=None)
        assert f.differential_evidence is None

    def test_metric_mismatch_happy_path_with_differential(self):
        f = _finding(
            type_="metric_mismatch",
            differential="Example ex004 scored 0.85 but the output is wrong.",
        )
        assert f.type == "metric_mismatch"


# ---------------------------------------------------------------------------
# Intervention — provenance invariant + auto-applicability rule.
# ---------------------------------------------------------------------------


class TestIntervention:
    def test_empty_fixes_rejected(self):
        with pytest.raises(ValidationError) as exc:
            Intervention(
                id="I1",
                type="prompt_patch",
                fixes=[],
                confidence=0.8,
                summary="s",
                patch={"mode": "append", "content": "x"},
            )
        assert "fixes" in str(exc.value)

    def test_fixes_must_look_like_finding_ids(self):
        # A stray free-form string in fixes is the Analyst hallucinating
        # a reference. Catch it at the schema.
        with pytest.raises(ValidationError):
            _intervention(fixes=["the first finding"])

    def test_id_shape_enforced(self):
        with pytest.raises(ValidationError):
            _intervention(id_="intervention-1")

    @pytest.mark.parametrize(
        "type_,auto",
        [
            ("prompt_patch", True),
            ("seed_reset", True),
            ("metric_patch", False),
            ("eval_addition", False),
            ("model_swap_suggestion", False),
        ],
    )
    def test_auto_applicability_matches_apr_21_review(self, type_, auto):
        # Per Apr 21 review: metric patches are ALWAYS report-only, no
        # exceptions. eval_addition and model_swap_suggestion are advisory.
        # Only prompt_patch and seed_reset may be auto-committed.
        i = _intervention(type_=type_)
        assert i.is_auto_applicable is auto

    def test_prompt_patch_rejects_unknown_mode(self):
        with pytest.raises(ValidationError):
            Intervention(
                id="I1",
                type="prompt_patch",
                fixes=["F1"],
                confidence=0.8,
                summary="s",
                patch={"mode": "merge", "content": "x"},
            )

    def test_prompt_patch_rejects_empty_content(self):
        with pytest.raises(ValidationError):
            Intervention(
                id="I1",
                type="prompt_patch",
                fixes=["F1"],
                confidence=0.8,
                summary="s",
                patch={"mode": "append", "content": "   "},
            )

    def test_seed_reset_requires_non_empty_new_seed(self):
        with pytest.raises(ValidationError):
            Intervention(
                id="I1",
                type="seed_reset",
                fixes=["F1"],
                confidence=0.9,
                summary="s",
                patch={"new_seed": ""},
            )

    def test_eval_addition_requires_non_empty_list(self):
        with pytest.raises(ValidationError):
            Intervention(
                id="I1",
                type="eval_addition",
                fixes=["F1"],
                confidence=0.8,
                summary="s",
                patch={"new_examples": []},
            )


# ---------------------------------------------------------------------------
# Cross-referential consistency at the PostmortemPlan level.
# ---------------------------------------------------------------------------


class TestPostmortemPlanCrossRef:
    def _plan(self, *, finding_ids: list[str], intervention_fixes: list[list[str]]) -> PostmortemPlan:
        findings = [_finding(id_=fid) for fid in finding_ids]
        interventions = [
            _intervention(id_=f"I{i+1}", fixes=fixes)
            for i, fixes in enumerate(intervention_fixes)
        ]
        return PostmortemPlan(
            diagnosis=Diagnosis(
                findings=findings,
                analyst_model_id="claude-opus-4-5",
                task_id="t1",
                slug="s1",
            ),
            proposal=Proposal(
                interventions=interventions,
                rationale="r",
            ),
        )

    def test_intervention_referencing_real_finding_passes(self):
        plan = self._plan(finding_ids=["F1", "F2"], intervention_fixes=[["F1", "F2"]])
        assert plan.proposal.interventions[0].fixes == ["F1", "F2"]

    def test_intervention_referencing_missing_finding_rejected(self):
        with pytest.raises(ValidationError) as exc:
            self._plan(finding_ids=["F1"], intervention_fixes=[["F2"]])
        # The error must name the specific unknown ID so the Analyst can
        # self-correct on a retry.
        assert "F2" in str(exc.value)

    def test_duplicate_finding_ids_rejected(self):
        with pytest.raises(ValidationError):
            self._plan(finding_ids=["F1", "F1"], intervention_fixes=[["F1"]])

    def test_partitioning_helpers_respect_auto_applicability(self):
        # Mix an auto-applicable and a report-only intervention; the plan's
        # helper methods must route them correctly.
        findings = [_finding(id_="F1")]
        interventions = [
            _intervention(id_="I1", type_="prompt_patch", fixes=["F1"]),
            _intervention(id_="I2", type_="metric_patch", fixes=["F1"]),
        ]
        plan = PostmortemPlan(
            diagnosis=Diagnosis(
                findings=findings,
                analyst_model_id="c",
                task_id="t",
                slug="s",
            ),
            proposal=Proposal(interventions=interventions, rationale="r"),
        )
        auto = plan.auto_applicable_interventions()
        report = plan.report_only_interventions()
        assert [i.id for i in auto] == ["I1"]
        assert [i.id for i in report] == ["I2"]


# ---------------------------------------------------------------------------
# Decision — outcome/deltas consistency.
# ---------------------------------------------------------------------------


class TestDecision:
    def _deltas(self, **overrides) -> DecisionDeltas:
        base = dict(
            global_delta=6.0,
            remediation_delta=20.0,
            max_scenario_regression=1.0,
            pre_best_score=50.0,
            post_best_score=56.0,
            retry_iterations_run=3,
        )
        base.update(overrides)
        return DecisionDeltas(**base)

    def test_accepted_requires_at_least_one_auto_applied(self):
        with pytest.raises(ValidationError):
            Decision(
                outcome="accepted",
                deltas=self._deltas(),
                auto_applied_intervention_ids=[],
                rationale="r",
            )

    def test_abstained_forbids_auto_applied(self):
        with pytest.raises(ValidationError):
            Decision(
                outcome="abstained",
                auto_applied_intervention_ids=["I1"],
                rationale="r",
            )

    def test_retry_dependent_outcomes_require_deltas(self):
        for outcome in ("accepted", "rejected", "partial"):
            with pytest.raises(ValidationError):
                Decision(
                    outcome=outcome,
                    deltas=None,
                    auto_applied_intervention_ids=["I1"] if outcome == "accepted" else [],
                    rationale="r",
                )

    def test_abstained_may_omit_deltas(self):
        d = Decision(
            outcome="abstained",
            deltas=None,
            auto_applied_intervention_ids=[],
            report_only_intervention_ids=["I1"],
            rationale="Only metric_patch interventions were proposed.",
        )
        assert d.deltas is None

    def test_accepted_happy_path(self):
        d = Decision(
            outcome="accepted",
            deltas=self._deltas(),
            auto_applied_intervention_ids=["I1"],
            rationale="Thresholds met on prompt_patch I1.",
        )
        assert d.outcome == "accepted"


# ---------------------------------------------------------------------------
# PostmortemConfig — threshold-ordering invariants.
# ---------------------------------------------------------------------------


class TestPostmortemConfig:
    def test_defaults_match_apr_21_review(self):
        cfg = PostmortemConfig()
        # Budget: $2.50 was the review's explicit number.
        assert cfg.cost_cap_usd == 2.50
        assert cfg.max_retry_iterations == 3
        assert cfg.min_confidence_seed_reset >= cfg.min_confidence_prompt_patch
        assert cfg.failure_threshold < cfg.success_threshold

    def test_failure_must_be_below_success(self):
        with pytest.raises(ValidationError):
            PostmortemConfig(failure_threshold=80.0, success_threshold=70.0)

    def test_seed_reset_confidence_cannot_be_below_prompt_patch(self):
        # seed_reset is a bigger intervention; requiring it to have at least
        # as high a confidence bar as prompt_patch prevents the operator
        # from misconfiguring their task into rubber-stamping big changes.
        with pytest.raises(ValidationError):
            PostmortemConfig(
                min_confidence_prompt_patch=0.9,
                min_confidence_seed_reset=0.5,
            )

    def test_config_attaches_to_run_config_with_defaults(self):
        # The actual wiring point: RunConfig without an explicit postmortem
        # block still has a well-formed default.
        from lpo.config.schema import RunConfig, TargetModelConfig

        rc = RunConfig(
            task_name="t",
            target_models=[TargetModelConfig(slug="s1", model_id="x", provider="lmstudio")],
        )
        assert isinstance(rc.postmortem, PostmortemConfig)
        assert rc.postmortem.cost_cap_usd == 2.50


# ---------------------------------------------------------------------------
# ExpectedImpact — alias handling for the reserved 'global' key.
# ---------------------------------------------------------------------------


class TestExpectedImpact:
    def test_accepts_json_style_global_key(self):
        # The Analyst emits JSON so the input will use the word 'global'
        # (reserved in Python). Pydantic's populate_by_name makes this work.
        impact = ExpectedImpact.model_validate({"global": [5.0, 12.0], "remediation": [15.0, 35.0]})
        assert impact.global_ == (5.0, 12.0)
        assert impact.remediation == (15.0, 35.0)
