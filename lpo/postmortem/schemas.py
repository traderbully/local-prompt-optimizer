"""Postmortem Pydantic schemas.

This module encodes the Apr 21 design-review invariants as type-level
constraints so the Postmortem Analyst can't skip them:

**Evidence invariant.** Every :class:`Finding` must carry an :class:`Evidence`
block with non-empty ``iterations``, ``example_ids``, and ``score_breakdown``.
A finding that doesn't cite concrete artifacts is rejected at validation time.

**Intervention provenance invariant.** Every :class:`Intervention` must list
the finding IDs it addresses via ``fixes`` (non-empty). Cross-referential
consistency (every fixes[i] references a real finding) is enforced by the
combined :class:`PostmortemPlan` model.

**Metric-patch safety invariant.** The decision gate in the engine layer
never auto-commits a ``metric_patch``. That rule doesn't live in the schema
(the schema allows proposing metric patches; report-only is a runtime
policy), but :meth:`Intervention.is_auto_applicable` reflects it here so
higher layers have a single source of truth.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Closed sets
# ---------------------------------------------------------------------------

FindingType = Literal[
    "scenario_blindspot",
    "prompt_gap",
    "metric_mismatch",
    "eval_coverage_gap",
    "overseer_local_optimum",
    "model_fit_issue",
]
"""Closed set of finding categories. See STAGE_8_DESIGN.md §4."""

InterventionType = Literal[
    "prompt_patch",
    "seed_reset",
    "metric_patch",
    "eval_addition",
    "model_swap_suggestion",
]
"""Closed set of intervention categories. See STAGE_8_DESIGN.md §5."""

DecisionOutcome = Literal["accepted", "rejected", "abstained", "partial"]
"""Four decision outcomes. See STAGE_8_DESIGN.md §7."""

Severity = Literal["low", "medium", "high", "critical"]


_FINDING_ID_RE = re.compile(r"^F\d+$")
_INTERVENTION_ID_RE = re.compile(r"^I\d+$")


# ---------------------------------------------------------------------------
# Evidence (the invariant at the heart of Stage 8)
# ---------------------------------------------------------------------------


class Evidence(BaseModel):
    """Concrete artifact citations backing a :class:`Finding`.

    The Apr 21 review made this non-optional: no finding may float
    without at least one iteration, one example, and a score breakdown
    that maps examples to per-iteration scores.

    ``extra='ignore'`` because real Analyst outputs tend to carry
    narration fields (``notes``, ``observation``) we don't need but
    shouldn't reject. The *positive* invariants (iterations + example_ids
    + score_breakdown non-empty and cross-consistent) are what matters.
    """

    model_config = ConfigDict(extra="ignore")

    iterations: list[int] = Field(
        ...,
        min_length=1,
        description="Iteration indices (1-based) the finding is drawn from.",
    )
    example_ids: list[str] = Field(
        ...,
        min_length=1,
        description="Eval example IDs cited by this finding.",
    )
    scenarios: list[str] = Field(
        default_factory=list,
        description=(
            "Scenario tags cited by this finding. May be empty for findings "
            "that are not scenario-scoped (e.g. a prompt_gap that applies "
            "across all scenarios)."
        ),
    )
    score_breakdown: dict[str, dict[str, float]] = Field(
        ...,
        description=(
            "Map of example_id -> {iter_label -> score}. iter_label is "
            "'iter_<n>' matching one of the iterations[] entries. Every "
            "key must appear in example_ids; every nested key must map to "
            "an iteration in iterations."
        ),
    )

    @field_validator("iterations")
    @classmethod
    def _iterations_positive(cls, v: list[int]) -> list[int]:
        if any(i < 1 for i in v):
            raise ValueError("Evidence.iterations must be 1-based (>= 1).")
        return v

    @field_validator("score_breakdown")
    @classmethod
    def _breakdown_non_empty(cls, v: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
        if not v:
            raise ValueError(
                "Evidence.score_breakdown must cite at least one example. "
                "Findings without concrete score evidence are rejected."
            )
        for ex_id, per_iter in v.items():
            if not per_iter:
                raise ValueError(
                    f"Evidence.score_breakdown[{ex_id!r}] is empty; every example "
                    "must have at least one iteration score."
                )
        return v

    @model_validator(mode="after")
    def _breakdown_keys_consistent(self) -> "Evidence":
        breakdown_examples = set(self.score_breakdown.keys())
        cited_examples = set(self.example_ids)
        missing = breakdown_examples - cited_examples
        if missing:
            raise ValueError(
                "Evidence.score_breakdown references example_ids not in "
                f"Evidence.example_ids: {sorted(missing)}. Every cited score "
                "must have a matching entry in example_ids."
            )

        allowed_iter_labels = {f"iter_{i}" for i in self.iterations}
        for ex_id, per_iter in self.score_breakdown.items():
            bad = set(per_iter.keys()) - allowed_iter_labels
            if bad:
                raise ValueError(
                    f"Evidence.score_breakdown[{ex_id!r}] uses iter labels "
                    f"{sorted(bad)} that are not in Evidence.iterations "
                    f"({sorted(allowed_iter_labels)})."
                )
        return self


# ---------------------------------------------------------------------------
# Finding
# ---------------------------------------------------------------------------


class Finding(BaseModel):
    """A single structured diagnosis item. See STAGE_8_DESIGN.md §4.

    ``extra='ignore'`` because Claude Opus reliably adds natural fields
    the schema doesn't name (``affected_scenarios``, ``priority``,
    ``confidence_rationale``, etc.). Those enrich the narrative but don't
    affect the decision gate; rejecting them just burns the Analyst's
    single retry budget on cosmetic re-emits.
    """

    model_config = ConfigDict(extra="ignore")

    id: str = Field(..., description="Stable ID of the form 'F<n>', e.g. 'F1'.")
    type: FindingType
    severity: Severity
    confidence: float = Field(..., ge=0.0, le=1.0)
    summary: str = Field(..., min_length=1)
    evidence: Evidence
    root_cause_hypothesis: str = Field(..., min_length=1)
    # Required only for metric_mismatch and model_fit_issue (see §4). For
    # other types, optional. The cross-type rule is enforced below.
    differential_evidence: str | None = None

    @field_validator("id")
    @classmethod
    def _id_shape(cls, v: str) -> str:
        if not _FINDING_ID_RE.match(v):
            raise ValueError(f"Finding.id must match F<n>, got {v!r}")
        return v

    @model_validator(mode="after")
    def _require_differential_for_strong_claims(self) -> "Finding":
        # metric_mismatch and model_fit_issue are the two types most prone to
        # hallucination (it's easy to blame "the metric" or "the model" when
        # the real problem is the prompt). Per §4, these require evidence
        # that actively distinguishes the diagnosis from simpler alternatives.
        if self.type in ("metric_mismatch", "model_fit_issue"):
            if not self.differential_evidence or not self.differential_evidence.strip():
                raise ValueError(
                    f"Finding.type={self.type!r} requires differential_evidence "
                    "(evidence that the problem is the metric/model rather than "
                    "the prompt). See STAGE_8_DESIGN.md §4."
                )
        return self


# ---------------------------------------------------------------------------
# Intervention
# ---------------------------------------------------------------------------


class ExpectedImpact(BaseModel):
    """Ranges the Analyst predicts for the intervention's effect.

    ``global`` is a Python reserved word; Pydantic's ``populate_by_name``
    lets us accept the JSON key ``"global"`` while exposing the attribute
    as ``global_`` in Python.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    global_: tuple[float, float] | None = Field(default=None, alias="global")
    remediation: tuple[float, float] | None = None


class Intervention(BaseModel):
    """A single proposed remediation. See STAGE_8_DESIGN.md §5.

    The ``patch`` field carries a type-specific payload. Rather than use a
    discriminated union (which complicates the JSON the Analyst has to emit),
    we validate the payload's shape per-type in :meth:`_validate_patch_shape`.

    ``extra='ignore'`` at the Intervention level; the ``patch`` dict itself
    is typed as ``dict[str, Any]`` and never had extra-forbid in the first
    place (patches need to carry arbitrary type-specific keys).
    """

    model_config = ConfigDict(extra="ignore")

    id: str = Field(..., description="Stable ID of the form 'I<n>'.")
    type: InterventionType
    fixes: list[str] = Field(
        ...,
        min_length=1,
        description=(
            "Finding IDs this intervention addresses. Non-empty by invariant. "
            "Cross-consistency (every ID maps to a real finding) is checked at "
            "the PostmortemPlan level."
        ),
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    summary: str = Field(..., min_length=1)
    expected_impact: ExpectedImpact | None = None
    patch: dict[str, Any]

    @field_validator("id")
    @classmethod
    def _id_shape(cls, v: str) -> str:
        if not _INTERVENTION_ID_RE.match(v):
            raise ValueError(f"Intervention.id must match I<n>, got {v!r}")
        return v

    @field_validator("fixes")
    @classmethod
    def _fixes_are_finding_ids(cls, v: list[str]) -> list[str]:
        for fid in v:
            if not _FINDING_ID_RE.match(fid):
                raise ValueError(
                    f"Intervention.fixes must contain finding IDs (F<n>); got {fid!r}"
                )
        return v

    @model_validator(mode="after")
    def _validate_patch_shape(self) -> "Intervention":
        # Per-type required keys in the patch payload. Keep minimal and
        # forgiving — the Analyst may emit extra keys as long as the
        # mandatory ones are present.
        required_by_type: dict[str, set[str]] = {
            "prompt_patch": {"mode", "content"},
            "seed_reset": {"new_seed"},
            "metric_patch": {"rationale"},
            "eval_addition": {"new_examples"},
            "model_swap_suggestion": {"suggested_models", "rationale"},
        }
        need = required_by_type.get(self.type, set())
        missing = need - set(self.patch.keys())
        if missing:
            raise ValueError(
                f"Intervention.type={self.type!r} requires patch keys "
                f"{sorted(need)}; missing {sorted(missing)}."
            )

        # Type-specific narrow checks.
        if self.type == "prompt_patch":
            mode = self.patch.get("mode")
            if mode not in {"append", "prepend", "replace"}:
                raise ValueError(
                    f"prompt_patch.mode must be 'append'|'prepend'|'replace'; got {mode!r}"
                )
            if not str(self.patch.get("content", "")).strip():
                raise ValueError("prompt_patch.content must be non-empty.")
        elif self.type == "seed_reset":
            if not str(self.patch.get("new_seed", "")).strip():
                raise ValueError("seed_reset.new_seed must be non-empty.")
        elif self.type == "eval_addition":
            examples = self.patch.get("new_examples")
            if not isinstance(examples, list) or not examples:
                raise ValueError(
                    "eval_addition.new_examples must be a non-empty list."
                )
        elif self.type == "model_swap_suggestion":
            suggested = self.patch.get("suggested_models")
            if not isinstance(suggested, list) or not suggested:
                raise ValueError(
                    "model_swap_suggestion.suggested_models must be a non-empty list."
                )
        return self

    @property
    def is_auto_applicable(self) -> bool:
        """Whether the decision gate may auto-commit this intervention.

        Per Apr 21 review, only prompt_patch and seed_reset are auto-
        applicable. Metric patches, eval additions, and model-swap
        suggestions are always report-only.
        """
        return self.type in ("prompt_patch", "seed_reset")


# ---------------------------------------------------------------------------
# Diagnosis + Proposal + combined Plan
# ---------------------------------------------------------------------------


class Diagnosis(BaseModel):
    """Analyst output: findings + bird's-eye observations.

    ``extra='ignore'`` (Analyst-emitted).
    """

    model_config = ConfigDict(extra="ignore")

    findings: list[Finding] = Field(..., min_length=1)
    metric_observations: list[str] = Field(default_factory=list)
    overseer_drift_observations: list[str] = Field(default_factory=list)
    analyst_model_id: str
    task_id: str
    slug: str
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @model_validator(mode="after")
    def _unique_finding_ids(self) -> "Diagnosis":
        ids = [f.id for f in self.findings]
        if len(set(ids)) != len(ids):
            raise ValueError(f"Diagnosis.findings has duplicate IDs: {ids}")
        return self


class Proposal(BaseModel):
    """Planner output: concrete interventions + human-review summary.

    ``extra='ignore'`` (Analyst-emitted).
    """

    model_config = ConfigDict(extra="ignore")

    interventions: list[Intervention] = Field(..., min_length=1)
    rationale: str = Field(..., min_length=1)
    human_review_summary: str = Field(default="")

    @model_validator(mode="after")
    def _unique_intervention_ids(self) -> "Proposal":
        ids = [i.id for i in self.interventions]
        if len(set(ids)) != len(ids):
            raise ValueError(f"Proposal.interventions has duplicate IDs: {ids}")
        return self


class PostmortemPlan(BaseModel):
    """The combined diagnosis + proposal as emitted by the Analyst in one shot.

    Cross-referential consistency (every intervention.fixes[i] maps to a
    real finding) is enforced here, not on the pieces in isolation, because
    the constraint spans both models.

    ``extra='ignore'`` because the Analyst sometimes wraps its output in
    a container with keys like ``version``, ``notes``, or ``run_id``.
    These don't affect behavior; drop them silently.
    """

    model_config = ConfigDict(extra="ignore")

    diagnosis: Diagnosis
    proposal: Proposal

    @model_validator(mode="after")
    def _cross_reference_fixes(self) -> "PostmortemPlan":
        finding_ids = {f.id for f in self.diagnosis.findings}
        for intervention in self.proposal.interventions:
            unknown = [fid for fid in intervention.fixes if fid not in finding_ids]
            if unknown:
                raise ValueError(
                    f"Intervention {intervention.id!r} references unknown finding "
                    f"IDs {unknown}. Known finding IDs: {sorted(finding_ids)}."
                )
        return self

    def auto_applicable_interventions(self) -> list[Intervention]:
        """Return the subset the decision gate is allowed to auto-commit."""
        return [i for i in self.proposal.interventions if i.is_auto_applicable]

    def report_only_interventions(self) -> list[Intervention]:
        """Return the subset that requires human approval (including all
        metric_patch interventions — never auto-commit per Apr 21 review)."""
        return [i for i in self.proposal.interventions if not i.is_auto_applicable]


# ---------------------------------------------------------------------------
# Decision + config
# ---------------------------------------------------------------------------


class DecisionDeltas(BaseModel):
    """Measurement block backing every decision."""

    model_config = ConfigDict(extra="forbid")

    global_delta: float
    remediation_delta: float
    max_scenario_regression: float
    pre_best_score: float
    post_best_score: float
    retry_iterations_run: int


class Decision(BaseModel):
    """Final gate outcome + rationale + provenance for the postmortem."""

    model_config = ConfigDict(extra="forbid")

    outcome: DecisionOutcome
    deltas: DecisionDeltas | None = None  # None when no retry was run (abstained)
    auto_applied_intervention_ids: list[str] = Field(default_factory=list)
    report_only_intervention_ids: list[str] = Field(default_factory=list)
    rationale: str = Field(..., min_length=1)
    thresholds_snapshot: dict[str, float] = Field(default_factory=dict)
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @model_validator(mode="after")
    def _consistent_with_outcome(self) -> "Decision":
        if self.outcome == "accepted" and not self.auto_applied_intervention_ids:
            raise ValueError(
                "Decision.outcome=accepted requires at least one "
                "auto_applied_intervention_ids entry."
            )
        if self.outcome == "abstained" and self.auto_applied_intervention_ids:
            raise ValueError(
                "Decision.outcome=abstained means nothing was auto-applied; "
                "auto_applied_intervention_ids must be empty."
            )
        if self.outcome in ("accepted", "rejected", "partial") and self.deltas is None:
            raise ValueError(
                f"Decision.outcome={self.outcome} requires a deltas block "
                "(the focused retry must have run)."
            )
        return self


class PostmortemConfig(BaseModel):
    """Per-task postmortem tuning, included in `config.yaml: postmortem`.

    These fields govern behavior *if* the operator invokes the postmortem.
    Per Apr 21 review there is no ``enabled`` flag — the phase is opt-in
    via MCP tool or CLI, never automatic.
    """

    model_config = ConfigDict(extra="forbid")

    cost_cap_usd: float = Field(
        default=2.50,
        ge=0.0,
        description=(
            "Separate budget for the postmortem phase (diagnostic call + "
            "focused retry). Independent of the task's cost_cap_usd."
        ),
    )
    max_retry_iterations: int = Field(default=3, ge=1, le=10)

    accept_threshold_global: float = Field(default=5.0)
    accept_threshold_remediation: float = Field(default=15.0)
    regression_tolerance: float = Field(default=3.0, ge=0.0)

    failure_threshold: float = Field(
        default=20.0,
        ge=0.0,
        le=100.0,
        description="Scenarios scoring <= this on the pre-run are 'failing'.",
    )
    success_threshold: float = Field(
        default=70.0,
        ge=0.0,
        le=100.0,
        description="Scenarios scoring >= this on the pre-run are 'passing'.",
    )

    min_confidence_prompt_patch: float = Field(default=0.70, ge=0.0, le=1.0)
    min_confidence_seed_reset: float = Field(default=0.85, ge=0.0, le=1.0)

    analyst_model_id: str = Field(default="claude-opus-4-7")

    @model_validator(mode="after")
    def _threshold_ordering(self) -> "PostmortemConfig":
        if self.failure_threshold >= self.success_threshold:
            raise ValueError(
                "postmortem.failure_threshold must be strictly less than "
                "postmortem.success_threshold."
            )
        if self.min_confidence_seed_reset < self.min_confidence_prompt_patch:
            raise ValueError(
                "postmortem.min_confidence_seed_reset should be >= "
                "postmortem.min_confidence_prompt_patch (seed_reset is a "
                "bigger intervention and merits a higher confidence bar)."
            )
        return self
