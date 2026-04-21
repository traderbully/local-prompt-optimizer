"""Intervention application + selection tests."""

from __future__ import annotations

import pytest

from lpo.postmortem.patches import (
    PatchApplicationError,
    apply_prompt_patch,
    apply_seed_reset,
    select_and_apply,
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
# apply_prompt_patch
# ---------------------------------------------------------------------------


class TestApplyPromptPatch:
    def test_append_default(self):
        result = apply_prompt_patch(
            "You are an assistant.",
            {"mode": "append", "content": "- Be concise."},
        )
        # Original prompt preserved, new rule appended with blank-line separation.
        assert result.startswith("You are an assistant.")
        assert "- Be concise." in result
        assert result.endswith("\n")

    def test_prepend_pushes_existing_content_down(self):
        result = apply_prompt_patch(
            "Existing prompt body.",
            {"mode": "prepend", "content": "PREAMBLE"},
        )
        assert result.index("PREAMBLE") < result.index("Existing")

    def test_replace_overwrites_everything(self):
        result = apply_prompt_patch(
            "old content",
            {"mode": "replace", "content": "completely new"},
        )
        assert "old" not in result
        assert "completely new" in result

    def test_append_with_after_section_inserts_at_right_place(self):
        source = (
            "# Preamble\n\n"
            "Hello.\n\n"
            "## Rules\n\n"
            "- Rule A\n\n"
            "## Notes\n\n"
            "Some notes.\n"
        )
        patched = apply_prompt_patch(
            source,
            {"mode": "append", "after_section": "## Rules", "content": "- Rule B"},
        )
        # New rule must be inside the Rules section (before Notes).
        rules_idx = patched.index("## Rules")
        notes_idx = patched.index("## Notes")
        rule_b_idx = patched.index("- Rule B")
        assert rules_idx < rule_b_idx < notes_idx

    def test_append_with_missing_section_falls_back_to_end(self):
        # Documented behavior: when after_section isn't present, append to
        # the end rather than silently dropping the rule.
        source = "Intro\n\n## Only Section\nstuff\n"
        patched = apply_prompt_patch(
            source,
            {"mode": "append", "after_section": "## Missing", "content": "- Fallback rule"},
        )
        assert "- Fallback rule" in patched
        assert patched.index("stuff") < patched.index("- Fallback rule")

    def test_append_section_header_matching_accepts_different_depths(self):
        # after_section='Rules' (no leading #) should match a real '## Rules' header.
        source = "# X\n\n## Rules\n- existing\n"
        patched = apply_prompt_patch(
            source,
            {"mode": "append", "after_section": "Rules", "content": "- added"},
        )
        assert "- added" in patched

    def test_empty_content_rejected(self):
        with pytest.raises(PatchApplicationError):
            apply_prompt_patch("body", {"mode": "append", "content": "   "})

    def test_unknown_mode_rejected(self):
        with pytest.raises(PatchApplicationError):
            apply_prompt_patch("body", {"mode": "merge", "content": "x"})


class TestApplySeedReset:
    def test_happy_path(self):
        assert apply_seed_reset({"new_seed": "Fresh seed."}).strip() == "Fresh seed."

    def test_empty_seed_rejected(self):
        with pytest.raises(PatchApplicationError):
            apply_seed_reset({"new_seed": "   "})


# ---------------------------------------------------------------------------
# select_and_apply — confidence filtering + stacking order
# ---------------------------------------------------------------------------


def _mkfind(id_: str = "F1") -> Finding:
    return Finding(
        id=id_,
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


def _mkint(
    id_: str,
    type_: str,
    confidence: float,
    *,
    fixes: list[str] | None = None,
    patch: dict | None = None,
) -> Intervention:
    fixes = fixes or ["F1"]
    if patch is None:
        patch = {
            "prompt_patch": {"mode": "append", "content": f"- rule from {id_}"},
            "seed_reset": {"new_seed": f"SEED FROM {id_}"},
            "metric_patch": {"rationale": "metric rationale"},
            "eval_addition": {"new_examples": [{"input": "x", "expected": "y"}]},
            "model_swap_suggestion": {"suggested_models": ["x"], "rationale": "r"},
        }[type_]
    return Intervention(
        id=id_,
        type=type_,
        fixes=fixes,
        confidence=confidence,
        summary="s",
        patch=patch,
    )


def _plan(interventions: list[Intervention]) -> PostmortemPlan:
    return PostmortemPlan(
        diagnosis=Diagnosis(
            findings=[_mkfind()],
            analyst_model_id="x",
            task_id="t",
            slug="s",
        ),
        proposal=Proposal(interventions=interventions, rationale="r"),
    )


class TestSelectAndApply:
    def test_all_above_floor_applied_in_order(self):
        cfg = PostmortemConfig()  # prompt_patch floor 0.70
        plan = _plan([
            _mkint("I1", "prompt_patch", 0.85),
            _mkint("I2", "prompt_patch", 0.75),
        ])
        result = select_and_apply(plan, current_best_prompt="BODY", cfg=cfg)
        assert result.applied_intervention_ids == ["I1", "I2"]
        assert result.skipped_low_confidence_ids == []
        assert result.patched_prompt is not None
        # Both rules present.
        assert "rule from I1" in result.patched_prompt
        assert "rule from I2" in result.patched_prompt

    def test_below_floor_skipped(self):
        cfg = PostmortemConfig(
            min_confidence_prompt_patch=0.80,
            min_confidence_seed_reset=0.90,
        )
        plan = _plan([
            _mkint("I1", "prompt_patch", 0.85),
            _mkint("I2", "prompt_patch", 0.60),  # below floor
        ])
        result = select_and_apply(plan, current_best_prompt="BODY", cfg=cfg)
        assert result.applied_intervention_ids == ["I1"]
        assert result.skipped_low_confidence_ids == ["I2"]

    def test_all_below_floor_returns_none_patch(self):
        # Raise both floors — the config validator enforces
        # seed_reset_floor >= prompt_patch_floor.
        cfg = PostmortemConfig(
            min_confidence_prompt_patch=0.90,
            min_confidence_seed_reset=0.95,
        )
        plan = _plan([
            _mkint("I1", "prompt_patch", 0.60),
            _mkint("I2", "prompt_patch", 0.50),
        ])
        result = select_and_apply(plan, current_best_prompt="BODY", cfg=cfg)
        assert result.patched_prompt is None
        assert result.applied_intervention_ids == []
        assert set(result.skipped_low_confidence_ids) == {"I1", "I2"}

    def test_seed_reset_applied_first_then_prompt_patch_stacks(self):
        # A seed_reset replaces the prompt entirely; a following prompt_patch
        # must see the new seed, not the old prompt.
        cfg = PostmortemConfig()
        plan = _plan([
            _mkint("I1", "prompt_patch", 0.85),
            _mkint("I2", "seed_reset", 0.90),  # seed_reset floor is 0.85
        ])
        result = select_and_apply(plan, current_best_prompt="ORIGINAL", cfg=cfg)
        # seed_reset runs first (stable sort), so the new body is SEED FROM I2
        # with I1's rule appended on top of it.
        assert result.patched_prompt is not None
        assert "ORIGINAL" not in result.patched_prompt
        assert "SEED FROM I2" in result.patched_prompt
        assert "rule from I1" in result.patched_prompt
        # Application order reported: seed first, then prompt_patch.
        assert result.applied_intervention_ids == ["I2", "I1"]

    def test_seed_reset_confidence_bar_higher_than_prompt_patch(self):
        # Default min_confidence_seed_reset=0.85. A seed_reset at 0.80 must
        # be skipped even though a prompt_patch at 0.80 would be applied.
        cfg = PostmortemConfig()
        plan = _plan([_mkint("I1", "seed_reset", 0.80)])
        result = select_and_apply(plan, current_best_prompt="BODY", cfg=cfg)
        assert result.patched_prompt is None
        assert result.skipped_low_confidence_ids == ["I1"]

    def test_report_only_interventions_pass_through(self):
        # metric_patch / eval_addition / model_swap_suggestion are never
        # applied. They must still be surfaced via report_only_intervention_ids
        # so the caller can write them into the report.
        cfg = PostmortemConfig()
        plan = _plan([
            _mkint("I1", "prompt_patch", 0.80),
            _mkint("I2", "metric_patch", 0.95),       # never auto-applied
            _mkint("I3", "eval_addition", 0.90),     # never auto-applied
            _mkint("I4", "model_swap_suggestion", 0.90),  # never auto-applied
        ])
        result = select_and_apply(plan, current_best_prompt="BODY", cfg=cfg)
        assert result.applied_intervention_ids == ["I1"]
        assert set(result.report_only_intervention_ids) == {"I2", "I3", "I4"}
        # No report-only payload ever bleeds into the patched prompt.
        assert "metric rationale" not in result.patched_prompt
