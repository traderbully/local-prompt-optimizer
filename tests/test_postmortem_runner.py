"""End-to-end postmortem orchestrator tests.

Tie the full pipeline together — Analyst (stubbed), patch application,
focused retry (stubbed), decision gate, artifact writing — and verify
the on-disk outcome for each of the four terminal states:

  accepted   — thresholds met, no report-only interventions.
  partial    — thresholds met AND the proposal had report-only items.
  rejected   — thresholds not met.
  abstained  — no auto-applicable interventions cleared the confidence
               floor (or mode=propose_only).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from lpo.core.cost import CostTracker
from lpo.core.history import IterationRecord
from lpo.core.iteration import IterationResult
from lpo.postmortem.runner import run_postmortem
from lpo.postmortem.schemas import PostmortemConfig
from lpo.scoring.aggregation import AggregatedScore

from tests.test_postmortem_artifacts import _write_iteration, _write_task_files


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


@dataclass
class _StubResp:
    text: str
    model_id: str = "claude-opus-stub"


class _StubAnalystClient:
    def __init__(self, responses: list[str]):
        self._queue = [_StubResp(text=r) for r in responses]
        self.calls: list[dict[str, Any]] = []

    async def complete(self, *, system, messages, temperature=0.0, max_tokens=8192):
        self.calls.append({"messages": list(messages)})
        return self._queue.pop(0)


@dataclass
class _StubRunner:
    canned_score: float
    per_example: dict[str, float]
    per_scenario: dict[str, float]

    async def run(self, *, iteration_index, prompt, eval_records, gold_standard, task_name):
        record = IterationRecord(
            index=iteration_index,
            prompt=prompt,
            aggregate_score=self.canned_score,
            per_example=dict(self.per_example),
            per_scenario=dict(self.per_scenario),
            failed_ids=[],
            outputs=[{"id": k, "output": "x", "score": v} for k, v in self.per_example.items()],
            decision="pending",
        )
        agg = AggregatedScore(
            aggregate=self.canned_score,
            per_example=dict(self.per_example),
            per_scenario=dict(self.per_scenario),
            failed_ids=[],
        )
        return IterationResult(record=record, aggregated=agg)


def _make_factory(runner: _StubRunner):
    def factory(task, target_cfg, cost):
        return runner, None
    return factory


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def task_with_history(tmp_path: Path):
    """A task whose main-ratchet run scored 40 global, with 'hard' as the
    failing scenario and 'easy' as the passing one. Winner already written."""
    root = tmp_path / "t"
    root.mkdir()
    _write_task_files(root, slug="stub")
    history = root / "runs" / "stub" / "history"
    _write_iteration(
        history / "iter_0001",
        index=1,
        prompt="ORIGINAL PROMPT BODY",
        aggregate=40.0,
        per_example={"ex001": 80.0, "ex002": 0.0},
        per_scenario={"easy": 80.0, "hard": 0.0},
        decision="initial",
    )
    # Winner dir mirrors the best iteration.
    winner = root / "runs" / "stub" / "winner"
    winner.mkdir(parents=True)
    (winner / "prompt.txt").write_text("ORIGINAL PROMPT BODY", encoding="utf-8")
    (winner / "report.md").write_text("# Original winner", encoding="utf-8")
    return root


def _valid_plan_json(*, intervention_confidence: float = 0.85, extra_interventions: list[dict] | None = None) -> str:
    plan = {
        "diagnosis": {
            "findings": [
                {
                    "id": "F1",
                    "type": "scenario_blindspot",
                    "severity": "high",
                    "confidence": 0.9,
                    "summary": "hard scenario scored 0.",
                    "evidence": {
                        "iterations": [1],
                        "example_ids": ["ex002"],
                        "scenarios": ["hard"],
                        "score_breakdown": {"ex002": {"iter_1": 0.0}},
                    },
                    "root_cause_hypothesis": "Prompt missing a rule for hard inputs.",
                }
            ],
            "metric_observations": [],
            "overseer_drift_observations": [],
            "analyst_model_id": "claude-opus-stub",
            "task_id": "test",
            "slug": "stub",
        },
        "proposal": {
            "interventions": [
                {
                    "id": "I1",
                    "type": "prompt_patch",
                    "fixes": ["F1"],
                    "confidence": intervention_confidence,
                    "summary": "Add hard-input rule.",
                    "patch": {
                        "mode": "append",
                        "content": "- For hard inputs: do the right thing.",
                    },
                }
            ],
            "rationale": "Addresses the only failing scenario.",
            "human_review_summary": "",
        },
    }
    if extra_interventions:
        plan["proposal"]["interventions"].extend(extra_interventions)
    return json.dumps(plan)


# ---------------------------------------------------------------------------
# Propose-only mode
# ---------------------------------------------------------------------------


class TestProposeOnly:
    @pytest.mark.asyncio
    async def test_writes_diagnosis_and_proposal_but_no_retry(self, task_with_history):
        analyst = _StubAnalystClient([_valid_plan_json()])
        result = await run_postmortem(
            task_with_history,
            slug="stub",
            analyst_client=analyst,
            mode="propose_only",
        )
        assert result.decision.outcome == "abstained"
        assert result.retry is None
        pm = result.postmortem_root
        assert (pm / "diagnosis.json").exists()
        assert (pm / "proposal.md").exists()
        assert (pm / "proposal.json").exists()
        assert (pm / "decision.json").exists()
        # No retry subdirectory.
        assert not (pm / "retry").exists()

    @pytest.mark.asyncio
    async def test_winner_is_untouched_in_propose_only(self, task_with_history):
        analyst = _StubAnalystClient([_valid_plan_json()])
        await run_postmortem(
            task_with_history,
            slug="stub",
            analyst_client=analyst,
            mode="propose_only",
        )
        winner = (task_with_history / "runs" / "stub" / "winner" / "prompt.txt").read_text(encoding="utf-8")
        assert winner == "ORIGINAL PROMPT BODY"
        # And no rollback directory was created.
        assert not (task_with_history / "runs" / "stub" / "winner.pre_postmortem").exists()


# ---------------------------------------------------------------------------
# Autonomous mode — each outcome
# ---------------------------------------------------------------------------


class TestAutonomousAccepted:
    @pytest.mark.asyncio
    async def test_thresholds_met_promotes_winner_with_provenance(self, task_with_history):
        # Pre: 40 global, hard=0. Post retry: 60 global, hard=40 (remediation +40).
        analyst = _StubAnalystClient([_valid_plan_json()])
        runner = _StubRunner(
            canned_score=60.0,
            per_example={"ex001": 80.0, "ex002": 40.0},
            per_scenario={"easy": 80.0, "hard": 40.0},
        )
        result = await run_postmortem(
            task_with_history,
            slug="stub",
            analyst_client=analyst,
            retry_runner_factory=_make_factory(runner),
        )
        assert result.decision.outcome == "accepted"
        assert result.decision.auto_applied_intervention_ids == ["I1"]
        # The retry's patched prompt was promoted to the winner.
        winner_prompt = (task_with_history / "runs" / "stub" / "winner" / "prompt.txt").read_text(encoding="utf-8")
        assert "For hard inputs" in winner_prompt
        # Original winner preserved under winner.pre_postmortem.
        rollback = task_with_history / "runs" / "stub" / "winner.pre_postmortem"
        assert rollback.exists()
        assert (rollback / "prompt.txt").read_text(encoding="utf-8") == "ORIGINAL PROMPT BODY"
        # Provenance recorded so future postmortems can detect stacking.
        prov = json.loads((task_with_history / "runs" / "stub" / "winner" / "provenance.json").read_text())
        assert prov["source"] == "postmortem"
        assert "I1" in prov["auto_applied_intervention_ids"]
        # prompt.txt.best also updated so the main ratchet picks it up next run.
        best = (task_with_history / "runs" / "stub" / "prompt.txt.best").read_text(encoding="utf-8")
        assert "For hard inputs" in best


class TestAutonomousPartial:
    @pytest.mark.asyncio
    async def test_mixed_auto_and_report_only_yields_partial(self, task_with_history):
        # Include a metric_patch alongside the prompt_patch. Thresholds pass
        # on the auto-applied change; the metric_patch is surfaced for
        # human review but not applied.
        metric_only = {
            "id": "I2",
            "type": "metric_patch",
            "fixes": ["F1"],
            "confidence": 0.95,
            "summary": "Metric should penalize the silent failure.",
            "patch": {"rationale": "Failing scenarios currently score 0 with no differentiation."},
        }
        analyst = _StubAnalystClient([_valid_plan_json(extra_interventions=[metric_only])])
        runner = _StubRunner(
            canned_score=60.0,
            per_example={"ex001": 80.0, "ex002": 40.0},
            per_scenario={"easy": 80.0, "hard": 40.0},
        )
        result = await run_postmortem(
            task_with_history,
            slug="stub",
            analyst_client=analyst,
            retry_runner_factory=_make_factory(runner),
        )
        assert result.decision.outcome == "partial"
        assert result.decision.auto_applied_intervention_ids == ["I1"]
        assert result.decision.report_only_intervention_ids == ["I2"]

    @pytest.mark.asyncio
    async def test_partial_still_promotes_the_winner(self, task_with_history):
        metric_only = {
            "id": "I2",
            "type": "metric_patch",
            "fixes": ["F1"],
            "confidence": 0.95,
            "summary": "s",
            "patch": {"rationale": "r"},
        }
        analyst = _StubAnalystClient([_valid_plan_json(extra_interventions=[metric_only])])
        runner = _StubRunner(
            canned_score=60.0,
            per_example={"ex001": 80.0, "ex002": 40.0},
            per_scenario={"easy": 80.0, "hard": 40.0},
        )
        await run_postmortem(
            task_with_history,
            slug="stub",
            analyst_client=analyst,
            retry_runner_factory=_make_factory(runner),
        )
        # Partial means "accepted on the auto-applicable subset" — the
        # prompt is still promoted. The metric_patch is preserved in
        # proposal.json and the report for human action.
        winner_prompt = (task_with_history / "runs" / "stub" / "winner" / "prompt.txt").read_text(encoding="utf-8")
        assert "For hard inputs" in winner_prompt


class TestAutonomousRejected:
    @pytest.mark.asyncio
    async def test_flat_score_produces_rejection_and_preserves_winner(self, task_with_history):
        # Retry scored the same as pre — all three thresholds fail the AND.
        analyst = _StubAnalystClient([_valid_plan_json()])
        runner = _StubRunner(
            canned_score=40.0,
            per_example={"ex001": 80.0, "ex002": 0.0},
            per_scenario={"easy": 80.0, "hard": 0.0},
        )
        result = await run_postmortem(
            task_with_history,
            slug="stub",
            analyst_client=analyst,
            retry_runner_factory=_make_factory(runner),
        )
        assert result.decision.outcome == "rejected"
        # Winner dir untouched, no rollback dir created.
        winner_prompt = (task_with_history / "runs" / "stub" / "winner" / "prompt.txt").read_text(encoding="utf-8")
        assert winner_prompt == "ORIGINAL PROMPT BODY"
        assert not (task_with_history / "runs" / "stub" / "winner.pre_postmortem").exists()

    @pytest.mark.asyncio
    async def test_regression_detected_even_with_global_improvement(self, task_with_history):
        # hard 0->40 (good), easy 80->70 (10-pt regression above 3-pt tolerance).
        analyst = _StubAnalystClient([_valid_plan_json()])
        runner = _StubRunner(
            canned_score=55.0,
            per_example={"ex001": 70.0, "ex002": 40.0},
            per_scenario={"easy": 70.0, "hard": 40.0},
        )
        result = await run_postmortem(
            task_with_history,
            slug="stub",
            analyst_client=analyst,
            retry_runner_factory=_make_factory(runner),
        )
        assert result.decision.outcome == "rejected"
        assert "max_scenario_regression" in result.decision.rationale


class TestAutonomousAbstained:
    @pytest.mark.asyncio
    async def test_low_confidence_intervention_abstains_without_retry(self, task_with_history):
        # Confidence 0.50 is below the 0.70 prompt_patch floor.
        analyst = _StubAnalystClient([_valid_plan_json(intervention_confidence=0.50)])
        # Any runner — should never be called.
        runner_calls = {"count": 0}

        def runner_factory(task, target_cfg, cost):
            runner_calls["count"] += 1
            raise AssertionError("focused retry must not run when no intervention clears the floor")

        result = await run_postmortem(
            task_with_history,
            slug="stub",
            analyst_client=analyst,
            retry_runner_factory=runner_factory,
        )
        assert result.decision.outcome == "abstained"
        assert result.retry is None
        assert runner_calls["count"] == 0
        assert "I1" in result.decision.rationale or "I1" in result.decision.report_only_intervention_ids


# ---------------------------------------------------------------------------
# Report + artifact integrity
# ---------------------------------------------------------------------------


class TestArtifactIntegrity:
    @pytest.mark.asyncio
    async def test_report_names_outcome_and_findings(self, task_with_history):
        analyst = _StubAnalystClient([_valid_plan_json()])
        runner = _StubRunner(
            canned_score=60.0,
            per_example={"ex001": 80.0, "ex002": 40.0},
            per_scenario={"easy": 80.0, "hard": 40.0},
        )
        result = await run_postmortem(
            task_with_history,
            slug="stub",
            analyst_client=analyst,
            retry_runner_factory=_make_factory(runner),
        )
        report = (result.postmortem_root / "report.md").read_text(encoding="utf-8")
        assert "accepted" in report.lower()
        assert "F1" in report
        assert "scenario_blindspot" in report
        assert "I1" in report

    @pytest.mark.asyncio
    async def test_diagnosis_json_round_trips_through_pydantic(self, task_with_history):
        from lpo.postmortem.schemas import Diagnosis

        analyst = _StubAnalystClient([_valid_plan_json()])
        result = await run_postmortem(
            task_with_history,
            slug="stub",
            analyst_client=analyst,
            mode="propose_only",
        )
        blob = (result.postmortem_root / "diagnosis.json").read_text(encoding="utf-8")
        # Parse back via the schema — validates that we wrote it correctly.
        diag = Diagnosis.model_validate_json(blob)
        assert diag.findings[0].id == "F1"
