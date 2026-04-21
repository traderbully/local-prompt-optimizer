"""MCP tool tests for ``lpo_run_postmortem`` (Stage 8, Batch 4).

The handler's production path instantiates a real AnthropicClient; here
we bypass that entirely by injecting a stub via the handler's
``_run_postmortem_cb`` seam. The stub returns a pre-made
:class:`PostmortemResult` so we can assert exactly how the MCP envelope
summarizes it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from lpo.core.authoring import TargetSpec, create_task_bundle
from lpo.postmortem.schemas import (
    Decision,
    DecisionDeltas,
    Diagnosis,
    Evidence,
    Finding,
    Intervention,
    PostmortemPlan,
    Proposal,
)
from lpo.server.mcp_server import LpoMcpHandlers


# ---------------------------------------------------------------------------
# Fixtures — a minimal task the tool can resolve a task_id against.
# ---------------------------------------------------------------------------


@pytest.fixture
def handlers(tmp_path: Path) -> LpoMcpHandlers:
    # Task bundle that the handler can TaskBundle.load — same shape the
    # other tests build via create_task_bundle.
    create_task_bundle(
        tasks_root=tmp_path,
        name="pm-task",
        task_description="desc",
        example_inputs=["a", "b", "c"],
        output_type="json",
        required_json_fields=["result"],
        targets=[
            TargetSpec(
                slug="stub",
                provider="stub",
                model_id="stub-model",
                base_url="",
                extra={"stub_mode": "fixed", "stub_fixed_text": '{"result": "x"}'},
            )
        ],
        strategy="single",
    )
    return LpoMcpHandlers(tasks_root=tmp_path)


def _mkplan() -> PostmortemPlan:
    evidence = Evidence(
        iterations=[1],
        example_ids=["ex001"],
        score_breakdown={"ex001": {"iter_1": 0.0}},
    )
    finding = Finding(
        id="F1",
        type="scenario_blindspot",
        severity="high",
        confidence=0.9,
        summary="s",
        evidence=evidence,
        root_cause_hypothesis="h",
    )
    intervention = Intervention(
        id="I1",
        type="prompt_patch",
        fixes=["F1"],
        confidence=0.85,
        summary="summary",
        patch={"mode": "append", "content": "- rule"},
    )
    return PostmortemPlan(
        diagnosis=Diagnosis(
            findings=[finding],
            analyst_model_id="claude-opus-4-5",
            task_id="pm-task",
            slug="stub",
        ),
        proposal=Proposal(interventions=[intervention], rationale="r"),
    )


@dataclass
class _FakePostmortemResult:
    decision: Decision
    plan: PostmortemPlan
    retry: Any
    postmortem_root: Path
    mode: str
    analyst_retries: int
    analyst_model_id: str
    total_cost_usd: float


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPostmortemToolDispatch:
    @pytest.mark.asyncio
    async def test_routes_to_injected_callback_and_summarizes_decision(
        self, handlers, tmp_path
    ):
        plan = _mkplan()
        decision = Decision(
            outcome="accepted",
            deltas=DecisionDeltas(
                global_delta=10.0,
                remediation_delta=25.0,
                max_scenario_regression=1.0,
                pre_best_score=40.0,
                post_best_score=50.0,
                retry_iterations_run=1,
            ),
            auto_applied_intervention_ids=["I1"],
            report_only_intervention_ids=[],
            rationale="thresholds met",
            thresholds_snapshot={"accept_threshold_global": 5.0},
        )
        captured: dict[str, Any] = {}

        async def cb(*, task_root, slug, mode, allow_on_cost_cap, task):
            captured.update(
                task_root=task_root, slug=slug, mode=mode,
                allow_on_cost_cap=allow_on_cost_cap,
                task_id=task.config.task_name,
            )
            return _FakePostmortemResult(
                decision=decision,
                plan=plan,
                retry=None,
                postmortem_root=tmp_path / "pm-task" / "runs" / "stub" / "postmortem",
                mode=mode,
                analyst_retries=0,
                analyst_model_id="claude-opus-4-5",
                total_cost_usd=0.42,
            )

        handlers._run_postmortem_cb = cb
        out = await handlers.call(
            "lpo_run_postmortem",
            {"task_id": "pm-task"},
        )
        # Handler must have dispatched to our callback with the resolved
        # arguments. task_id on the bundle is 'pm-task'.
        assert captured["task_id"] == "pm-task"
        assert captured["slug"] == "stub"  # defaulted to first target
        assert captured["mode"] == "autonomous"  # default
        assert captured["allow_on_cost_cap"] is False
        # Envelope summarizes the decision + deltas.
        assert out["outcome"] == "accepted"
        assert out["auto_applied_intervention_ids"] == ["I1"]
        assert out["report_only_intervention_ids"] == []
        assert out["finding_ids"] == ["F1"]
        assert out["analyst_model_id"] == "claude-opus-4-5"
        assert out["total_cost_usd"] == 0.42
        assert out["deltas"]["global"] == 10.0
        assert out["deltas"]["remediation"] == 25.0
        assert out["deltas"]["max_scenario_regression"] == 1.0
        assert out["rationale"] == "thresholds met"

    @pytest.mark.asyncio
    async def test_explicit_slug_and_mode_are_forwarded(self, handlers):
        captured: dict[str, Any] = {}

        async def cb(*, task_root, slug, mode, allow_on_cost_cap, task):
            captured.update(slug=slug, mode=mode, allow_on_cost_cap=allow_on_cost_cap)
            return _FakePostmortemResult(
                decision=Decision(
                    outcome="abstained",
                    deltas=None,
                    auto_applied_intervention_ids=[],
                    report_only_intervention_ids=[],
                    rationale="propose_only",
                ),
                plan=_mkplan(),
                retry=None,
                postmortem_root=Path("/tmp/x"),
                mode=mode,
                analyst_retries=0,
                analyst_model_id="m",
                total_cost_usd=0.0,
            )

        handlers._run_postmortem_cb = cb
        out = await handlers.call(
            "lpo_run_postmortem",
            {
                "task_id": "pm-task",
                "slug": "stub",
                "mode": "propose_only",
                "allow_on_cost_cap": True,
            },
        )
        assert captured == {
            "slug": "stub",
            "mode": "propose_only",
            "allow_on_cost_cap": True,
        }
        # abstained outcomes omit the deltas block.
        assert out["outcome"] == "abstained"
        assert "deltas" not in out

    @pytest.mark.asyncio
    async def test_unknown_task_returns_error_not_crash(self, handlers):
        # The MCP dispatcher converts exceptions into {error: ...} payloads
        # rather than propagating. Sanity-check this for a missing task
        # bundle — callers shouldn't be able to crash the server by asking
        # about a task that doesn't exist.
        out = await handlers.call("lpo_run_postmortem", {"task_id": "does_not_exist"})
        assert "error" in out
