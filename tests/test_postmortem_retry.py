"""Focused-retry tests.

We never touch the network. A stub IterationRunner is injected via
``runner_factory`` that produces a deterministic :class:`IterationResult`
and records the prompt it was asked to score. This gives us full
control over the measured scores without any real generation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from lpo.core.history import IterationRecord
from lpo.core.iteration import IterationResult
from lpo.core.task import TaskBundle
from lpo.postmortem.retry import FocusedRetryResult, run_focused_retry
from lpo.postmortem.schemas import PostmortemConfig
from lpo.scoring.aggregation import AggregatedScore

from tests.test_postmortem_artifacts import _write_iteration, _write_task_files


# ---------------------------------------------------------------------------
# Stub runner — records calls, returns canned iteration results.
# ---------------------------------------------------------------------------


@dataclass
class _StubRunner:
    canned_score: float
    per_example: dict[str, float]
    per_scenario: dict[str, float]
    calls: list[dict[str, Any]]

    async def run(
        self,
        *,
        iteration_index: int,
        prompt: str,
        eval_records,
        gold_standard,
        task_name,
    ) -> IterationResult:
        self.calls.append({
            "iteration_index": iteration_index,
            "prompt": prompt,
            "task_name": task_name,
            "eval_count": len(eval_records),
        })
        record = IterationRecord(
            index=iteration_index,
            prompt=prompt,
            aggregate_score=self.canned_score,
            per_example=dict(self.per_example),
            per_scenario=dict(self.per_scenario),
            failed_ids=[],
            outputs=[
                {"id": rec_id, "output": f"stub-{rec_id}", "score": score}
                for rec_id, score in self.per_example.items()
            ],
            decision="pending",
        )
        agg = AggregatedScore(
            aggregate=self.canned_score,
            per_example=dict(self.per_example),
            per_scenario=dict(self.per_scenario),
            failed_ids=[],
        )
        return IterationResult(record=record, aggregated=agg)


@pytest.fixture
def built_task(tmp_path: Path) -> TaskBundle:
    root = tmp_path / "task1"
    root.mkdir()
    _write_task_files(root, slug="stub")
    # Write at least one historical iteration so the run layout exists.
    history = root / "runs" / "stub" / "history"
    _write_iteration(
        history / "iter_0001",
        index=1,
        prompt="p1",
        aggregate=30.0,
        per_example={"ex001": 30.0, "ex002": 30.0},
    )
    return TaskBundle.load(root)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFocusedRetry:
    @pytest.mark.asyncio
    async def test_runs_single_iteration_with_patched_prompt(self, built_task, tmp_path):
        from lpo.core.cost import CostTracker

        stub = _StubRunner(
            canned_score=55.0,
            per_example={"ex001": 60.0, "ex002": 50.0},
            per_scenario={"easy": 60.0, "hard": 50.0},
            calls=[],
        )

        def factory(task, target_cfg, cost):
            return stub, None

        result = await run_focused_retry(
            built_task,
            slug="stub",
            target_cfg=built_task.config.target_models[0],
            patched_prompt="PATCHED PROMPT BODY",
            cfg=PostmortemConfig(),
            cost=CostTracker(),
            runner_factory=factory,
        )
        assert isinstance(result, FocusedRetryResult)
        assert result.best_score == pytest.approx(55.0)
        # v1 runs exactly one iteration even with max_retry_iterations=3.
        assert len(stub.calls) == 1
        assert stub.calls[0]["prompt"] == "PATCHED PROMPT BODY"
        assert stub.calls[0]["iteration_index"] == 1

    @pytest.mark.asyncio
    async def test_writes_iteration_artifacts_in_postmortem_retry_directory(self, built_task):
        from lpo.core.cost import CostTracker

        stub = _StubRunner(
            canned_score=42.0,
            per_example={"ex001": 50.0, "ex002": 34.0},
            per_scenario={"easy": 50.0, "hard": 34.0},
            calls=[],
        )

        def factory(task, target_cfg, cost):
            return stub, None

        result = await run_focused_retry(
            built_task,
            slug="stub",
            target_cfg=built_task.config.target_models[0],
            patched_prompt="P",
            cfg=PostmortemConfig(),
            cost=CostTracker(),
            runner_factory=factory,
        )
        retry_root = result.retry_root
        # The four engine-compatible files land inside iter_0001/.
        iter_dir = retry_root / "iter_0001"
        for name in ("prompt.txt", "outputs.jsonl", "scores.json", "decision.json"):
            assert (iter_dir / name).exists(), f"missing {name}"
        scores = json.loads((iter_dir / "scores.json").read_text(encoding="utf-8"))
        assert scores["aggregate"] == pytest.approx(42.0)
        # The patched prompt is also written to postmortem/retry/prompt.txt
        # as a sibling so operators can diff it without opening the iter dir.
        assert (retry_root / "prompt.txt").read_text(encoding="utf-8") == "P"

    @pytest.mark.asyncio
    async def test_cost_tracked_before_and_after(self, built_task):
        from lpo.core.cost import CostTracker

        cost = CostTracker()
        # Prime with some prior spend so we verify delta math not raw totals.
        cost.record("seed-model", prompt_tokens=100, completion_tokens=100)
        prior = cost.total_usd

        class _ChargingRunner(_StubRunner):
            async def run(self, **kwargs):
                # Simulate the scorer charging 0.005 for the retry.
                cost.record("retry-model", prompt_tokens=500, completion_tokens=500)
                return await super().run(**kwargs)

        stub = _ChargingRunner(
            canned_score=50.0,
            per_example={"ex001": 50.0, "ex002": 50.0},
            per_scenario={"easy": 50.0, "hard": 50.0},
            calls=[],
        )

        def factory(task, target_cfg, cost_unused):
            return stub, None

        # Register a rate so the charge is > 0.
        cost.set_rate("retry-model", input_per_mtok=10.0, output_per_mtok=10.0)

        result = await run_focused_retry(
            built_task,
            slug="stub",
            target_cfg=built_task.config.target_models[0],
            patched_prompt="P",
            cfg=PostmortemConfig(),
            cost=cost,
            runner_factory=factory,
        )
        assert result.cost_before_retry_usd == pytest.approx(prior)
        assert result.cost_after_retry_usd > result.cost_before_retry_usd
        assert result.cost_usd > 0

    @pytest.mark.asyncio
    async def test_cleanup_is_called_when_factory_returns_one(self, built_task):
        from lpo.core.cost import CostTracker

        stub = _StubRunner(
            canned_score=50.0,
            per_example={"ex001": 50.0, "ex002": 50.0},
            per_scenario={"easy": 50.0, "hard": 50.0},
            calls=[],
        )

        cleanup_called = {"count": 0}

        async def cleanup():
            cleanup_called["count"] += 1

        def factory(task, target_cfg, cost):
            return stub, cleanup

        await run_focused_retry(
            built_task,
            slug="stub",
            target_cfg=built_task.config.target_models[0],
            patched_prompt="P",
            cfg=PostmortemConfig(),
            cost=CostTracker(),
            runner_factory=factory,
        )
        assert cleanup_called["count"] == 1
