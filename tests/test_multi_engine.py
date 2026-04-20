"""Stage 4 — multi-target orchestration tests.

Uses stub target clients and a no-op mutator so the ratchet runs without any
network calls. The aim is to pin down the *orchestration* contract:

* Strategy B runs N independent ratchet loops, one per target, each with its
  own run directory and its own winning prompt.
* Strategy C runs a single shared prompt against every target every iteration,
  combines scores by ``unified_aggregation``, and ratchets on that combined
  signal. Per-model breakdowns are persisted in the iteration's scores.json.
* The comparison report surfaces per-model results and a winner recommendation
  that matches the strategy.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from lpo.config.schema import (
    DeterministicMetric,
    DeterministicRule,
    RunConfig,
    TargetModelConfig,
)
from lpo.core.comparison import write_comparison_report
from lpo.core.cost import CostTracker
from lpo.core.multi_engine import (
    MultiResult,
    UNIFIED_SLUG,
    combine_scores,
    run_parallel_independent,
    run_unified_portable,
    validate_runtime,
)
from lpo.core.task import TaskBundle

EXAMPLE = Path(__file__).parent.parent / "tasks" / "example_json_extract"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _copy_example(tmp_path: Path) -> Path:
    dst = tmp_path / "task"
    shutil.copytree(EXAMPLE, dst)
    for sub in ("runs", "logs", "comparison"):
        p = dst / sub
        if p.exists():
            shutil.rmtree(p)
    return dst


def _patch_config(task_root: Path, **overrides) -> None:
    """Rewrite ``config.yaml`` with merged overrides. Keeps all un-overridden keys."""
    import yaml

    cfg_path = task_root / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg.update(overrides)
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def _stub_target(slug: str, mode: str, **extra) -> dict:
    return {
        "slug": slug,
        "provider": "stub",
        "model_id": f"stub/{slug}",
        "base_url": "",
        "stub_mode": mode,
        **extra,
    }


# ---------------------------------------------------------------------------
# combine_scores
# ---------------------------------------------------------------------------


def test_combine_min():
    assert combine_scores({"a": 80, "b": 60}, "min") == 60


def test_combine_mean():
    assert combine_scores({"a": 80, "b": 60}, "mean") == 70


def test_combine_weighted_mean():
    assert combine_scores({"a": 100, "b": 0}, "weighted_mean", {"a": 3.0, "b": 1.0}) == pytest.approx(75.0)


def test_combine_weighted_mean_requires_weights():
    with pytest.raises(ValueError):
        combine_scores({"a": 80}, "weighted_mean")


def test_combine_unknown_method():
    with pytest.raises(ValueError):
        combine_scores({"a": 80}, "median")


def test_combine_empty():
    assert combine_scores({}, "min") == 0.0


# ---------------------------------------------------------------------------
# validate_runtime
# ---------------------------------------------------------------------------


def _make_cfg(**over) -> RunConfig:
    base = {
        "task_name": "t",
        "target_strategy": "unified_portable",
        "target_models": [
            {"slug": "a", "provider": "stub", "model_id": "stub/a"},
            {"slug": "b", "provider": "stub", "model_id": "stub/b"},
        ],
    }
    base.update(over)
    return RunConfig.model_validate(base)


def test_validate_runtime_weighted_mean_requires_weights():
    cfg = _make_cfg(unified_aggregation="mean")
    # mean is fine without weights.
    validate_runtime(cfg)

    # weighted_mean without weights is rejected by the Pydantic schema already,
    # so we construct the bad state explicitly and use validate_runtime as the
    # runtime second line of defence.
    cfg2 = _make_cfg(
        unified_aggregation="weighted_mean",
        target_models=[
            {"slug": "a", "provider": "stub", "model_id": "stub/a", "weight": 1.0},
            {"slug": "b", "provider": "stub", "model_id": "stub/b", "weight": 1.0},
        ],
    )
    # Now drop one weight post-construction to exercise validate_runtime.
    cfg2.target_models[1].weight = None
    with pytest.raises(ValueError, match="weighted_mean"):
        validate_runtime(cfg2)


# ---------------------------------------------------------------------------
# Strategy B: parallel_independent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parallel_independent_runs_once_per_target(tmp_path):
    task_root = _copy_example(tmp_path)
    # Two stubs: one that ALWAYS returns valid gold JSON → scores high.
    # One that returns echo of the user input → scores low.
    _patch_config(
        task_root,
        target_strategy="parallel_independent",
        parallel_execution="sequential",
        target_models=[
            _stub_target("strong", "fixed", stub_fixed_text=_gold_for(task_root, "ex_001")),
            _stub_target("weak", "echo"),
        ],
        stop_conditions={"target_score": 999, "max_iterations": 1, "plateau_patience": 10, "cost_cap_usd": 1.0},
        metric_evolution={"enabled": False, "check_every_n_iterations": 5, "require_user_approval": True},
        mode="manual",  # disables Overseer (NullMutator fallback)
    )

    task = TaskBundle.load(task_root)
    cost = CostTracker()
    result = await run_parallel_independent(task, cost=cost, mutator_mode="null")

    assert result.strategy == "parallel_independent"
    assert {r.slug for r in result.per_model} == {"strong", "weak"}
    # Per-model artifacts created.
    assert (task_root / "runs" / "strong" / "prompt.txt.best").exists()
    assert (task_root / "runs" / "weak" / "prompt.txt.best").exists()
    assert (task_root / "runs" / "strong" / "winner" / "prompt.txt").exists()
    assert (task_root / "runs" / "weak" / "winner" / "prompt.txt").exists()
    # Strong target scored higher than weak (at least the ex_001 match).
    strong = next(r for r in result.per_model if r.slug == "strong")
    weak = next(r for r in result.per_model if r.slug == "weak")
    assert strong.best_score > weak.best_score


@pytest.mark.asyncio
async def test_parallel_independent_isolation(tmp_path):
    """Each model's run directory must be independent — no cross-contamination."""
    task_root = _copy_example(tmp_path)
    _patch_config(
        task_root,
        target_strategy="parallel_independent",
        parallel_execution="sequential",
        target_models=[
            _stub_target("a", "fixed", stub_fixed_text="alpha"),
            _stub_target("b", "fixed", stub_fixed_text="beta"),
        ],
        stop_conditions={"target_score": 999, "max_iterations": 1, "plateau_patience": 10, "cost_cap_usd": 1.0},
        metric_evolution={"enabled": False, "check_every_n_iterations": 5, "require_user_approval": True},
        mode="manual",
    )
    task = TaskBundle.load(task_root)
    cost = CostTracker()
    await run_parallel_independent(task, cost=cost, mutator_mode="null")

    # Each run's iter_0001/outputs.jsonl contains only its own stub output.
    for slug, expected_output in [("a", "alpha"), ("b", "beta")]:
        lines = (task_root / "runs" / slug / "history" / "iter_0001" / "outputs.jsonl").read_text(
            encoding="utf-8"
        ).strip().splitlines()
        for line in lines:
            row = json.loads(line)
            assert row["output"] == expected_output


# ---------------------------------------------------------------------------
# Strategy C: unified_portable
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unified_portable_single_shared_prompt(tmp_path):
    task_root = _copy_example(tmp_path)
    # Same stub for both targets so the combined score is deterministic.
    fixed = _gold_for(task_root, "ex_001")
    _patch_config(
        task_root,
        target_strategy="unified_portable",
        unified_aggregation="min",
        parallel_execution="sequential",
        target_models=[
            _stub_target("t1", "fixed", stub_fixed_text=fixed),
            _stub_target("t2", "fixed", stub_fixed_text=fixed),
        ],
        stop_conditions={"target_score": 999, "max_iterations": 1, "plateau_patience": 10, "cost_cap_usd": 1.0},
        metric_evolution={"enabled": False, "check_every_n_iterations": 5, "require_user_approval": True},
        mode="manual",
    )
    task = TaskBundle.load(task_root)
    cost = CostTracker()
    result = await run_unified_portable(task, cost=cost, mutator_mode="null")

    assert result.strategy == "unified_portable"
    # Single shared prompt directory.
    unified_root = task_root / "runs" / UNIFIED_SLUG
    assert (unified_root / "prompt.txt.best").exists()
    assert (unified_root / "winner" / "prompt.txt").exists()
    # No per-model run dirs for Strategy C.
    assert not (task_root / "runs" / "t1" / "prompt.txt.best").exists()
    # scores.json carries per-model breakdown.
    scores = json.loads(
        (unified_root / "history" / "iter_0001" / "scores.json").read_text(encoding="utf-8")
    )
    assert set(scores["per_model"].keys()) == {"t1", "t2"}
    assert "aggregate" in scores
    # outputs.jsonl rows are tagged with model_slug.
    rows = [
        json.loads(l)
        for l in (unified_root / "history" / "iter_0001" / "outputs.jsonl")
        .read_text(encoding="utf-8")
        .strip()
        .splitlines()
    ]
    assert {r["model_slug"] for r in rows} == {"t1", "t2"}
    # shared_best_score equals per_model best (both stubs identical).
    t1 = scores["per_model"]["t1"]["aggregate"]
    assert result.shared_best_score == pytest.approx(t1)


@pytest.mark.asyncio
async def test_unified_portable_min_vs_mean(tmp_path):
    """With one strong + one weak target, min < mean and both differ from either target's score."""
    task_root = _copy_example(tmp_path)
    _patch_config(
        task_root,
        target_strategy="unified_portable",
        unified_aggregation="min",
        parallel_execution="sequential",
        target_models=[
            _stub_target("strong", "fixed", stub_fixed_text=_gold_for(task_root, "ex_001")),
            _stub_target("weak", "echo"),
        ],
        stop_conditions={"target_score": 999, "max_iterations": 1, "plateau_patience": 10, "cost_cap_usd": 1.0},
        metric_evolution={"enabled": False, "check_every_n_iterations": 5, "require_user_approval": True},
        mode="manual",
    )
    task = TaskBundle.load(task_root)
    cost = CostTracker()

    result_min = await run_unified_portable(task, cost=cost, mutator_mode="null")
    min_combined = result_min.shared_best_score

    # Re-run with mean.
    shutil.rmtree(task_root / "runs")
    _patch_config(task_root, unified_aggregation="mean")
    task = TaskBundle.load(task_root)
    result_mean = await run_unified_portable(task, cost=CostTracker(), mutator_mode="null")
    mean_combined = result_mean.shared_best_score

    # Mean > min when targets are asymmetric.
    assert mean_combined > min_combined


# ---------------------------------------------------------------------------
# Comparison report
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_comparison_report_strategy_b(tmp_path):
    task_root = _copy_example(tmp_path)
    _patch_config(
        task_root,
        target_strategy="parallel_independent",
        parallel_execution="sequential",
        target_models=[
            _stub_target("strong", "fixed", stub_fixed_text=_gold_for(task_root, "ex_001")),
            _stub_target("weak", "echo"),
        ],
        stop_conditions={"target_score": 999, "max_iterations": 1, "plateau_patience": 10, "cost_cap_usd": 1.0},
        metric_evolution={"enabled": False, "check_every_n_iterations": 5, "require_user_approval": True},
        mode="manual",
    )
    task = TaskBundle.load(task_root)
    cost = CostTracker()
    result = await run_parallel_independent(task, cost=cost, mutator_mode="null")
    summary_path, report_path = write_comparison_report(task.root, task.config.task_name, result)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["strategy"] == "parallel_independent"
    assert {r["slug"] for r in summary["per_model"]} == {"strong", "weak"}
    assert summary["winner_recommendation"]["slug"] == "strong"

    report = report_path.read_text(encoding="utf-8")
    assert "Cross-model comparison" in report
    assert "`strong`" in report and "`weak`" in report
    # Both per-model winning prompts are embedded.
    assert "Winning prompts (per model)" in report


@pytest.mark.asyncio
async def test_comparison_report_strategy_c(tmp_path):
    task_root = _copy_example(tmp_path)
    _patch_config(
        task_root,
        target_strategy="unified_portable",
        unified_aggregation="min",
        parallel_execution="sequential",
        target_models=[
            _stub_target("t1", "fixed", stub_fixed_text=_gold_for(task_root, "ex_001")),
            _stub_target("t2", "fixed", stub_fixed_text=_gold_for(task_root, "ex_001")),
        ],
        stop_conditions={"target_score": 999, "max_iterations": 1, "plateau_patience": 10, "cost_cap_usd": 1.0},
        metric_evolution={"enabled": False, "check_every_n_iterations": 5, "require_user_approval": True},
        mode="manual",
    )
    task = TaskBundle.load(task_root)
    cost = CostTracker()
    result = await run_unified_portable(task, cost=cost, mutator_mode="null")
    summary_path, report_path = write_comparison_report(task.root, task.config.task_name, result)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["strategy"] == "unified_portable"
    assert "shared_best_score" in summary
    assert summary["winner_recommendation"]["slug"] == "(shared prompt)"
    report = report_path.read_text(encoding="utf-8")
    assert "Shared winning prompt" in report
    # Strategy C should NOT include per-model prompts block.
    assert "Winning prompts (per model)" not in report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gold_for(task_root: Path, id_: str) -> str:
    """Return the gold JSON string for the given eval id as the stub's fixed text."""
    for line in (task_root / "gold_standard.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec["id"] == id_:
            return json.dumps(rec["output"], ensure_ascii=False)
    raise KeyError(id_)
