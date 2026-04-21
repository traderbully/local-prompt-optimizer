"""Stage 8 — Postmortem artifact loader tests.

Covers the loader's ability to round-trip a real engine-written history
layout into typed :class:`RunHistoryBundle` objects. A fixture writes out
the same filenames that ``lpo.core.history.IterationRecord.write`` and
``lpo.core.multi_engine._write_unified_iteration`` produce, so if the
engine's on-disk format drifts these tests will catch the regression.

Both single-target and unified (Strategy C) layouts are exercised because
the Analyst will be asked to reason about both.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lpo.postmortem.artifacts import (
    IterationArtifact,
    RunHistoryBundle,
    load_run_history,
)


# ---------------------------------------------------------------------------
# Fixture helpers — write a minimal-but-realistic task + run layout.
# ---------------------------------------------------------------------------


def _write_task_files(root: Path, *, slug: str = "s1") -> None:
    (root / "task.md").write_text("A task.", encoding="utf-8")
    (root / "prompt_seed.txt").write_text("Seed prompt.", encoding="utf-8")
    (root / "eval_set.jsonl").write_text(
        '{"id": "ex001", "input": "q1", "scenario": "easy"}\n'
        '{"id": "ex002", "input": "q2", "scenario": "hard"}\n',
        encoding="utf-8",
    )
    (root / "gold_standard.jsonl").write_text(
        '{"id": "ex001", "output": "a1"}\n'
        '{"id": "ex002", "output": "a2"}\n',
        encoding="utf-8",
    )
    (root / "metric.yaml").write_text(
        "type: deterministic\nrules:\n  - name: r1\n    weight: 100\n    check: is_valid_json\n",
        encoding="utf-8",
    )
    (root / "config.yaml").write_text(
        "task_name: test\n"
        "target_models:\n"
        f"  - slug: {slug}\n"
        "    model_id: some-model\n"
        "    provider: lmstudio\n"
        "target_strategy: single\n",
        encoding="utf-8",
    )


def _write_iteration(
    iter_dir: Path,
    *,
    index: int,
    prompt: str,
    aggregate: float,
    per_example: dict[str, float],
    per_scenario: dict[str, float] | None = None,
    failed_ids: list[str] | None = None,
    decision: str = "accepted",
    delta: float = 0.0,
    cost_usd: float = 0.01,
    overseer_analysis: str | None = None,
    outputs: list[dict] | None = None,
    per_model: dict | None = None,
) -> None:
    iter_dir.mkdir(parents=True, exist_ok=True)
    (iter_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    outputs = outputs if outputs is not None else [
        {"id": ex_id, "output": f"out-{ex_id}"} for ex_id in per_example.keys()
    ]
    (iter_dir / "outputs.jsonl").write_text(
        "\n".join(json.dumps(o) for o in outputs) + "\n",
        encoding="utf-8",
    )
    scores_blob: dict = {
        "aggregate": aggregate,
        "per_example": per_example,
        "per_scenario": per_scenario or {},
        "failed_ids": failed_ids or [],
    }
    if per_model is not None:
        scores_blob["per_model"] = per_model
    (iter_dir / "scores.json").write_text(json.dumps(scores_blob), encoding="utf-8")
    (iter_dir / "decision.json").write_text(
        json.dumps({
            "decision": decision,
            "delta": delta,
            "timings": {"total_s": 1.5},
            "cost_usd": cost_usd,
            "timestamp": "2026-04-21T01:30:00+00:00",
            "notes": "",
        }),
        encoding="utf-8",
    )
    if overseer_analysis is not None:
        (iter_dir / "overseer_analysis.md").write_text(overseer_analysis, encoding="utf-8")


@pytest.fixture
def built_run(tmp_path: Path) -> tuple[Path, str]:
    """A three-iteration single-target run with a winner dir."""
    root = tmp_path / "task1"
    root.mkdir()
    _write_task_files(root, slug="stub")
    history = root / "runs" / "stub" / "history"
    _write_iteration(
        history / "iter_0001",
        index=1,
        prompt="Prompt v1",
        aggregate=43.3,
        per_example={"ex001": 43.3, "ex002": 43.3},
        per_scenario={"easy": 43.3, "hard": 43.3},
        failed_ids=[],
        decision="initial",
        delta=43.3,
    )
    _write_iteration(
        history / "iter_0002",
        index=2,
        prompt="Prompt v2",
        aggregate=55.0,
        per_example={"ex001": 60.0, "ex002": 50.0},
        per_scenario={"easy": 60.0, "hard": 50.0},
        failed_ids=[],
        decision="accepted",
        delta=11.7,
        overseer_analysis="The Overseer thought deeply about this.",
    )
    _write_iteration(
        history / "iter_0003",
        index=3,
        prompt="Prompt v3",
        aggregate=48.0,
        per_example={"ex001": 50.0, "ex002": 46.0},
        per_scenario={"easy": 50.0, "hard": 46.0},
        failed_ids=["ex002"],
        decision="rejected",
        delta=-7.0,
        overseer_analysis="A rejected attempt.",
    )
    winner_dir = root / "runs" / "stub" / "winner"
    winner_dir.mkdir(parents=True)
    (winner_dir / "prompt.txt").write_text("Prompt v2", encoding="utf-8")
    (winner_dir / "report.md").write_text("# Winner\nProvenance: ratchet", encoding="utf-8")
    return root, "stub"


# ---------------------------------------------------------------------------
# Loader — happy path
# ---------------------------------------------------------------------------


class TestLoaderHappyPath:
    def test_loads_all_three_iterations(self, built_run):
        root, slug = built_run
        bundle = load_run_history(root, slug)
        assert isinstance(bundle, RunHistoryBundle)
        assert bundle.slug == "stub"
        assert bundle.total_iterations == 3
        assert [it.index for it in bundle.iterations] == [1, 2, 3]

    def test_iteration_prompt_and_outputs_round_trip(self, built_run):
        root, slug = built_run
        bundle = load_run_history(root, slug)
        it2 = bundle.iterations[1]
        assert it2.prompt == "Prompt v2"
        # outputs.jsonl — one row per example.
        assert {row["id"] for row in it2.outputs} == {"ex001", "ex002"}

    def test_scores_aggregate_and_breakdowns_parsed(self, built_run):
        root, slug = built_run
        bundle = load_run_history(root, slug)
        it2 = bundle.iterations[1]
        assert it2.scores.aggregate == pytest.approx(55.0)
        assert it2.scores.per_example == {"ex001": 60.0, "ex002": 50.0}
        assert it2.scores.per_scenario == {"easy": 60.0, "hard": 50.0}
        assert it2.scores.failed_ids == []
        assert it2.scores.per_model == {}  # single-target run

    def test_decision_fields_parsed(self, built_run):
        root, slug = built_run
        bundle = load_run_history(root, slug)
        it1 = bundle.iterations[0]
        assert it1.decision.decision == "initial"
        assert it1.decision.delta == pytest.approx(43.3)
        assert it1.decision.cost_usd == pytest.approx(0.01)
        assert it1.decision.timings == {"total_s": 1.5}

    def test_overseer_analysis_absent_on_first_iter_present_on_later(self, built_run):
        root, slug = built_run
        bundle = load_run_history(root, slug)
        assert bundle.iterations[0].overseer_analysis is None
        assert "Overseer" in (bundle.iterations[1].overseer_analysis or "")
        assert "rejected" in (bundle.iterations[2].overseer_analysis or "")

    def test_winner_artifacts_loaded(self, built_run):
        root, slug = built_run
        bundle = load_run_history(root, slug)
        assert bundle.winner_prompt == "Prompt v2"
        assert bundle.winner_report is not None
        assert "Provenance" in bundle.winner_report

    def test_best_iteration_prefers_highest_score(self, built_run):
        # Iter 2 scored 55.0, iter 1 scored 43.3, iter 3 scored 48.0 —
        # best_iteration must return iter 2.
        root, slug = built_run
        bundle = load_run_history(root, slug)
        best = bundle.best_iteration
        assert best is not None
        assert best.index == 2

    def test_best_iteration_breaks_ties_by_earliest_index(self, tmp_path):
        # Two iterations tied at 50.0 — the ratchet considers the earlier
        # one the winner (first-to-reach). best_iteration must agree.
        root = tmp_path / "tied"
        root.mkdir()
        _write_task_files(root, slug="tied")
        history = root / "runs" / "tied" / "history"
        _write_iteration(
            history / "iter_0001",
            index=1,
            prompt="p1",
            aggregate=50.0,
            per_example={"ex001": 50.0},
        )
        _write_iteration(
            history / "iter_0002",
            index=2,
            prompt="p2",
            aggregate=50.0,
            per_example={"ex001": 50.0},
        )
        bundle = load_run_history(root, "tied")
        assert bundle.best_iteration is not None
        assert bundle.best_iteration.index == 1


# ---------------------------------------------------------------------------
# Loader — unified Strategy C layout
# ---------------------------------------------------------------------------


class TestUnifiedLayoutPassthrough:
    def test_per_model_block_preserved(self, tmp_path):
        # Strategy C writes scores.json with a per_model dict and
        # combined_per_example / combined_per_scenario keys instead of
        # the plain names. Loader must accept both shapes.
        root = tmp_path / "unified"
        root.mkdir()
        _write_task_files(root, slug="unified")
        history = root / "runs" / "unified" / "history"
        iter_dir = history / "iter_0001"
        iter_dir.mkdir(parents=True)
        (iter_dir / "prompt.txt").write_text("p", encoding="utf-8")
        (iter_dir / "outputs.jsonl").write_text(
            json.dumps({"id": "ex001", "output": "x", "model_slug": "lms"}) + "\n",
            encoding="utf-8",
        )
        (iter_dir / "scores.json").write_text(
            json.dumps({
                "aggregate": 42.0,
                "aggregation_method": "min",
                "combined_per_example": {"ex001": 42.0},
                "combined_per_scenario": {"easy": 42.0},
                "failed_ids": [],
                "per_model": {
                    "lms": {"aggregate": 50.0, "per_example": {"ex001": 50.0}, "per_scenario": {}, "failed_ids": []},
                    "or":  {"aggregate": 42.0, "per_example": {"ex001": 42.0}, "per_scenario": {}, "failed_ids": []},
                },
            }),
            encoding="utf-8",
        )
        (iter_dir / "decision.json").write_text(
            json.dumps({"decision": "initial", "delta": 42.0, "timings": {}, "cost_usd": 0.0, "timestamp": "", "notes": ""}),
            encoding="utf-8",
        )
        bundle = load_run_history(root, "unified")
        assert bundle.iterations[0].scores.per_example == {"ex001": 42.0}  # from combined_*
        assert bundle.iterations[0].scores.per_scenario == {"easy": 42.0}
        assert set(bundle.iterations[0].scores.per_model.keys()) == {"lms", "or"}
        assert bundle.iterations[0].scores.per_model["lms"]["aggregate"] == 50.0


# ---------------------------------------------------------------------------
# Loader — tolerance for partial / missing artifacts
# ---------------------------------------------------------------------------


class TestLoaderTolerance:
    def test_run_with_no_iterations_returns_empty_list(self, tmp_path):
        # An engine run that crashed before writing any iteration should
        # still produce a valid bundle — just with .iterations == [].
        root = tmp_path / "empty"
        root.mkdir()
        _write_task_files(root, slug="empty")
        (root / "runs" / "empty").mkdir(parents=True)
        bundle = load_run_history(root, "empty")
        assert bundle.iterations == []
        assert bundle.winner_prompt is None
        assert bundle.best_iteration is None

    def test_missing_slug_raises(self, tmp_path):
        root = tmp_path / "missing"
        root.mkdir()
        _write_task_files(root, slug="exists")
        # runs/<slug>/ doesn't exist at all.
        with pytest.raises(FileNotFoundError):
            load_run_history(root, "does_not_exist")

    def test_malformed_jsonl_lines_skipped_not_raised(self, tmp_path):
        # A partial-write outputs.jsonl with one good line and one garbage
        # line shouldn't kill the loader — Stage 8 runs on completed tasks
        # but we'd rather be lenient on ex-post reads than brittle.
        root = tmp_path / "partial"
        root.mkdir()
        _write_task_files(root, slug="partial")
        iter_dir = root / "runs" / "partial" / "history" / "iter_0001"
        iter_dir.mkdir(parents=True)
        (iter_dir / "prompt.txt").write_text("p", encoding="utf-8")
        (iter_dir / "outputs.jsonl").write_text(
            json.dumps({"id": "ex001", "output": "x"}) + "\n"
            "this is not valid json\n",
            encoding="utf-8",
        )
        (iter_dir / "scores.json").write_text('{"aggregate": 10.0}', encoding="utf-8")
        (iter_dir / "decision.json").write_text('{"decision": "initial"}', encoding="utf-8")
        bundle = load_run_history(root, "partial")
        outputs = bundle.iterations[0].outputs
        assert len(outputs) == 1
        assert outputs[0]["id"] == "ex001"

    def test_iter_directories_with_nonmatching_names_ignored(self, tmp_path):
        # A stray directory under history/ that isn't iter_NNNN must not
        # derail the sort order or the loader.
        root = tmp_path / "stray"
        root.mkdir()
        _write_task_files(root, slug="stray")
        history = root / "runs" / "stray" / "history"
        _write_iteration(
            history / "iter_0001",
            index=1,
            prompt="p",
            aggregate=10.0,
            per_example={"ex001": 10.0},
        )
        (history / "notes").mkdir()
        (history / "notes" / "random.txt").write_text("stray file", encoding="utf-8")
        bundle = load_run_history(root, "stray")
        assert [it.index for it in bundle.iterations] == [1]
