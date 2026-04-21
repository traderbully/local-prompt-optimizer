from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from lpo.core.engine import RatchetEngine, StopReason
from lpo.core.mutator import MutationProposal, PromptMutator
from lpo.core.task import TaskBundle
from lpo.scoring.deterministic import build_scorer
from tests.conftest import StubClient


EXAMPLE = Path(__file__).resolve().parents[1] / "tasks" / "example_json_extract"


def _make_gold_responder():
    """Return a responder that outputs the exact gold JSON for each eval id."""
    gold_path = EXAMPLE / "gold_standard.jsonl"
    gold_by_input: dict[str, str] = {}
    eval_path = EXAMPLE / "eval_set.jsonl"
    inputs = {json.loads(line)["id"]: json.loads(line)["input"] for line in eval_path.read_text(encoding="utf-8").splitlines() if line.strip()}
    outputs = {json.loads(line)["id"]: json.loads(line)["output"] for line in gold_path.read_text(encoding="utf-8").splitlines() if line.strip()}
    for eid, text in inputs.items():
        gold_by_input[text] = json.dumps(outputs[eid])

    def responder(_system: str, user: str, _seed: int | None) -> str:
        return gold_by_input[user]

    return responder


@pytest.mark.asyncio
async def test_engine_hits_target_score_on_perfect_outputs(tmp_path):
    # Copy example task into tmp dir so the run doesn't pollute the repo
    import shutil

    task_root = tmp_path / "example_json_extract"
    shutil.copytree(EXAMPLE, task_root)

    task = TaskBundle.load(task_root)
    target = task.config.target_models[0]
    scorer = build_scorer(task.metric)
    client = StubClient(responder=_make_gold_responder())

    engine = RatchetEngine(task=task, target_cfg=target, client=client, scorer=scorer)
    result = await engine.run()

    assert result.stop_reason == StopReason.TARGET_SCORE
    assert result.best_score == pytest.approx(100.0)
    # Iteration 1 should have been accepted as initial
    assert result.iterations[0].decision == "initial"
    # Winner artifacts written
    assert (engine.paths.winner_root / "prompt.txt").read_text().strip() != ""
    assert (engine.paths.winner_root / "report.md").exists()


class _RegressingMutator(PromptMutator):
    """First proposes a worse prompt (empty), then returns best to trigger no-op."""

    def __init__(self) -> None:
        self.calls = 0

    async def propose(self, *, current_prompt, best_prompt, history, user_feedback=""):
        self.calls += 1
        if self.calls == 1:
            return MutationProposal(new_prompt="(broken)\n", rationale="test")
        return MutationProposal(new_prompt=best_prompt, rationale="revert")


@pytest.mark.asyncio
async def test_engine_reverts_on_regression(tmp_path):
    import shutil

    task_root = tmp_path / "task"
    shutil.copytree(EXAMPLE, task_root)
    # Raise max_iterations to give the mutator a chance to run
    cfg_path = task_root / "config.yaml"
    cfg_path.write_text(cfg_path.read_text().replace("max_iterations: 10", "max_iterations: 3").replace("target_score: 95", "target_score: 999"))

    task = TaskBundle.load(task_root)
    target = task.config.target_models[0]
    scorer = build_scorer(task.metric)

    # Responder returns perfect output when prompted with the seed prompt,
    # garbage when the mutator has broken the prompt.
    gold_responder = _make_gold_responder()

    def responder(system: str, user: str, seed: int | None) -> str:
        if "(broken)" in system:
            return "not json"
        return gold_responder(system, user, seed)

    client = StubClient(responder=responder)
    mutator = _RegressingMutator()
    engine = RatchetEngine(
        task=task, target_cfg=target, client=client, scorer=scorer, mutator=mutator
    )
    result = await engine.run()

    # Best score stays at 100 from iteration 1; iteration 2 is rejected.
    assert result.best_score == pytest.approx(100.0)
    assert result.iterations[0].decision == "initial"
    assert result.iterations[1].decision == "rejected"
    # prompt.txt should have been reverted to the best prompt after rejection
    assert engine.paths.current_prompt.read_text() == engine.paths.best_prompt.read_text()


@pytest.mark.asyncio
async def test_atomic_writes_produce_expected_files(tmp_path):
    import shutil

    task_root = tmp_path / "task"
    shutil.copytree(EXAMPLE, task_root)
    task = TaskBundle.load(task_root)
    target = task.config.target_models[0]
    scorer = build_scorer(task.metric)
    client = StubClient(responder=_make_gold_responder())
    engine = RatchetEngine(task=task, target_cfg=target, client=client, scorer=scorer)
    await engine.run()

    iter1 = engine.paths.iteration_dir(1)
    assert (iter1 / "prompt.txt").exists()
    assert (iter1 / "outputs.jsonl").exists()
    assert (iter1 / "scores.json").exists()
    assert (iter1 / "decision.json").exists()
    scores = json.loads((iter1 / "scores.json").read_text())
    assert scores["aggregate"] == pytest.approx(100.0)

    # Stage 7 forensic follow-up: every row in outputs.jsonl must carry
    # finish_reason and reasoning_tokens (both may be null for the stub
    # client, but the KEYS must be present so operators can grep for
    # truncation patterns without forensic re-runs).
    rows = [
        json.loads(line)
        for line in (iter1 / "outputs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows, "expected at least one output row"
    for row in rows:
        assert "finish_reason" in row
        assert "reasoning_tokens" in row
