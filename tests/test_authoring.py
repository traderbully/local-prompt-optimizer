"""Unit tests for :mod:`lpo.core.authoring`.

Focused on disk contents + round-trip loadability. Gold-standard generation
is exercised with a stub client so we don't require ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from lpo.core.authoring import (
    TargetSpec,
    create_task_bundle,
    generate_gold_standard,
)
from lpo.core.task import TaskBundle


# ---------------------------------------------------------------------------
# create_task_bundle
# ---------------------------------------------------------------------------


def test_create_task_bundle_text_default(tmp_path: Path):
    path = create_task_bundle(
        tmp_path,
        name="Haiku Writer",
        task_description="Compose a three-line haiku about the given topic.",
        example_inputs=["morning fog", "autumn leaves", "busy city"],
        output_type="text",
    )
    assert path.name == "Haiku_Writer"
    assert (path / "config.yaml").exists()
    assert (path / "metric.yaml").exists()
    assert (path / "eval_set.jsonl").exists()
    assert (path / "prompt_seed.txt").exists()
    assert (path / "task.md").exists()
    # Gold is NOT generated automatically.
    assert not (path / "gold_standard.jsonl").exists()

    # Round-trip through the real loader.
    task = TaskBundle.load(path)
    assert task.config.task_name == "Haiku Writer"
    assert task.config.target_strategy == "single"
    assert task.config.output_type == "text"
    assert task.metric.type == "rubric"
    assert len(task.eval_records) == 3
    assert task.eval_records[0].id == "ex001"


def test_create_task_bundle_json_with_required_fields(tmp_path: Path):
    path = create_task_bundle(
        tmp_path,
        name="extract_contact",
        task_description="Extract name and email.",
        example_inputs=[{"text": "Hi I'm Alice (alice@example.com)"}],
        output_type="json",
        required_json_fields=["name", "email"],
    )
    task = TaskBundle.load(path)
    assert task.metric.type == "deterministic"
    rule_names = {r.name for r in task.metric.rules}
    assert {"json_valid", "required_fields_present", "field_exact_match"} <= rule_names


def test_create_task_bundle_rejects_existing(tmp_path: Path):
    create_task_bundle(
        tmp_path, name="dup", task_description="x", example_inputs=["a"],
    )
    with pytest.raises(FileExistsError):
        create_task_bundle(
            tmp_path, name="dup", task_description="x", example_inputs=["a"],
        )


def test_create_task_bundle_overwrite_ok(tmp_path: Path):
    create_task_bundle(
        tmp_path, name="dup", task_description="x", example_inputs=["a"],
    )
    # Overwrite with different inputs; should succeed and replace eval set.
    path = create_task_bundle(
        tmp_path,
        name="dup",
        task_description="new",
        example_inputs=["x1", "x2"],
        overwrite=True,
    )
    task = TaskBundle.load(path)
    assert len(task.eval_records) == 2
    assert "new" in (path / "task.md").read_text(encoding="utf-8")


def test_create_task_bundle_scenario_tags_applied(tmp_path: Path):
    path = create_task_bundle(
        tmp_path,
        name="classify",
        task_description="Classify the sentence.",
        example_inputs=["A", "B", "C"],
        scenario_tags=["positive", "negative", None],
    )
    task = TaskBundle.load(path)
    assert task.eval_records[0].scenario == "positive"
    assert task.eval_records[1].scenario == "negative"
    assert task.eval_records[2].scenario is None


def test_create_task_bundle_multi_target_strategy(tmp_path: Path):
    path = create_task_bundle(
        tmp_path,
        name="multi",
        task_description="Any task.",
        example_inputs=["a", "b"],
        strategy="parallel_independent",
        targets=[
            TargetSpec(slug="model-a", model_id="a"),
            TargetSpec(slug="model-b", model_id="b"),
        ],
    )
    task = TaskBundle.load(path)
    assert task.config.target_strategy == "parallel_independent"
    assert [m.slug for m in task.config.target_models] == ["model-a", "model-b"]


def test_create_task_bundle_strategy_target_mismatch(tmp_path: Path):
    with pytest.raises(ValueError):
        create_task_bundle(
            tmp_path,
            name="bad",
            task_description="x",
            example_inputs=["a"],
            strategy="parallel_independent",
            targets=[TargetSpec(slug="only")],
        )


def test_create_task_bundle_empty_examples_rejected(tmp_path: Path):
    with pytest.raises(ValueError):
        create_task_bundle(
            tmp_path, name="z", task_description="x", example_inputs=[],
        )


# ---------------------------------------------------------------------------
# generate_gold_standard — stub client
# ---------------------------------------------------------------------------


class _StubResult:
    def __init__(self, text: str):
        self.text = text
        self.prompt_tokens = 1
        self.completion_tokens = 1
        self.model_id = "stub"
        self.stop_reason = "end_turn"
        self.raw: dict[str, Any] = {}


class _StubClient:
    """Deterministic substitute for AnthropicClient used by the gold-standard
    generator. Records every call and returns a canned output derived from
    the input so tests can assert per-example behavior."""

    def __init__(self, *, json_mode: bool = False):
        self.calls: list[str] = []
        self._json_mode = json_mode

    async def complete(self, *, system: str, messages: list[Any], **kw: Any) -> _StubResult:
        assert "Gold Standard Source" in system
        text = messages[-1].content
        self.calls.append(text)
        if self._json_mode:
            # Echo the input text as a trivial JSON object.
            return _StubResult(json.dumps({"echo": "ok"}))
        # Extract a short deterministic answer.
        return _StubResult(f"GOLD[{len(self.calls)}]")

    async def aclose(self) -> None:
        pass


@pytest.mark.asyncio
async def test_generate_gold_standard_writes_jsonl(tmp_path: Path):
    path = create_task_bundle(
        tmp_path,
        name="sum",
        task_description="Return a one-line summary.",
        example_inputs=["doc A", "doc B", "doc C"],
    )
    stub = _StubClient()
    n = await generate_gold_standard(path, client=stub)
    assert n == 3
    assert len(stub.calls) == 3
    gold_path = path / "gold_standard.jsonl"
    assert gold_path.exists()
    lines = gold_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    first = json.loads(lines[0])
    assert first["id"] == "ex001"
    assert first["output"] == "GOLD[1]"

    # Round-trip: TaskBundle should now report gold present.
    task = TaskBundle.load(path)
    assert len(task.gold_standard) == 3


@pytest.mark.asyncio
async def test_generate_gold_standard_json_parses_when_valid(tmp_path: Path):
    path = create_task_bundle(
        tmp_path,
        name="extract",
        task_description="Extract fields.",
        example_inputs=[{"x": 1}],
        output_type="json",
        required_json_fields=["echo"],
    )
    stub = _StubClient(json_mode=True)
    await generate_gold_standard(path, client=stub)
    task = TaskBundle.load(path)
    # For output_type=json the authoring layer parses the text into structure.
    assert task.gold_standard["ex001"].output == {"echo": "ok"}


@pytest.mark.asyncio
async def test_generate_gold_standard_no_overwrite_is_idempotent(tmp_path: Path):
    path = create_task_bundle(
        tmp_path, name="idem", task_description="x", example_inputs=["a"],
    )
    stub = _StubClient()
    n1 = await generate_gold_standard(path, client=stub)
    # Second call without overwrite should be a no-op: no extra client calls.
    stub2 = _StubClient()
    n2 = await generate_gold_standard(path, client=stub2)
    assert n1 == 1
    assert n2 == 1
    assert stub2.calls == []


@pytest.mark.asyncio
async def test_generate_gold_standard_overwrite_regenerates(tmp_path: Path):
    path = create_task_bundle(
        tmp_path, name="reg", task_description="x", example_inputs=["a"],
    )
    await generate_gold_standard(path, client=_StubClient())
    stub2 = _StubClient()
    await generate_gold_standard(path, client=stub2, overwrite=True)
    assert len(stub2.calls) == 1
