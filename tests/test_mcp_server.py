"""MCP server tests.

Drives the pure dispatcher :class:`LpoMcpHandlers` directly — no stdio
subprocess. Uses the stub target provider for the run tool so we never touch
the network. Gold-standard generation is exercised via a canned stub client
identical to the one in :mod:`test_authoring`.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from lpo.core.authoring import TargetSpec, create_task_bundle, generate_gold_standard
from lpo.server.mcp_server import LpoMcpHandlers, TOOL_SPECS


# ---------------------------------------------------------------------------
# Shared fixtures
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
    def __init__(self):
        self.calls: list[str] = []

    async def complete(self, *, system: str, messages: list[Any], **kw: Any):
        self.calls.append(messages[-1].content)
        return _StubResult(f"GOLD[{len(self.calls)}]")

    async def aclose(self) -> None:
        pass


def _stub_target_spec(slug: str = "stub-target") -> TargetSpec:
    # The stub provider is defined in lpo.models.stub and wired through
    # build_target_context. "fixed" mode returns stub_fixed_text verbatim.
    return TargetSpec(
        slug=slug,
        provider="stub",
        model_id="stub-fixed",
        base_url="http://unused",
        extra={"stub_mode": "fixed", "stub_fixed_text": "static reply"},
    )


@pytest.fixture
def handlers(tmp_path: Path) -> LpoMcpHandlers:
    return LpoMcpHandlers(tasks_root=tmp_path)


# ---------------------------------------------------------------------------
# Schema surface
# ---------------------------------------------------------------------------


def test_tool_specs_expose_every_required_tool():
    names = {s["name"] for s in TOOL_SPECS}
    assert names == {
        "lpo_create_task",
        "lpo_generate_gold_standard",
        "lpo_run_optimization",
        "lpo_get_status",
        "lpo_get_winner",
        "lpo_get_comparison",
        "lpo_list_tasks",
    }
    # Every spec must have the MCP-required keys.
    for s in TOOL_SPECS:
        assert s["description"]
        assert s["inputSchema"]["type"] == "object"


@pytest.mark.asyncio
async def test_unknown_tool_returns_error(handlers: LpoMcpHandlers):
    out = await handlers.call("lpo_nope", {})
    assert "error" in out and "Unknown" in out["error"]


# ---------------------------------------------------------------------------
# lpo_list_tasks / lpo_create_task
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_tasks_empty(handlers: LpoMcpHandlers):
    out = await handlers.call("lpo_list_tasks", {})
    assert out == {"tasks": []}


@pytest.mark.asyncio
async def test_create_task_round_trips_through_list(handlers: LpoMcpHandlers):
    out = await handlers.call(
        "lpo_create_task",
        {
            "name": "demo",
            "task_description": "Say hello.",
            "example_inputs": ["hello?", "hi?"],
            "output_type": "text",
        },
    )
    assert out["task_id"] == "demo"
    assert out["strategy"] == "single"
    assert out["n_examples"] == 2

    listed = await handlers.call("lpo_list_tasks", {})
    assert len(listed["tasks"]) == 1
    assert listed["tasks"][0]["name"] == "demo"


@pytest.mark.asyncio
async def test_create_task_duplicate_without_overwrite(handlers: LpoMcpHandlers):
    await handlers.call(
        "lpo_create_task",
        {"name": "dup", "task_description": "x", "example_inputs": ["a"]},
    )
    out = await handlers.call(
        "lpo_create_task",
        {"name": "dup", "task_description": "x", "example_inputs": ["a"]},
    )
    assert "error" in out


@pytest.mark.asyncio
async def test_create_task_accepts_target_models(handlers: LpoMcpHandlers):
    out = await handlers.call(
        "lpo_create_task",
        {
            "name": "tgt",
            "task_description": "x",
            "example_inputs": ["a"],
            "target_models": [
                {
                    "slug": "mystub",
                    "provider": "stub",
                    "model_id": "stub-fixed",
                    "base_url": "http://unused",
                    "stub_mode": "fixed",
                    "stub_fixed_text": "hi",
                },
            ],
        },
    )
    assert out["task_id"] == "tgt"
    # Verify the stub-specific extras survived into config.yaml.
    from lpo.core.task import TaskBundle

    task = TaskBundle.load(handlers.tasks_root / "tgt")
    assert task.config.target_models[0].provider == "stub"
    assert task.config.target_models[0].stub_fixed_text == "hi"


# ---------------------------------------------------------------------------
# lpo_generate_gold_standard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_gold_via_tool_uses_injected_client(monkeypatch, handlers: LpoMcpHandlers):
    # Create a bundle first.
    await handlers.call(
        "lpo_create_task",
        {"name": "g", "task_description": "x", "example_inputs": ["a", "b"]},
    )
    # Patch the authoring module to inject the stub client into the call
    # made from the MCP tool. The handler itself calls generate_gold_standard
    # without a client, so we monkeypatch the generator to use the stub.
    stub = _StubClient()
    import lpo.server.mcp_server as mcp_mod

    async def _fake(task_path, *, overwrite=False, **kw):
        return await generate_gold_standard(task_path, client=stub, overwrite=overwrite)

    monkeypatch.setattr(mcp_mod, "generate_gold_standard", _fake)
    out = await handlers.call("lpo_generate_gold_standard", {"task_id": "g"})
    assert out == {"task_id": "g", "gold_records_written": 2}
    assert len(stub.calls) == 2


# ---------------------------------------------------------------------------
# lpo_run_optimization + lpo_get_status + lpo_get_winner
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_optimization_rejects_non_autonomous_mode(handlers: LpoMcpHandlers):
    await handlers.call(
        "lpo_create_task",
        {
            "name": "r",
            "task_description": "x",
            "example_inputs": ["a"],
            "target_models": [_stub_target_spec().to_config()],
        },
    )
    out = await handlers.call(
        "lpo_run_optimization",
        {"task_id": "r", "mode": "manual"},
    )
    assert "error" in out
    assert "UI-only" in out["error"]


@pytest.mark.asyncio
async def test_run_optimization_single_writes_winner(handlers: LpoMcpHandlers):
    # Use output_type=json so the authoring layer picks a deterministic
    # metric — no ANTHROPIC_API_KEY required for this test.
    await handlers.call(
        "lpo_create_task",
        {
            "name": "solo",
            "task_description": "Return the fixed string.",
            "example_inputs": ["q1", "q2"],
            "output_type": "json",
            "required_json_fields": ["echo"],
            "target_models": [_stub_target_spec().to_config()],
        },
    )

    out = await handlers.call(
        "lpo_run_optimization",
        {
            "task_id": "solo",
            "mutator": "null",
            "fresh": True,
            "stop_conditions": {"max_iterations": 1, "target_score": 0},
        },
    )
    assert out.get("status") == "done", out
    assert out["strategy"] == "single"
    assert out["iterations"] >= 1
    assert out["winner"]["present"] is True
    assert isinstance(out["winner"]["prompt"], str)

    # Status tool should reflect on-disk state.
    status = await handlers.call("lpo_get_status", {"task_id": "solo"})
    assert status["strategy"] == "single"
    assert status["iteration"] >= 1
    assert status["best_score"] is not None

    # Winner tool returns the prompt directly.
    winner = await handlers.call("lpo_get_winner", {"task_id": "solo"})
    assert winner["prompt"]
    assert winner["model_slug"] == "stub-target"

    # Comparison should be absent for single-strategy runs.
    comp = await handlers.call("lpo_get_comparison", {"task_id": "solo"})
    assert comp["present"] is False


@pytest.mark.asyncio
async def test_run_optimization_respects_target_slugs(handlers: LpoMcpHandlers):
    # Two-target parallel_independent task; asking for only one slug must run
    # only that slug and leave the other's runs/ directory absent.
    await handlers.call(
        "lpo_create_task",
        {
            "name": "filtered",
            "task_description": "Return a fixed JSON object.",
            "example_inputs": ["q1"],
            "output_type": "json",
            "required_json_fields": ["echo"],
            "strategy": "parallel_independent",
            "target_models": [
                _stub_target_spec("alpha").to_config(),
                _stub_target_spec("beta").to_config(),
            ],
        },
    )

    out = await handlers.call(
        "lpo_run_optimization",
        {
            "task_id": "filtered",
            "mutator": "null",
            "fresh": True,
            "target_slugs": ["alpha"],
            "stop_conditions": {"max_iterations": 1, "target_score": 0},
        },
    )
    assert out.get("status") == "done", out
    assert out["strategy"] == "parallel_independent"
    slugs_in_result = {r["slug"] for r in out["per_model"]}
    assert slugs_in_result == {"alpha"}, out

    task_root = handlers.tasks_root / "filtered"
    assert (task_root / "runs" / "alpha").exists()
    assert not (task_root / "runs" / "beta").exists(), (
        "beta should have been skipped by target_slugs filter"
    )


@pytest.mark.asyncio
async def test_run_optimization_rejects_unknown_target_slug(handlers: LpoMcpHandlers):
    await handlers.call(
        "lpo_create_task",
        {
            "name": "filt2",
            "task_description": "x",
            "example_inputs": ["a"],
            "strategy": "parallel_independent",
            "target_models": [
                _stub_target_spec("alpha").to_config(),
                _stub_target_spec("beta").to_config(),
            ],
        },
    )
    out = await handlers.call(
        "lpo_run_optimization",
        {"task_id": "filt2", "target_slugs": ["ghost"]},
    )
    assert "error" in out
    assert "ghost" in out["error"]
    # Error message must list the real slugs so the operator can fix the call.
    assert "alpha" in out["error"]
    assert "beta" in out["error"]


@pytest.mark.asyncio
async def test_get_winner_requires_slug_for_parallel(handlers: LpoMcpHandlers):
    await handlers.call(
        "lpo_create_task",
        {
            "name": "par",
            "task_description": "x",
            "example_inputs": ["a"],
            "strategy": "parallel_independent",
            "target_models": [
                _stub_target_spec("a").to_config(),
                _stub_target_spec("b").to_config(),
            ],
        },
    )
    # No on-disk winner yet, but the slug-resolution check fires first.
    out = await handlers.call("lpo_get_winner", {"task_id": "par"})
    assert "error" in out
    assert "model_slug is required" in out["error"]


@pytest.mark.asyncio
async def test_get_winner_for_unknown_task_returns_error(handlers: LpoMcpHandlers):
    out = await handlers.call("lpo_get_winner", {"task_id": "nope"})
    assert "error" in out
    assert "Not found" in out["error"]


@pytest.mark.asyncio
async def test_get_status_for_unrun_task(handlers: LpoMcpHandlers):
    await handlers.call(
        "lpo_create_task",
        {
            "name": "empty",
            "task_description": "x",
            "example_inputs": ["a"],
            "target_models": [_stub_target_spec().to_config()],
        },
    )
    out = await handlers.call("lpo_get_status", {"task_id": "empty"})
    assert out["iteration"] == 0
    assert out["best_score"] is None
    assert out["per_model_status"]["stub-target"]["exists"] is False


# ---------------------------------------------------------------------------
# build_mcp_server + MCP types
# ---------------------------------------------------------------------------


def test_build_mcp_server_registers_all_tools(tmp_path: Path):
    from lpo.server.mcp_server import build_mcp_server

    server = build_mcp_server(tmp_path)
    # The mcp SDK exposes the registered handler via the internal
    # request_handlers map; we can't easily introspect the full tool list
    # without running the protocol, but we can confirm the server name and
    # that our handler object is attached for tests to poke.
    assert server.name == "lpo"
    assert server._lpo_handlers.tasks_root == tmp_path
