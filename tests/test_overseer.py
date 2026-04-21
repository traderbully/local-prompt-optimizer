from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pytest

from lpo.core.cost import CostTracker
from lpo.core.engine import RatchetEngine, StopReason
from lpo.core.history import IterationRecord
from lpo.core.task import TaskBundle
from lpo.models.anthropic_client import AnthropicMessage, AnthropicResult
from lpo.overseer.agent import OverseerMutator
from lpo.overseer.context import (
    ConversationContext,
    IterationTurn,
    estimate_tokens,
    format_iteration_turn,
)
from lpo.overseer.prompt_writer import parse_overseer_response
from lpo.scoring.deterministic import build_scorer
from tests.conftest import StubClient

EXAMPLE = Path(__file__).resolve().parents[1] / "tasks" / "example_json_extract"


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def test_parser_accepts_well_formed_response():
    text = """
    <analysis>The model produced prose wrapping, not strict JSON.</analysis>
    <hypothesis>Adding an explicit 'JSON only' constraint should lift json_valid to 100.</hypothesis>
    <prompt>
    Return ONLY a JSON object with keys: name, date, location.
    </prompt>
    """
    parsed = parse_overseer_response(text)
    assert parsed is not None
    assert "JSON only" in parsed.hypothesis
    assert "ONLY a JSON object" in parsed.new_prompt
    assert parsed.analysis.startswith("The model")


def test_parser_returns_none_when_prompt_missing():
    text = "<analysis>x</analysis><hypothesis>y</hypothesis>"
    assert parse_overseer_response(text) is None


def test_parser_tolerates_tag_casing_and_whitespace():
    text = "  <ANALYSIS>a</ANALYSIS>\n<Prompt>\n  hi\n</Prompt>  "
    parsed = parse_overseer_response(text)
    assert parsed is not None
    assert parsed.new_prompt == "hi"


# ---------------------------------------------------------------------------
# Turn formatting (SDP §5.3)
# ---------------------------------------------------------------------------


def test_iteration_turn_format_matches_sdp_shape():
    rec = IterationRecord(
        index=7,
        prompt="SYS PROMPT BODY",
        aggregate_score=87.3,
        per_example={"ex_003": 40.0, "ex_011": 55.0},
        per_scenario={"casual_tone": 92.0, "formal_tone": 85.0, "edge_case": 71.0},
        failed_ids=["ex_003", "ex_011"],
        outputs=[
            {"id": "ex_003", "input": "in-3", "output": "out-3", "score": 40.0, "rationale": "bad fmt"},
            {"id": "ex_011", "input": "in-11", "output": "out-11", "score": 55.0, "rationale": "bad tone"},
        ],
        decision="accepted",
        delta=2.1,
    )
    turn = IterationTurn.from_record(rec)
    body = format_iteration_turn(turn)
    # SDP §5.3 anchors
    assert "ITERATION 7" in body
    assert "Aggregate score: 87.30 (delta +2.10)" in body
    assert "Scenario breakdown" in body
    assert "casual_tone=92.0" in body
    assert "Failed example ids: ex_003, ex_011" in body
    assert "Prompt used:" in body
    assert "```" in body
    # Failed example details included
    assert "id=ex_003" in body


# ---------------------------------------------------------------------------
# Context budgeting
# ---------------------------------------------------------------------------


def test_context_does_not_summarize_below_budget():
    ctx = ConversationContext(max_tokens=1_000, keep_recent_pairs=2)
    ctx.add_user("hi" * 10)
    ctx.add_assistant("ok" * 10)
    assert not ctx.needs_summarization("new user msg")


def test_context_summarizes_when_over_budget():
    ctx = ConversationContext(max_tokens=50, keep_recent_pairs=1)
    # 3 user/assistant pairs, then ask: do we need summarization?
    for i in range(3):
        ctx.add_user("u" * 200)
        ctx.add_assistant("a" * 200)
    assert ctx.needs_summarization("pending user msg " * 10)
    ctx.apply_summary("short summary")
    # After folding, only the most recent pair should remain verbatim.
    assert len(ctx.turns) == 2
    assert ctx.summary == "short summary"


def test_estimate_tokens_monotonic():
    assert estimate_tokens("short") < estimate_tokens("a much longer body of text " * 5)


# ---------------------------------------------------------------------------
# Cost tracker
# ---------------------------------------------------------------------------


def test_cost_tracker_longest_prefix_and_total():
    t = CostTracker()
    # Specific-snapshot id should resolve to the longer "claude-opus-4-5"
    # prefix ($5 input / $25 output), not the shorter "claude-opus-4" family
    # prefix ($15 / $75). This is exactly why the rate table keeps both.
    call = t.record("claude-opus-4-5-20250101", prompt_tokens=1_000_000, completion_tokens=0)
    assert call.usd == pytest.approx(5.0)
    t.record("claude-opus-4-5-20250101", prompt_tokens=0, completion_tokens=1_000_000)
    # Add $25 output rate → total $30.
    assert t.total_usd == pytest.approx(30.0)
    assert t.over_cap(29.0)
    assert not t.over_cap(100.0)


def test_cost_tracker_opus_4_family_prefix_still_15_75():
    """Opus 4 / 4.1 stayed on the legacy $15/$75 tier. The shorter
    'claude-opus-4' family prefix must still resolve those snapshots even
    after the 4.5/4.6/4.7 rate split."""
    t = CostTracker()
    call = t.record("claude-opus-4-1-20250805", prompt_tokens=1_000_000, completion_tokens=1_000_000)
    assert call.usd == pytest.approx(90.0)  # 15 + 75


def test_cost_tracker_haiku_4_5_uses_new_rate():
    """Haiku 4.5 is $1/$5, distinct from the legacy $0.80/$4 of 3.5."""
    t = CostTracker()
    call = t.record("claude-haiku-4-5-20251001", prompt_tokens=1_000_000, completion_tokens=1_000_000)
    assert call.usd == pytest.approx(6.0)  # 1 + 5


def test_cost_tracker_unknown_model_is_free():
    t = CostTracker()
    call = t.record("some-local-model", 10_000, 10_000)
    assert call.usd == 0.0


# ---------------------------------------------------------------------------
# OverseerMutator — with stubbed Anthropic client
# ---------------------------------------------------------------------------


class FakeAnthropic:
    """Stand-in for AnthropicClient. Responds via a scripted callable."""

    def __init__(
        self,
        responder: Callable[[str, list[AnthropicMessage]], str],
        cost_tracker: CostTracker | None = None,
    ) -> None:
        self.responder = responder
        self.cost = cost_tracker or CostTracker()
        self.model_id = "claude-opus-4-fake"
        self.calls: list[dict] = []

    async def aclose(self) -> None:
        return None

    async def complete(
        self,
        *,
        system: str,
        messages: list[AnthropicMessage],
        temperature: float = 0.3,
        max_tokens: int | None = None,
    ) -> AnthropicResult:
        text = self.responder(system, messages)
        self.calls.append({"system": system, "messages": messages, "text": text})
        # Pretend each call cost a trivial amount so cost tracker accumulates.
        self.cost.record(self.model_id, prompt_tokens=100, completion_tokens=50)
        return AnthropicResult(
            text=text,
            prompt_tokens=100,
            completion_tokens=50,
            model_id=self.model_id,
        )


def _copy_example(tmp_path: Path) -> Path:
    dst = tmp_path / "task"
    shutil.copytree(EXAMPLE, dst)
    # Strip any prior run artifacts so the engine doesn't resume from them.
    for sub in ("runs", "logs"):
        p = dst / sub
        if p.exists():
            shutil.rmtree(p)
    return dst


def _make_target_responder_for(current_prompt_match: Callable[[str], str]):
    """Returns a StubClient responder that returns gold JSON only when the
    current prompt matches ``current_prompt_match``, otherwise an empty string.
    """

    gold_by_id = {
        json.loads(line)["id"]: json.loads(line)["output"]
        for line in (EXAMPLE / "gold_standard.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    input_to_id = {
        json.loads(line)["input"]: json.loads(line)["id"]
        for line in (EXAMPLE / "eval_set.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }

    def responder(system: str, user: str, _seed):
        gold = gold_by_id[input_to_id[user]]
        # Decide output quality based on system prompt.
        return current_prompt_match(system) or json.dumps(gold)

    return responder


@pytest.mark.asyncio
async def test_overseer_proposes_and_engine_accepts(tmp_path, monkeypatch):
    task_root = _copy_example(tmp_path)
    # Force config: allow a few iterations, low plateau patience.
    cfg = (task_root / "config.yaml").read_text().replace("max_iterations: 10", "max_iterations: 4")
    (task_root / "config.yaml").write_text(cfg)

    task = TaskBundle.load(task_root)
    target = task.config.target_models[0]
    scorer = build_scorer(task.metric)

    # Target: returns gibberish for the seed prompt, returns gold for the new prompt.
    SEED = task.seed_prompt.strip()
    NEW = "RETURN STRICT JSON: {\"name\":..., \"date\":..., \"location\":...}"

    def match(system: str) -> str:
        # If the system is the seed, return non-JSON to drive low score.
        if system.strip() == SEED:
            return "not json at all"
        return ""  # signal "use gold"

    target_client = StubClient(responder=_make_target_responder_for(match))

    # Overseer: always proposes the NEW prompt.
    overseer_response = (
        "<analysis>Outputs were not JSON.</analysis>"
        "<hypothesis>Force strict JSON.</hypothesis>"
        f"<prompt>{NEW}</prompt>"
    )
    cost = CostTracker()
    fake = FakeAnthropic(lambda s, m: overseer_response, cost_tracker=cost)
    mutator = OverseerMutator(client=fake, task=task)

    engine = RatchetEngine(
        task=task,
        target_cfg=target,
        client=target_client,
        scorer=scorer,
        mutator=mutator,
        cost_tracker=cost,
    )
    result = await engine.run()

    # Iter 1 = seed (bad), iter 2 = NEW prompt (perfect) → should hit target score.
    assert result.stop_reason == StopReason.TARGET_SCORE
    assert result.best_score == pytest.approx(100.0)
    # Seed iter was accepted as "initial" despite low score (nothing to compare to).
    assert result.iterations[0].decision == "initial"
    assert result.iterations[1].decision == "accepted"
    assert result.iterations[1].aggregate_score > result.iterations[0].aggregate_score
    # Overseer analysis was persisted next to iter 1.
    analysis_md = engine.paths.iteration_dir(1) / "overseer_analysis.md"
    assert analysis_md.exists()
    assert "Force strict JSON" in analysis_md.read_text(encoding="utf-8")
    # Cost tracker recorded at least one call.
    assert result.total_cost_usd > 0.0


@pytest.mark.asyncio
async def test_overseer_clarifies_on_malformed_response(tmp_path):
    task_root = _copy_example(tmp_path)
    task = TaskBundle.load(task_root)
    target = task.config.target_models[0]
    scorer = build_scorer(task.metric)
    SEED = task.seed_prompt.strip()

    # Target: always returns gibberish so every iteration is failure.
    target_client = StubClient(responder=lambda s, u, sd: "nope")

    call_count = {"n": 0}

    def responder(system: str, messages: list[AnthropicMessage]) -> str:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return "<analysis>broken reply, no prompt tag</analysis>"
        return (
            "<analysis>ok</analysis>"
            "<hypothesis>h</hypothesis>"
            "<prompt>FIXED PROMPT</prompt>"
        )

    cost = CostTracker()
    fake = FakeAnthropic(responder, cost_tracker=cost)
    mutator = OverseerMutator(client=fake, task=task)

    # Only run one mutation by capping iterations to 2.
    cfg_path = task_root / "config.yaml"
    cfg_path.write_text(
        cfg_path.read_text()
        .replace("max_iterations: 10", "max_iterations: 2")
        .replace("target_score: 95", "target_score: 999")
        .replace("plateau_patience: 3", "plateau_patience: 99")
    )
    task = TaskBundle.load(task_root)
    target = task.config.target_models[0]

    engine = RatchetEngine(
        task=task,
        target_cfg=target,
        client=target_client,
        scorer=scorer,
        mutator=mutator,
        cost_tracker=cost,
    )
    await engine.run()

    # Both overseer calls were issued: first was malformed, retry succeeded.
    assert call_count["n"] == 2
    # The fixed prompt was used for iter 2 (persisted in its history folder).
    iter2_prompt = engine.paths.iteration_dir(2) / "prompt.txt"
    assert iter2_prompt.read_text(encoding="utf-8").strip() == "FIXED PROMPT"


@pytest.mark.asyncio
async def test_cost_cap_stops_engine(tmp_path):
    task_root = _copy_example(tmp_path)
    cfg_path = task_root / "config.yaml"
    cfg_path.write_text(
        cfg_path.read_text()
        .replace("max_iterations: 10", "max_iterations: 5")
        .replace("cost_cap_usd: 2.00", "cost_cap_usd: 0.0001")
    )
    task = TaskBundle.load(task_root)
    target = task.config.target_models[0]
    scorer = build_scorer(task.metric)

    target_client = StubClient(responder=lambda s, u, sd: "nope")

    # Overseer that always proposes something new (so loop would otherwise keep going).
    counter = {"n": 0}

    def responder(system: str, messages: list[AnthropicMessage]) -> str:
        counter["n"] += 1
        return (
            "<analysis>a</analysis><hypothesis>h</hypothesis>"
            f"<prompt>iteration-{counter['n']}</prompt>"
        )

    cost = CostTracker()
    fake = FakeAnthropic(responder, cost_tracker=cost)
    mutator = OverseerMutator(client=fake, task=task)

    engine = RatchetEngine(
        task=task,
        target_cfg=target,
        client=target_client,
        scorer=scorer,
        mutator=mutator,
        cost_tracker=cost,
    )
    result = await engine.run()
    assert result.stop_reason == StopReason.COST_CAP
    assert result.total_cost_usd >= 0.0001
