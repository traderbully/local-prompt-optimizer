from __future__ import annotations

import json
from typing import Callable

import pytest

from lpo.config.schema import ConversationalMetric, EvalRecord
from lpo.core.cost import CostTracker
from lpo.models.anthropic_client import AnthropicMessage, AnthropicResult
from lpo.overseer.context import ConversationContext
from lpo.scoring.base import ScoringContext
from lpo.scoring.conversational import ConversationalScorer, _parse
from lpo.scoring.factory import build_scorer


class FakeJudge:
    def __init__(self, responder: Callable[[str, list[AnthropicMessage]], str]) -> None:
        self.responder = responder
        self.model_id = "claude-opus-4-judge"
        self.cost = CostTracker()
        self.calls: list[dict] = []

    async def aclose(self) -> None:
        return None

    async def complete(
        self,
        *,
        system: str,
        messages: list[AnthropicMessage],
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> AnthropicResult:
        text = self.responder(system, messages)
        self.calls.append({"system": system, "messages": list(messages), "text": text})
        self.cost.record(self.model_id, 300, 100)
        return AnthropicResult(text=text, prompt_tokens=300, completion_tokens=100, model_id=self.model_id)


def _metric() -> ConversationalMetric:
    return ConversationalMetric(
        type="conversational",
        overseer_model="claude-opus-4-judge",
        stated_goal="Outputs should feel warm and human, not like a robot wrote them.",
    )


def _records() -> list[EvalRecord]:
    return [
        EvalRecord(id="a", input="Hello", scenario="greeting"),
        EvalRecord(id="b", input="Goodbye", scenario="greeting"),
    ]


@pytest.mark.asyncio
async def test_scores_all_examples_in_one_call():
    def responder(system: str, messages: list[AnthropicMessage]) -> str:
        # Verify both ids and the stated goal made it into the prompts.
        assert "warm and human" in system
        user = messages[-1].content
        assert "id=a" in user and "id=b" in user
        return json.dumps({
            "scores": {
                "a": {"score": 80, "rationale": "warm enough"},
                "b": {"score": 40, "rationale": "robotic"},
            }
        })

    judge = FakeJudge(responder)
    scorer = ConversationalScorer(_metric(), judge)
    results = await scorer.score_iteration(
        outputs={"a": "Hey there!", "b": "Farewell, human."},
        eval_records=_records(),
        gold_standard={},
        context=ScoringContext("t", 1),
    )
    assert results["a"].aggregate == 80.0
    assert results["b"].aggregate == 40.0
    assert "warm" in results["a"].rationale
    assert len(judge.calls) == 1


@pytest.mark.asyncio
async def test_context_accumulates_across_iterations():
    responses = iter([
        json.dumps({"scores": {"a": {"score": 60, "rationale": "r1"}, "b": {"score": 55, "rationale": "r2"}}}),
        json.dumps({"scores": {"a": {"score": 85, "rationale": "r3"}, "b": {"score": 80, "rationale": "r4"}}}),
    ])
    judge = FakeJudge(lambda s, m: next(responses))
    scorer = ConversationalScorer(_metric(), judge)

    await scorer.score_iteration(
        outputs={"a": "v1a", "b": "v1b"},
        eval_records=_records(),
        gold_standard={},
        context=ScoringContext("t", 1),
    )
    # Iteration 2 should see iteration 1's exchange in the messages list.
    await scorer.score_iteration(
        outputs={"a": "v2a", "b": "v2b"},
        eval_records=_records(),
        gold_standard={},
        context=ScoringContext("t", 2),
    )
    # Second judge call received iteration 1's user+assistant pair before its own user message.
    second_messages = judge.calls[1]["messages"]
    assert len(second_messages) == 3  # prior user, prior assistant, new user
    assert "ITERATION 1" in second_messages[0].content
    assert "ITERATION 2" in second_messages[2].content


@pytest.mark.asyncio
async def test_retries_on_malformed_response():
    attempts = {"n": 0}

    def responder(system: str, messages: list[AnthropicMessage]) -> str:
        attempts["n"] += 1
        if attempts["n"] == 1:
            return "sorry, I can't produce JSON right now"
        return json.dumps({
            "scores": {
                "a": {"score": 70, "rationale": "ok"},
                "b": {"score": 65, "rationale": "ok"},
            }
        })

    judge = FakeJudge(responder)
    scorer = ConversationalScorer(_metric(), judge)
    results = await scorer.score_iteration(
        outputs={"a": "x", "b": "y"},
        eval_records=_records(),
        gold_standard={},
        context=ScoringContext("t", 1),
    )
    assert attempts["n"] == 2
    assert results["a"].aggregate == 70.0


@pytest.mark.asyncio
async def test_returns_zeros_when_judge_never_produces_valid_json():
    judge = FakeJudge(lambda s, m: "nope, just prose forever")
    scorer = ConversationalScorer(_metric(), judge, max_retries=1)
    results = await scorer.score_iteration(
        outputs={"a": "x", "b": "y"},
        eval_records=_records(),
        gold_standard={},
        context=ScoringContext("t", 1),
    )
    assert results["a"].aggregate == 0.0
    assert "unparseable" in results["a"].rationale


def test_parse_accepts_code_fenced_json():
    text = "```json\n{\"scores\": {\"a\": {\"score\": 50}}}\n```"
    p = _parse(text)
    assert p["a"]["score"] == 50


def test_factory_rejects_conversational_without_judge_factory():
    with pytest.raises(ValueError, match="judge_factory"):
        build_scorer(_metric())


def test_factory_builds_conversational_scorer():
    judge = FakeJudge(lambda s, m: '{"scores":{}}')
    scorer = build_scorer(_metric(), judge_factory=lambda _id: judge)
    assert isinstance(scorer, ConversationalScorer)


@pytest.mark.asyncio
async def test_score_single_example_raises_wrong_method():
    scorer = ConversationalScorer(_metric(), FakeJudge(lambda s, m: ""))
    with pytest.raises(RuntimeError, match="score_iteration"):
        await scorer.score(
            "x", None, EvalRecord(id="a", input="i"), ScoringContext("t", 1)
        )
