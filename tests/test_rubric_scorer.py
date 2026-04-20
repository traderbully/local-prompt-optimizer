from __future__ import annotations

from typing import Callable

import pytest

from lpo.config.schema import EvalRecord, GoldRecord, RubricCriterion, RubricMetric
from lpo.core.cost import CostTracker
from lpo.models.anthropic_client import AnthropicMessage, AnthropicResult
from lpo.scoring.base import ScoringContext
from lpo.scoring.factory import build_scorer
from lpo.scoring.rubric import RubricScorer, _parse_judge_response


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
        self.calls.append({"system": system, "messages": messages, "text": text})
        self.cost.record(self.model_id, 200, 100)
        return AnthropicResult(text=text, prompt_tokens=200, completion_tokens=100, model_id=self.model_id)


def _metric() -> RubricMetric:
    return RubricMetric(
        type="rubric",
        judge_model="claude-opus-4-judge",
        criteria=[
            RubricCriterion(name="accuracy", weight=50, description="Is it right?"),
            RubricCriterion(name="style", weight=50, description="Does it sound nice?",
                            anchors={0: "bad", 50: "ok", 100: "great"}),
        ],
    )


def test_parser_accepts_plain_json():
    text = '{"scores": {"accuracy": {"score": 85, "rationale": "mostly right"}, "style": {"score": 90, "rationale": "punchy"}}}'
    parsed = _parse_judge_response(text, ["accuracy", "style"])
    assert parsed["accuracy"] == (85.0, "mostly right")
    assert parsed["style"] == (90.0, "punchy")


def test_parser_accepts_bare_number():
    text = '{"scores": {"accuracy": 70, "style": 60}}'
    parsed = _parse_judge_response(text, ["accuracy", "style"])
    assert parsed["accuracy"] == (70.0, "")


def test_parser_accepts_prose_wrapping():
    text = 'Sure! Here are the scores:\n```json\n{"scores": {"accuracy": {"score": 80}, "style": {"score": 55}}}\n```\n'
    parsed = _parse_judge_response(text, ["accuracy", "style"])
    assert parsed["accuracy"][0] == 80.0


def test_parser_reports_missing_criterion():
    text = '{"scores": {"accuracy": {"score": 80}}}'
    parsed = _parse_judge_response(text, ["accuracy", "style"])
    assert parsed["style"][0] == 0.0
    assert "did not return" in parsed["style"][1]


def test_parser_raises_on_garbage():
    import pytest as _pt

    with _pt.raises(ValueError):
        _parse_judge_response("no json here at all", ["a"])


@pytest.mark.asyncio
async def test_rubric_scorer_weighted_aggregate_and_rationale():
    def responder(system: str, messages: list[AnthropicMessage]) -> str:
        return (
            '{"scores": {"accuracy": {"score": 90, "rationale": "correct"}, '
            '"style": {"score": 60, "rationale": "a bit stiff"}}}'
        )

    judge = FakeJudge(responder)
    scorer = RubricScorer(_metric(), judge)
    r = await scorer.score(
        "some output",
        GoldRecord(id="x", output="gold"),
        EvalRecord(id="x", input="some input"),
        ScoringContext("t", 1),
    )
    # 90*50 + 60*50 = 7500 / 100 = 75
    assert r.aggregate == pytest.approx(75.0)
    assert r.per_criterion == {"accuracy": 90.0, "style": 60.0}
    assert "correct" in r.rationale and "a bit stiff" in r.rationale
    # Judge call happened exactly once, with the rubric and the gold in the user msg.
    assert len(judge.calls) == 1
    user = judge.calls[0]["messages"][-1].content
    assert "accuracy" in user and "style" in user and "some output" in user


@pytest.mark.asyncio
async def test_rubric_scorer_handles_malformed_judge_gracefully():
    judge = FakeJudge(lambda s, m: "I can't score this, sorry.")
    scorer = RubricScorer(_metric(), judge)
    r = await scorer.score(
        "o", None, EvalRecord(id="x", input="i"), ScoringContext("t", 1)
    )
    assert r.aggregate == 0.0
    assert "unparseable" in r.rationale


def test_factory_rejects_rubric_without_judge_factory():
    with pytest.raises(ValueError, match="judge_factory"):
        build_scorer(_metric())


def test_factory_builds_rubric_scorer_with_factory():
    judge = FakeJudge(lambda s, m: '{"scores":{}}')
    scorer = build_scorer(_metric(), judge_factory=lambda _id: judge)
    assert isinstance(scorer, RubricScorer)
    assert scorer.judge is judge
