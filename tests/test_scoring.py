from __future__ import annotations

import pytest

from lpo.config.schema import (
    DeterministicMetric,
    DeterministicRule,
    EvalRecord,
    GoldRecord,
)
from lpo.scoring.aggregation import aggregate_scores
from lpo.scoring.base import ScoringContext
from lpo.scoring.deterministic import DeterministicScorer, _extract_json, build_scorer


def _rule(name: str, weight: float, check: str, params=None) -> DeterministicRule:
    return DeterministicRule(name=name, weight=weight, check=check, params=params)


@pytest.mark.asyncio
async def test_deterministic_scorer_full_credit():
    rules = [
        _rule("json_valid", 30, "is_valid_json"),
        _rule("has_keys", 40, "has_keys", ["a", "b"]),
        _rule("exact", 30, "exact_match_against_gold"),
    ]
    scorer = DeterministicScorer(rules)
    gold = GoldRecord(id="x", output={"a": 1, "b": 2})
    rec = EvalRecord(id="x", input="whatever")
    out = '{"a": 1, "b": 2}'
    r = await scorer.score(out, gold, rec, ScoringContext("t", 1))
    assert r.aggregate == pytest.approx(100.0)
    assert r.per_criterion == {"json_valid": 100.0, "has_keys": 100.0, "exact": 100.0}


@pytest.mark.asyncio
async def test_deterministic_scorer_partial_credit():
    rules = [
        _rule("json_valid", 50, "is_valid_json"),
        _rule("has_keys", 50, "has_keys", ["a", "b", "c"]),
    ]
    scorer = DeterministicScorer(rules)
    rec = EvalRecord(id="x", input="whatever")
    # Has 'a' and 'b' but not 'c'
    out = '{"a": 1, "b": 2}'
    r = await scorer.score(out, None, rec, ScoringContext("t", 1))
    assert r.per_criterion["json_valid"] == 100.0
    assert r.per_criterion["has_keys"] == pytest.approx(200.0 / 3)
    assert r.aggregate == pytest.approx((100.0 + 200.0 / 3) / 2)


@pytest.mark.asyncio
async def test_deterministic_scorer_invalid_json_zero():
    rules = [
        _rule("json_valid", 50, "is_valid_json"),
        _rule("has_keys", 50, "has_keys", ["a"]),
    ]
    scorer = DeterministicScorer(rules)
    rec = EvalRecord(id="x", input="whatever")
    r = await scorer.score("not json at all", None, rec, ScoringContext("t", 1))
    assert r.aggregate == 0.0


def test_extract_json_tolerates_fences_and_prose():
    assert _extract_json("```json\n{\"a\": 1}\n```") == {"a": 1}
    assert _extract_json("sure, here it is: {\"a\": 1} ok?") == {"a": 1}
    with pytest.raises(ValueError):
        _extract_json("nothing here")


def test_build_scorer_rejects_non_deterministic():
    from lpo.config.schema import ConversationalMetric

    metric = ConversationalMetric(
        type="conversational", overseer_model="claude", stated_goal="be good"
    )
    with pytest.raises(NotImplementedError):
        build_scorer(metric)


@pytest.mark.asyncio
async def test_aggregate_weights_and_scenarios():
    rules = [_rule("json_valid", 100, "is_valid_json")]
    scorer = DeterministicScorer(rules)
    recs = [
        EvalRecord(id="a", input="x", scenario="s1", weight=1.0),
        EvalRecord(id="b", input="x", scenario="s1", weight=3.0),
        EvalRecord(id="c", input="x", scenario="s2", weight=1.0),
    ]
    outputs = {"a": "{}", "b": "not json", "c": "{}"}
    ctx = ScoringContext("t", 1)
    results = {
        r.id: await scorer.score(outputs[r.id], None, r, ctx) for r in recs
    }
    agg = aggregate_scores(recs, results)
    # a=100, b=0 (weight 3), c=100  => (100*1 + 0*3 + 100*1) / 5 = 40
    assert agg.aggregate == pytest.approx(40.0)
    assert agg.per_scenario["s1"] == pytest.approx(25.0)  # (100*1 + 0*3)/4
    assert agg.per_scenario["s2"] == pytest.approx(100.0)
    assert "b" in agg.failed_ids
