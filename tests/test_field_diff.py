from __future__ import annotations

import pytest

from lpo.config.schema import (
    DeterministicRule,
    EvalRecord,
    GoldRecord,
)
from lpo.scoring.base import ScoringContext
from lpo.scoring.deterministic import (
    CheckResult,
    DeterministicScorer,
    check_exact_match_against_gold,
)


def _rec(id_: str = "x") -> EvalRecord:
    return EvalRecord(id=id_, input="whatever")


def test_exact_match_dict_all_fields_match():
    gold = GoldRecord(id="x", output={"a": 1, "b": 2, "c": 3})
    r = check_exact_match_against_gold('{"a":1,"b":2,"c":3}', gold, _rec(), None)
    assert isinstance(r, CheckResult)
    assert r.score == 100.0


def test_exact_match_dict_partial_credit_with_field_diff():
    gold = GoldRecord(id="x", output={"name": "Rustfest", "date": "2025-11-02", "location": "online"})
    out = '{"name": "Rustfest", "date": "2025-11-02", "location": "Online"}'
    r = check_exact_match_against_gold(out, gold, _rec(), None)
    assert isinstance(r, CheckResult)
    # 2/3 match → 66.67
    assert r.score == pytest.approx(200.0 / 3)
    # Detail must surface the specific mismatched field for the Overseer.
    assert "location" in r.detail
    assert "expected" in r.detail and "online" in r.detail
    assert "got" in r.detail and "Online" in r.detail
    # Matched fields also named so the Overseer knows what NOT to touch.
    assert "name" in r.detail
    assert "date" in r.detail


def test_exact_match_reports_missing_keys():
    gold = GoldRecord(id="x", output={"a": 1, "b": 2})
    r = check_exact_match_against_gold('{"a": 1}', gold, _rec(), None)
    assert r.score == pytest.approx(50.0)
    assert "missing" in r.detail.lower() and "b" in r.detail


def test_exact_match_list_partial_credit():
    gold = GoldRecord(id="x", output=[1, 2, 3])
    r = check_exact_match_against_gold("[1, 2, 5]", gold, _rec(), None)
    assert r.score == pytest.approx(200.0 / 3)
    assert "2/3" in r.detail


def test_exact_match_string_fallback_mismatch_shows_diff():
    gold = GoldRecord(id="x", output="hello world")
    r = check_exact_match_against_gold("howdy world", gold, _rec(), None)
    assert r.score == 0.0
    assert "expected" in r.detail and "got" in r.detail


@pytest.mark.asyncio
async def test_deterministic_scorer_rationale_includes_field_diff():
    """The rationale surfaced to the Overseer must show which field failed."""
    rules = [
        DeterministicRule(name="exact", weight=100, check="exact_match_against_gold"),
    ]
    scorer = DeterministicScorer(rules)
    gold = GoldRecord(
        id="x",
        output={"name": "Rustfest", "date": "2025-11-02", "location": "online"},
    )
    out = '{"name": "Rustfest", "date": "2025-11-02", "location": "Online"}'
    r = await scorer.score(out, gold, _rec(), ScoringContext("t", 1))
    assert "location" in r.rationale
    assert "online" in r.rationale
    assert "Online" in r.rationale
