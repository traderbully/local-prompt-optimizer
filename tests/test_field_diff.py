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
    check_numeric_range,
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


# ---------------------------------------------------------------------------
# PowerShell verify_command semantic equivalence (Stage-8 Analyst I3).
# `-not (Test-Path 'X')` and `Test-Path 'X'` are semantically equivalent
# verifications of successful deletion (inverted boolean conventions).
# The normalizer must make them compare equal, symmetrically.
# ---------------------------------------------------------------------------


def test_ps_not_testpath_equals_bare_testpath():
    """Model emits `-not (Test-Path X)`; gold uses bare `Test-Path X`."""
    gold = GoldRecord(
        id="x",
        output={
            "command": "Remove-Item 'C:\\temp\\f.txt'",
            "verify_command": "Test-Path 'C:\\temp\\f.txt'",
        },
    )
    out = (
        '{"command": "Remove-Item \'C:\\\\temp\\\\f.txt\'", '
        '"verify_command": "-not (Test-Path \'C:\\\\temp\\\\f.txt\')"}'
    )
    r = check_exact_match_against_gold(out, gold, _rec(), None)
    assert r.score == 100.0, r.detail


def test_ps_bare_testpath_equals_not_wrapped_in_gold():
    """Symmetric case: gold uses `-not (Test-Path X)`; model emits bare."""
    gold = GoldRecord(
        id="x",
        output={"verify_command": "-not (Test-Path 'C:\\temp\\f.txt')"},
    )
    r = check_exact_match_against_gold(
        '{"verify_command": "Test-Path \'C:\\\\temp\\\\f.txt\'"}',
        gold, _rec(), None,
    )
    assert r.score == 100.0, r.detail


def test_ps_normalization_does_not_cross_paths():
    """Normalization must only collapse `-not (Test-Path X)` to `Test-Path X`.
    Paths that differ must still mismatch."""
    gold = GoldRecord(id="x", output={"verify_command": "Test-Path 'C:\\a'"})
    r = check_exact_match_against_gold(
        '{"verify_command": "-not (Test-Path \'C:\\\\b\')"}',
        gold, _rec(), None,
    )
    assert r.score == 0.0


# ---------------------------------------------------------------------------
# exact_match_against_gold `fields` subset filter (Stage-8 F1 on
# jarvis_routing_opus, 2026-04-22). When `params={"fields": [...]}` is
# supplied only those keys participate in exact-match; other gold keys are
# expected to be scored by a different rule.
# ---------------------------------------------------------------------------


def test_fields_filter_scores_subset_only():
    gold = GoldRecord(id="x", output={"skill_id": "s1", "action": "a1", "confidence": 0.98})
    out = '{"skill_id": "s1", "action": "a1", "confidence": 0.42}'
    # Without the filter, confidence mismatch drops the score to 2/3.
    r_full = check_exact_match_against_gold(out, gold, _rec(), None)
    assert r_full.score == pytest.approx(200.0 / 3)
    # With a filter that excludes confidence, we should get a perfect score.
    r_filtered = check_exact_match_against_gold(
        out, gold, _rec(), {"fields": ["skill_id", "action"]}
    )
    assert r_filtered.score == 100.0


def test_fields_filter_empty_scope_does_not_divide_by_zero():
    gold = GoldRecord(id="x", output={"a": 1})
    r = check_exact_match_against_gold(
        '{"a": 2}', gold, _rec(), {"fields": ["nonexistent"]}
    )
    assert r.score == 100.0  # no fields in scope → trivially satisfied


def test_fields_filter_rejects_malformed_params():
    gold = GoldRecord(id="x", output={"a": 1})
    with pytest.raises(ValueError):
        check_exact_match_against_gold('{"a": 1}', gold, _rec(), {"fields": "a,b"})


# ---------------------------------------------------------------------------
# numeric_range check (Stage-8 F1). Score 100 if `output[field]` is a
# number in [min, max]; else 0. Used to replace exact-match on
# subjective/self-estimated numeric fields like confidence.
# ---------------------------------------------------------------------------


def test_numeric_range_inside_band_scores_100():
    r = check_numeric_range('{"confidence": 0.85}', None, _rec(), {"field": "confidence", "min": 0.5, "max": 1.0})
    assert r.score == 100.0


def test_numeric_range_boundary_inclusive():
    params = {"field": "x", "min": 0.5, "max": 1.0}
    assert check_numeric_range('{"x": 0.5}', None, _rec(), params).score == 100.0
    assert check_numeric_range('{"x": 1.0}', None, _rec(), params).score == 100.0


def test_numeric_range_outside_band_scores_0():
    params = {"field": "confidence", "min": 0.5, "max": 1.0}
    assert check_numeric_range('{"confidence": 0.3}', None, _rec(), params).score == 0.0
    assert check_numeric_range('{"confidence": 1.2}', None, _rec(), params).score == 0.0


def test_numeric_range_rejects_bool_masquerading_as_number():
    # Regression: bool is a subclass of int in Python; we must not treat
    # `true`/`false` as 1/0 here.
    r = check_numeric_range(
        '{"confidence": true}', None, _rec(), {"field": "confidence", "min": 0.5, "max": 1.0}
    )
    assert r.score == 0.0


def test_numeric_range_missing_field_scores_0():
    r = check_numeric_range('{"other": 0.9}', None, _rec(), {"field": "confidence"})
    assert r.score == 0.0
    assert "missing" in r.detail


def test_numeric_range_unparseable_output_scores_0():
    r = check_numeric_range("not json at all", None, _rec(), {"field": "x"})
    assert r.score == 0.0


def test_ps_normalization_ignores_unrelated_strings():
    """A field that *contains* the words but isn't the whole-value pattern
    must not be normalized. Regression against over-eager matching."""
    gold = GoldRecord(id="x", output={"user_message": "Will run Test-Path to verify."})
    r = check_exact_match_against_gold(
        '{"user_message": "Will not run -not (Test-Path ...) here."}',
        gold, _rec(), None,
    )
    assert r.score == 0.0


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
