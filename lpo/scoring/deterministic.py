"""Code-scored metrics (Type A). See `LPO_SDP.md` §4.4.

Checks now return either a bare ``float`` (legacy) or a :class:`CheckResult`
carrying a short explanation. The explanation is surfaced to the Overseer via
the per-example rationale — this is what closes the "exact_match gives no
per-field feedback" gap identified in the Stage 2 smoke test.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Union

from lpo.config.schema import (
    DeterministicMetric,
    DeterministicRule,
    EvalRecord,
    GoldRecord,
    MetricConfig,
)
from lpo.scoring.base import ScoreResult, Scorer, ScoringContext


@dataclass
class CheckResult:
    score: float
    detail: str = ""


CheckReturn = Union[float, CheckResult]
CheckFn = Callable[[str, "GoldRecord | None", "EvalRecord", Any], CheckReturn]


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def _extract_json(text: str) -> Any:
    """Tolerant JSON extraction.

    Strategy:
      1. strip ```json fences if present
      2. try direct json.loads
      3. fall back to first {...} or [...] span

    Raises ``ValueError`` if nothing parseable is found.
    """
    s = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", s, re.DOTALL | re.IGNORECASE)
    if fence:
        s = fence.group(1).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start = s.find(opener)
        end = s.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(s[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("no parseable JSON found")


def _short(value: Any, max_len: int = 60) -> str:
    s = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
    s = s.replace("\n", " ")
    if len(s) > max_len:
        s = s[: max_len - 1] + "…"
    return s


# PowerShell verify-command semantic-equivalence normalizer. Motivated by
# Stage-8 Analyst finding I3 on windows_file_ops_reliability: deletion
# scenarios have two equally-valid verify forms —
#   (a) `Test-Path 'X'`          (returns False on successful deletion)
#   (b) `-not (Test-Path 'X')`   (returns True on successful deletion)
# Both verify the same thing with inverted boolean semantics. Without
# this normalizer the exact-string metric penalised whichever form the
# gold didn't happen to pick. The regex is intentionally narrow: it only
# matches when the entire value is `-not ( Test-Path ... )`, so non-
# PowerShell fields are unaffected. Applied symmetrically to both output
# and gold in :func:`check_exact_match_against_gold`.
_PS_NOT_TESTPATH_RE = re.compile(
    r"^\s*-not\s*\(\s*(Test-Path\b.*?)\s*\)\s*$",
    re.IGNORECASE | re.DOTALL,
)


def _normalize_ps_testpath(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    m = _PS_NOT_TESTPATH_RE.match(value)
    return m.group(1).strip() if m else value


# ---------------------------------------------------------------------------
# Built-in checks
# ---------------------------------------------------------------------------


def check_is_valid_json(output: str, gold: GoldRecord | None, inp: EvalRecord, params: Any) -> CheckReturn:
    try:
        _extract_json(output)
        return CheckResult(100.0)
    except ValueError:
        return CheckResult(0.0, "output did not contain parseable JSON")


def check_has_keys(output: str, gold: GoldRecord | None, inp: EvalRecord, params: Any) -> CheckReturn:
    keys: list[str]
    if isinstance(params, list):
        keys = list(params)
    elif isinstance(params, dict) and "keys" in params:
        keys = list(params["keys"])
    else:
        raise ValueError("has_keys requires params as list[str] or {keys: [...]}")
    try:
        obj = _extract_json(output)
    except ValueError:
        return CheckResult(0.0, "JSON unparseable")
    if not isinstance(obj, dict):
        return CheckResult(0.0, f"top-level JSON is {type(obj).__name__}, expected object")
    present = [k for k in keys if k in obj]
    missing = [k for k in keys if k not in obj]
    score = 100.0 * len(present) / len(keys) if keys else 100.0
    detail = "" if not missing else f"missing keys: {', '.join(missing)}"
    return CheckResult(score, detail)


def check_exact_match_against_gold(
    output: str, gold: GoldRecord | None, inp: EvalRecord, params: Any
) -> CheckReturn:
    """Partial-credit JSON comparison against gold.

    When both sides parse as JSON objects, scores ``100 * matches / expected``
    and the detail lists the exact mismatched fields with their expected and
    actual values. This is the feedback channel the Overseer uses to narrow in
    on which field is wrong (identified in the Stage 2 smoke test).

    Optional ``params={"fields": [...]}`` restricts the comparison to a
    subset of gold keys. Useful when some fields are subjective / tolerance-
    band scored by a separate rule and shouldn't participate in exact match.
    Motivated by Stage-8 F1 on jarvis_routing_opus (2026-04-22).
    """
    if gold is None:
        return CheckResult(0.0, "no gold standard for this example")
    try:
        out_obj = _extract_json(output)
    except ValueError:
        out_obj = None
    gold_obj = gold.output

    fields_filter: set[str] | None = None
    if isinstance(params, dict) and "fields" in params:
        raw = params["fields"]
        if not isinstance(raw, list) or not all(isinstance(x, str) for x in raw):
            raise ValueError("exact_match_against_gold params.fields must be list[str]")
        fields_filter = set(raw)

    # Dict vs dict → field-level partial credit + diff detail.
    if isinstance(out_obj, dict) and isinstance(gold_obj, dict):
        if not gold_obj:
            return CheckResult(100.0 if out_obj == gold_obj else 0.0)
        matches: list[str] = []
        mismatches: list[str] = []
        missing: list[str] = []
        keys_considered: list[str] = []
        for k, expected in gold_obj.items():
            if fields_filter is not None and k not in fields_filter:
                continue
            keys_considered.append(k)
            if k not in out_obj:
                missing.append(k)
                continue
            # Apply PowerShell Test-Path normalization symmetrically. See the
            # header comment on _PS_NOT_TESTPATH_RE for why this is safe:
            # values that don't match the narrow `-not ( Test-Path ... )`
            # shape pass through unchanged, so no other field / task is
            # affected. Stage-8 Analyst finding I3 (2026-04-21).
            expected_norm = _normalize_ps_testpath(expected)
            actual_norm = _normalize_ps_testpath(out_obj[k])
            if actual_norm == expected_norm:
                matches.append(k)
            else:
                mismatches.append(
                    f"{k}: expected={_short(expected)} got={_short(out_obj[k])}"
                )
        total = len(keys_considered) if fields_filter is not None else len(gold_obj)
        # fields_filter may name keys the gold doesn't contain — treat as 0/0
        # rather than dividing by zero.
        if total == 0:
            return CheckResult(100.0, "no fields in scope")
        correct = len(matches)
        score = 100.0 * correct / total
        parts: list[str] = []
        if matches:
            parts.append(f"match: {', '.join(matches)}")
        if mismatches:
            parts.append("mismatch: " + "; ".join(mismatches))
        if missing:
            parts.append(f"missing: {', '.join(missing)}")
        return CheckResult(score, " | ".join(parts))

    # List vs list → element-wise partial credit (order-sensitive).
    if isinstance(out_obj, list) and isinstance(gold_obj, list):
        if not gold_obj:
            return CheckResult(100.0 if out_obj == gold_obj else 0.0)
        n = max(len(out_obj), len(gold_obj))
        correct = sum(
            1
            for i in range(min(len(out_obj), len(gold_obj)))
            if out_obj[i] == gold_obj[i]
        )
        return CheckResult(
            100.0 * correct / n,
            f"list match {correct}/{n}; lens(out={len(out_obj)}, gold={len(gold_obj)})",
        )

    # Fallback: whole-value equality.
    if out_obj is not None and gold_obj is not None:
        ok = out_obj == gold_obj
    else:
        ok = output.strip() == str(gold_obj).strip()
    return CheckResult(100.0 if ok else 0.0, "" if ok else f"expected={_short(gold_obj)} got={_short(output)}")


def check_substring_match(output: str, gold: GoldRecord | None, inp: EvalRecord, params: Any) -> CheckReturn:
    needles: list[str]
    if isinstance(params, list):
        needles = [str(x) for x in params]
        present = [n for n in needles if n in output]
        missing = [n for n in needles if n not in output]
        score = 100.0 * len(present) / len(needles) if needles else 100.0
        return CheckResult(score, "" if not missing else f"missing substrings: {', '.join(missing)}")
    if isinstance(params, dict) and "any_of" in params:
        needles = [str(x) for x in params["any_of"]]
        hit = any(n in output for n in needles)
        return CheckResult(100.0 if hit else 0.0, "" if hit else f"none of {needles} present")
    raise ValueError("substring_match requires params as list[str] or {any_of: [...]}")


def check_regex_match(output: str, gold: GoldRecord | None, inp: EvalRecord, params: Any) -> CheckReturn:
    if isinstance(params, list):
        pattern = params[0]
    elif isinstance(params, dict):
        pattern = params["pattern"]
    else:
        pattern = params
    ok = re.search(str(pattern), output) is not None
    return CheckResult(100.0 if ok else 0.0, "" if ok else f"pattern {pattern!r} did not match")


def check_numeric_range(
    output: str, gold: GoldRecord | None, inp: EvalRecord, params: Any
) -> CheckReturn:
    """Score 100 if ``output[field]`` is a number in ``[min, max]``, else 0.

    Gold-independent: this check exists for fields whose gold value is a
    placeholder / subjective estimate that cannot meaningfully be exact-
    matched (e.g. a self-reported confidence). Introduced for Stage-8 F1 on
    jarvis_routing_opus (2026-04-22): confidence was exact-matched against
    hidden per-example golds, capping scores at 90 despite being specified
    as a subjective self-estimate in the task description.

    params: ``{"field": str, "min": float (optional), "max": float (optional)}``.
    Missing min/max default to -inf / +inf, i.e. a pure "must be a number"
    check. Non-numeric values and missing fields always score 0.
    """
    if not isinstance(params, dict) or "field" not in params:
        raise ValueError("numeric_range requires params={field, min?, max?}")
    field = params["field"]
    lo = float(params.get("min", float("-inf")))
    hi = float(params.get("max", float("inf")))
    try:
        obj = _extract_json(output)
    except ValueError:
        return CheckResult(0.0, "JSON unparseable")
    if not isinstance(obj, dict):
        return CheckResult(0.0, f"top-level JSON is {type(obj).__name__}, expected object")
    if field not in obj:
        return CheckResult(0.0, f"field {field!r} missing")
    val = obj[field]
    # Reject bools even though bool is a subclass of int — a confidence of
    # `true` should not score as 1.0.
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return CheckResult(0.0, f"field {field!r} is {type(val).__name__}, expected number")
    if lo <= float(val) <= hi:
        return CheckResult(100.0, f"{field}={val} in [{lo}, {hi}]")
    return CheckResult(0.0, f"{field}={val} not in [{lo}, {hi}]")


CHECK_REGISTRY: dict[str, CheckFn] = {
    "is_valid_json": check_is_valid_json,
    "has_keys": check_has_keys,
    "exact_match_against_gold": check_exact_match_against_gold,
    "numeric_range": check_numeric_range,
    "substring_match": check_substring_match,
    "regex_match": check_regex_match,
}


def register_check(name: str, fn: CheckFn) -> None:
    """Plug in a user-defined check at runtime."""
    CHECK_REGISTRY[name] = fn


def _coerce(result: CheckReturn) -> CheckResult:
    if isinstance(result, CheckResult):
        return result
    return CheckResult(score=float(result))


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


class DeterministicScorer(Scorer):
    def __init__(self, rules: list[DeterministicRule]) -> None:
        if not rules:
            raise ValueError("DeterministicScorer requires at least one rule")
        for r in rules:
            if r.check not in CHECK_REGISTRY:
                raise ValueError(f"Unknown deterministic check: {r.check!r}")
        self.rules = rules
        self._total_weight = sum(r.weight for r in rules) or 1.0

    async def score(
        self,
        output: str | bytes,
        gold: GoldRecord | None,
        input_record: EvalRecord,
        context: ScoringContext,
    ) -> ScoreResult:
        text = output.decode("utf-8", errors="replace") if isinstance(output, bytes) else output
        per: dict[str, float] = {}
        weighted_sum = 0.0
        parts: list[str] = []
        for rule in self.rules:
            fn = CHECK_REGISTRY[rule.check]
            result = _coerce(fn(text, gold, input_record, rule.params))
            per[rule.name] = result.score
            weighted_sum += result.score * rule.weight
            label = f"{rule.name}={result.score:.0f}"
            if result.detail:
                label += f" [{result.detail}]"
            parts.append(label)
        aggregate = weighted_sum / self._total_weight
        return ScoreResult(aggregate=aggregate, per_criterion=per, rationale="; ".join(parts)).clamp()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_scorer(metric: MetricConfig, **_ignored: Any) -> Scorer:
    """Lightweight factory used when only deterministic metrics are needed.

    The full factory — which needs a judge client for rubric / conversational
    metrics — lives in :mod:`lpo.scoring.factory` so this module stays free of
    Anthropic imports.
    """
    if isinstance(metric, DeterministicMetric):
        return DeterministicScorer(metric.rules)
    raise NotImplementedError(
        f"build_scorer() in deterministic.py only handles deterministic metrics; "
        f"got {type(metric).__name__}. Use lpo.scoring.factory.build_scorer for "
        "rubric / conversational metrics."
    )
