"""Aggregation helpers across eval examples. See `LPO_SDP.md` §4.2, §5.3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from lpo.config.schema import EvalRecord
from lpo.scoring.base import ScoreResult


@dataclass
class AggregatedScore:
    aggregate: float
    per_example: dict[str, float]
    per_scenario: dict[str, float]
    failed_ids: list[str]


def aggregate_scores(
    eval_records: Iterable[EvalRecord],
    results: dict[str, ScoreResult],
    *,
    fail_threshold: float = 70.0,
) -> AggregatedScore:
    per_example: dict[str, float] = {}
    weighted_sum = 0.0
    total_weight = 0.0
    failed: list[str] = []
    for rec in eval_records:
        r = results.get(rec.id)
        if r is None:
            continue
        per_example[rec.id] = r.aggregate
        weighted_sum += r.aggregate * rec.weight
        total_weight += rec.weight
        if r.aggregate < fail_threshold:
            failed.append(rec.id)
    aggregate = weighted_sum / total_weight if total_weight else 0.0
    return AggregatedScore(
        aggregate=aggregate,
        per_example=per_example,
        per_scenario=scenario_breakdown(eval_records, results),
        failed_ids=failed,
    )


def scenario_breakdown(
    eval_records: Iterable[EvalRecord],
    results: dict[str, ScoreResult],
) -> dict[str, float]:
    buckets: dict[str, list[tuple[float, float]]] = {}
    for rec in eval_records:
        if rec.scenario is None:
            continue
        r = results.get(rec.id)
        if r is None:
            continue
        buckets.setdefault(rec.scenario, []).append((r.aggregate, rec.weight))
    out: dict[str, float] = {}
    for name, items in buckets.items():
        wsum = sum(s * w for s, w in items)
        tw = sum(w for _, w in items)
        out[name] = wsum / tw if tw else 0.0
    return out
