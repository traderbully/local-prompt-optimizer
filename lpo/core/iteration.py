"""Single-iteration executor: run prompt against eval set and score.

See `LPO_SDP.md` §5.4 (concurrency), §5.2 (interfaces).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from lpo.config.schema import EvalRecord, GoldRecord, TargetModelConfig
from lpo.core.history import IterationRecord
from lpo.models.base import ModelClient
from lpo.scoring.aggregation import AggregatedScore, aggregate_scores
from lpo.scoring.base import ScoreResult, Scorer, ScoringContext


def _seed_for(record: EvalRecord, cfg: TargetModelConfig) -> int | None:
    if cfg.seed_policy == "unlocked":
        return None
    if cfg.seed_policy == "fixed":
        return cfg.fixed_seed
    # fixed_per_example — stable hash of id
    h = 0
    for ch in record.id:
        h = (h * 31 + ord(ch)) & 0xFFFFFFFF
    return h ^ cfg.fixed_seed


def _format_input(value: Any) -> str:
    if isinstance(value, str):
        return value
    import json as _json

    return _json.dumps(value, ensure_ascii=False)


@dataclass
class IterationResult:
    record: IterationRecord
    aggregated: AggregatedScore


class IterationRunner:
    def __init__(
        self,
        *,
        client: ModelClient,
        target_cfg: TargetModelConfig,
        scorer: Scorer,
        concurrency: int = 4,
    ) -> None:
        self.client = client
        self.target_cfg = target_cfg
        self.scorer = scorer
        self.semaphore = asyncio.Semaphore(max(1, concurrency))

    async def _generate_one(
        self,
        prompt: str,
        record: EvalRecord,
    ) -> dict[str, Any]:
        """Generate one target output. Scoring is a separate phase so that
        batch/stateful scorers (Type C) can see the whole eval set at once."""
        async with self.semaphore:
            gen_start = time.perf_counter()
            gen = await self.client.generate(
                system_prompt=prompt,
                user_input=_format_input(record.input),
                seed=_seed_for(record, self.target_cfg),
                temperature=self.target_cfg.temperature,
                max_tokens=self.target_cfg.max_tokens,
            )
            gen_latency = time.perf_counter() - gen_start
        return {
            "id": record.id,
            "scenario": record.scenario,
            "input": record.input,
            "output": gen.text,
            "seed": gen.seed,
            "prompt_tokens": gen.prompt_tokens,
            "completion_tokens": gen.completion_tokens,
            "latency_s": round(gen_latency, 4),
        }

    async def run(
        self,
        *,
        iteration_index: int,
        prompt: str,
        eval_records: list[EvalRecord],
        gold_standard: dict[str, GoldRecord],
        task_name: str,
    ) -> IterationResult:
        t0 = time.perf_counter()

        # Phase 1: generate all target outputs concurrently.
        gen_t0 = time.perf_counter()
        rows = await asyncio.gather(*[self._generate_one(prompt, rec) for rec in eval_records])
        gen_elapsed = time.perf_counter() - gen_t0
        outputs = {row["id"]: row["output"] for row in rows}

        # Phase 2: score the batch. Per-example scorers fall through to the
        # base Scorer.score_iteration fan-out.
        score_t0 = time.perf_counter()
        ctx = ScoringContext(task_name=task_name, iteration_index=iteration_index)
        score_map = await self.scorer.score_iteration(
            outputs=outputs,
            eval_records=eval_records,
            gold_standard=gold_standard,
            context=ctx,
        )
        score_elapsed = time.perf_counter() - score_t0

        # Attach scores to rows in the original eval order.
        for row in rows:
            s = score_map.get(row["id"])
            row["score"] = s.aggregate if s else 0.0
            row["per_criterion"] = s.per_criterion if s else {}
            row["rationale"] = s.rationale if s else "no score returned"

        aggregated = aggregate_scores(eval_records, score_map)
        elapsed = time.perf_counter() - t0

        record = IterationRecord(
            index=iteration_index,
            prompt=prompt,
            aggregate_score=aggregated.aggregate,
            per_example=aggregated.per_example,
            per_scenario=aggregated.per_scenario,
            failed_ids=aggregated.failed_ids,
            outputs=rows,
            decision="pending",
            timings={
                "total_s": round(elapsed, 3),
                "generate_s": round(gen_elapsed, 3),
                "score_s": round(score_elapsed, 3),
            },
        )
        return IterationResult(record=record, aggregated=aggregated)
