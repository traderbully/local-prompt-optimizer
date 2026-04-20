"""Scorer abstraction. See `LPO_SDP.md` §5.2, §4.4.

Two entry points:

* :meth:`Scorer.score` — score a single (output, gold, input) triple.
  Most deterministic and rubric checks are naturally per-example.
* :meth:`Scorer.score_iteration` — score an entire batch of outputs for an
  iteration in one go. The default implementation fans out to :meth:`score`.
  Stateful scorers (e.g. Type C conversational) override this so they can
  share a single overseer call across the whole eval set.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from lpo.config.schema import EvalRecord, GoldRecord


@dataclass
class ScoringContext:
    """Context passed to each scorer invocation."""

    task_name: str
    iteration_index: int
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoreResult:
    """Result of scoring a single output.

    ``aggregate`` is on a 0..100 scale. ``per_criterion`` maps criterion name to
    its raw 0..100 sub-score. ``rationale`` is a short human-readable summary —
    this is the string the Overseer sees in failed-example rows, so it should
    be specific (e.g. ``"location: expected='online' got='Online'"``).
    """

    aggregate: float
    per_criterion: dict[str, float] = field(default_factory=dict)
    rationale: str = ""

    def clamp(self) -> "ScoreResult":
        self.aggregate = max(0.0, min(100.0, self.aggregate))
        self.per_criterion = {k: max(0.0, min(100.0, v)) for k, v in self.per_criterion.items()}
        return self


class Scorer(ABC):
    @abstractmethod
    async def score(
        self,
        output: str | bytes,
        gold: GoldRecord | None,
        input_record: EvalRecord,
        context: ScoringContext,
    ) -> ScoreResult: ...

    async def score_iteration(
        self,
        *,
        outputs: dict[str, str],
        eval_records: list[EvalRecord],
        gold_standard: dict[str, GoldRecord],
        context: ScoringContext,
    ) -> dict[str, ScoreResult]:
        """Score an entire iteration. Default: fan out to ``score`` per example."""

        async def _one(rec: EvalRecord) -> tuple[str, ScoreResult]:
            out = outputs.get(rec.id, "")
            gold = gold_standard.get(rec.id)
            return rec.id, await self.score(out, gold, rec, context)

        results = await asyncio.gather(*[_one(r) for r in eval_records])
        return dict(results)

    async def aclose(self) -> None:
        """Release scorer-owned resources (e.g. Anthropic client)."""
        return None
