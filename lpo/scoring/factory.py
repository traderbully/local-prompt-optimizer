"""Scorer factory. Imports Anthropic lazily so deterministic-only callers
don't pay the import cost."""

from __future__ import annotations

from typing import Callable

from lpo.config.schema import (
    ConversationalMetric,
    DeterministicMetric,
    MetricConfig,
    RubricMetric,
)
from lpo.core.cost import CostTracker
from lpo.scoring.base import Scorer
from lpo.scoring.deterministic import DeterministicScorer


JudgeFactory = Callable[[str], "AnthropicClient"]  # noqa: F821 - forward ref
"""Callable: ``model_id -> AnthropicClient``. Must return a fresh client the
caller is responsible for closing."""


def build_scorer(
    metric: MetricConfig,
    *,
    judge_factory: JudgeFactory | None = None,
    cost_tracker: CostTracker | None = None,
    rubric_concurrency: int = 4,
) -> Scorer:
    """Return the appropriate :class:`Scorer` for the given metric config.

    ``judge_factory`` is only required for rubric / conversational metrics and
    is injected rather than imported so that tests can swap in a stub client
    without touching the network.
    """
    if isinstance(metric, DeterministicMetric):
        return DeterministicScorer(metric.rules)

    if isinstance(metric, RubricMetric):
        from lpo.scoring.rubric import RubricScorer

        if judge_factory is None:
            raise ValueError("Rubric metrics require a judge_factory")
        judge = judge_factory(metric.judge_model)
        return RubricScorer(metric, judge, concurrency=rubric_concurrency)

    if isinstance(metric, ConversationalMetric):
        from lpo.scoring.conversational import ConversationalScorer

        if judge_factory is None:
            raise ValueError("Conversational metrics require a judge_factory")
        judge = judge_factory(metric.overseer_model)
        return ConversationalScorer(metric, judge)

    raise ValueError(f"Unknown metric type: {type(metric).__name__}")


def default_judge_factory(cost_tracker: CostTracker) -> JudgeFactory:
    """Return a judge factory that builds real :class:`AnthropicClient` instances
    sharing the provided ``cost_tracker``."""
    from lpo.models.anthropic_client import AnthropicClient

    def make(model_id: str) -> AnthropicClient:
        return AnthropicClient(model_id=model_id, cost_tracker=cost_tracker)

    return make
