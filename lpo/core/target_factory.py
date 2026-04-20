"""Per-target wiring.

A "target context" is everything the single-target :class:`RatchetEngine`
needs to operate on one model: a target client, a scorer instance, a
mutator (with its own Overseer conversation state if applicable), and a
list of async resources to close on teardown.

The multi-target orchestrator (:mod:`lpo.core.multi_engine`) builds one
context per model. Contexts are always fresh — they never share judge
conversations or Overseer state, which is what the SDP requires for
Strategy B (isolated per-model optimization).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from lpo.config.schema import TargetModelConfig
from lpo.core.cost import CostTracker
from lpo.core.mutator import NullMutator, PromptMutator
from lpo.core.task import TaskBundle
from lpo.models.base import ModelClient
from lpo.models.registry import build_client
from lpo.scoring.base import Scorer
from lpo.scoring.factory import build_scorer, default_judge_factory

log = logging.getLogger("lpo.target_factory")

AsyncCleanup = Callable[[], Awaitable[None]]


@dataclass
class TargetContext:
    """Everything the ratchet loop needs for one model."""

    cfg: TargetModelConfig
    client: ModelClient
    scorer: Scorer
    mutator: PromptMutator
    cleanups: list[AsyncCleanup] = field(default_factory=list)

    @property
    def slug(self) -> str:
        return self.cfg.slug

    async def aclose(self) -> None:
        """Release all target-owned async resources."""
        for fn in self.cleanups:
            try:
                await fn()
            except Exception:  # noqa: BLE001
                log.exception("cleanup %s raised", fn)


def build_target_context(
    task: TaskBundle,
    target: TargetModelConfig,
    cost: CostTracker,
    *,
    mutator_mode: str = "auto",
) -> TargetContext:
    """Build a fresh :class:`TargetContext` for ``target``.

    ``mutator_mode`` is one of ``auto | overseer | null`` and matches the CLI
    flag. ``auto`` enables the Overseer when ``mode=autonomous`` and
    ``ANTHROPIC_API_KEY`` is set.

    Scorers are built with their own judge client when rubric or conversational
    — the factory wires them through the shared ``cost`` tracker so run totals
    stay correct.
    """
    client = build_client(target, cost_tracker=cost)

    judge_factory = default_judge_factory(cost)
    scorer = build_scorer(task.metric, judge_factory=judge_factory, cost_tracker=cost)

    mutator, overseer_client = _build_mutator(mutator_mode, task, cost)

    cleanups: list[AsyncCleanup] = [client.aclose, scorer.aclose]
    if overseer_client is not None:
        cleanups.append(overseer_client.aclose)

    return TargetContext(
        cfg=target,
        client=client,
        scorer=scorer,
        mutator=mutator,
        cleanups=cleanups,
    )


def _build_mutator(
    mutator_mode: str,
    task: TaskBundle,
    cost: CostTracker,
) -> tuple[PromptMutator, object | None]:
    """Mirror of the CLI's mutator selection, extracted so multi-engine paths
    don't duplicate the logic. Returns (mutator, closeable_or_None)."""
    name = (mutator_mode or "auto").lower()
    if name == "null":
        return NullMutator(), None
    if name not in {"auto", "overseer"}:
        raise ValueError(f"mutator_mode must be auto|overseer|null, got {mutator_mode!r}")

    want_overseer = name == "overseer" or (
        name == "auto"
        and task.config.mode == "autonomous"
        and os.environ.get("ANTHROPIC_API_KEY")
    )
    if not want_overseer:
        if name == "overseer":
            raise ValueError(
                "mutator_mode=overseer requires ANTHROPIC_API_KEY in the environment."
            )
        return NullMutator(), None

    from lpo.models.anthropic_client import AnthropicClient
    from lpo.overseer.agent import OverseerMutator

    overseer_cfg = task.config.overseer_model
    client = AnthropicClient(
        model_id=overseer_cfg.model_id,
        api_key_env=overseer_cfg.api_key_env,
        cost_tracker=cost,
    )
    return OverseerMutator(client=client, task=task), client
