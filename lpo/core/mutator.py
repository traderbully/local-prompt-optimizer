"""Prompt mutator abstraction.

Stage 1 ships ``NullMutator`` which returns the prompt unchanged — this exercises
the ratchet scaffolding end-to-end against the seed prompt. Stage 2 will add
``OverseerMutator`` which delegates to a frontier model (see `LPO_SDP.md` §5.3).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from lpo.core.history import IterationRecord


@dataclass
class MutationProposal:
    new_prompt: str
    rationale: str = ""
    analysis: str = ""


class PromptMutator(ABC):
    """Produces a new prompt given the history of iterations so far.

    ``user_feedback`` is an optional free-text nudge from the user — populated
    by :class:`lpo.core.engine.RatchetEngine` when it's running under Manual
    mode (and non-empty thumbs/rating signals under Supervised). Mutators that
    understand the overseer context (:class:`~lpo.overseer.agent.OverseerMutator`)
    inject this into the next iteration turn; simple mutators ignore it.
    """

    @abstractmethod
    async def propose(
        self,
        *,
        current_prompt: str,
        best_prompt: str,
        history: list[IterationRecord],
        user_feedback: str = "",
    ) -> MutationProposal: ...


class NullMutator(PromptMutator):
    """Returns the prompt unchanged. Used for Stage 1 scaffolding and tests."""

    async def propose(
        self,
        *,
        current_prompt: str,
        best_prompt: str,
        history: list[IterationRecord],
        user_feedback: str = "",
    ) -> MutationProposal:
        return MutationProposal(
            new_prompt=best_prompt,
            rationale="NullMutator: no mutation applied (Stage 1 scaffolding).",
        )
