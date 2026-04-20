"""Overseer-driven prompt mutator. See `LPO_SDP.md` §3.1, §5.3, §6.1.

The Overseer sees the task goal, the metric rubric, a rolling summary of past
iterations, and the verbatim record of the most recent iterations. It responds
with a tagged structure (``<analysis>``, ``<hypothesis>``, ``<prompt>``) that
the parser turns into a :class:`MutationProposal`.

Defensive behaviour:
  * Malformed responses trigger exactly one clarifying retry before we fall
    back to the current best prompt (i.e. the engine will hit ``MUTATOR_NOOP``
    and stop cleanly instead of exploding).
  * When the conversation token estimate exceeds ``max_tokens``, oldest turns
    are collapsed into ``ConversationContext.summary`` via a dedicated
    summarization call.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from lpo.config.schema import (
    ConversationalMetric,
    DeterministicMetric,
    MetricConfig,
    RubricMetric,
)
from lpo.core.history import IterationRecord
from lpo.core.mutator import MutationProposal, PromptMutator
from lpo.core.task import TaskBundle
from lpo.models.anthropic_client import AnthropicClient, AnthropicMessage
from lpo.overseer.context import (
    ConversationContext,
    IterationTurn,
    format_iteration_turn,
)
from lpo.overseer.prompt_writer import OverseerResponse, parse_overseer_response

log = logging.getLogger("lpo.overseer")


OVERSEER_SYSTEM_TEMPLATE = """\
You are the Overseer for an autonomous prompt-optimization loop. A local
open-weight model (the "Target") is being asked to perform a user-defined task;
its outputs are scored against a fixed metric, and you iteratively rewrite the
Target's system prompt to raise the score.

## Task goal

{task_md}

## Metric

{metric_summary}

## History summary

{history_summary}

## Response format — required

Respond with exactly these three XML-style tags, in this order:

<analysis>
Free text. Diagnose *why* the failing examples failed. Tie each failure to a
concrete mechanism: missing constraint, ambiguous instruction, wrong format,
etc. Keep this under 200 words.
</analysis>
<hypothesis>
One sentence describing the change you're about to make and what score
movement you expect.
</hypothesis>
<prompt>
The complete new system prompt for the Target. This replaces the previous
prompt wholesale — do not output a diff. No surrounding prose, no code fences,
no commentary. Just the prompt body.
</prompt>

Rules:
- Optimize for the *metric*, not just subjective quality.
- Never regress on currently passing scenarios. If you must trade off, note it.
- Keep the new prompt concise; bloat is a failure mode.
- If the previous iteration was accepted, prefer small refinements. If it was
  rejected, try a materially different approach.
"""


SUMMARIZATION_SYSTEM = """\
You are compressing an Overseer conversation so it fits in a bounded context.
Produce a single dense paragraph that records:
  - which prompt directions have been tried and their score outcomes,
  - which scenarios remain weak and why,
  - any hypotheses the Overseer has ruled out.

No preamble, no markdown headings. Just the paragraph. Keep it under 400 words.
"""


def _format_metric(metric: MetricConfig) -> str:
    if isinstance(metric, DeterministicMetric):
        lines = ["Deterministic rules (weighted 0-100 each, combined by weight):"]
        for r in metric.rules:
            tail = f" params={r.params}" if r.params is not None else ""
            lines.append(f"  - {r.name} (weight={r.weight}) check={r.check}{tail}")
        return "\n".join(lines)
    if isinstance(metric, RubricMetric):
        lines = [f"Rubric (judge={metric.judge_model}):"]
        for c in metric.criteria:
            lines.append(f"  - {c.name} (weight={c.weight}): {c.description}")
        return "\n".join(lines)
    if isinstance(metric, ConversationalMetric):
        return f"Conversational, stated goal:\n{metric.stated_goal}"
    return str(metric)  # pragma: no cover


@dataclass
class OverseerMutator(PromptMutator):
    """PromptMutator driven by an Anthropic overseer conversation."""

    client: AnthropicClient
    task: TaskBundle
    context: ConversationContext = field(default_factory=ConversationContext)
    temperature: float = 0.4
    max_tokens_per_call: int = 4096
    failed_sample_limit: int = 3
    consecutive_noops: int = 0

    def __post_init__(self) -> None:
        self._system_prompt_cache: str | None = None

    # ------------------------------------------------------------------
    # PromptMutator
    # ------------------------------------------------------------------

    async def propose(
        self,
        *,
        current_prompt: str,
        best_prompt: str,
        history: list[IterationRecord],
        user_feedback: str = "",
    ) -> MutationProposal:
        if not history:  # pragma: no cover - engine always has history
            return MutationProposal(new_prompt=current_prompt, rationale="no history yet")

        latest = history[-1]
        turn = IterationTurn.from_record(
            latest,
            failed_sample_limit=self.failed_sample_limit,
            user_feedback=user_feedback,
        )
        user_msg = format_iteration_turn(turn)

        await self._maybe_summarize(pending_user=user_msg)

        parsed, raw_text = await self._call_overseer(user_msg)

        # One clarifying retry on malformed output.
        if parsed is None:
            log.warning("Overseer response malformed; issuing clarifying retry.")
            clarifier = (
                "Your previous response did not contain a valid <prompt>...</prompt> "
                "block. Please re-send using EXACTLY the required three-tag format: "
                "<analysis>...</analysis><hypothesis>...</hypothesis><prompt>...</prompt>."
            )
            # Record the malformed exchange so the retry sees the full thread.
            self.context.add_user(user_msg)
            self.context.add_assistant(raw_text)
            parsed, raw_text = await self._call_overseer(clarifier)
            user_msg = clarifier  # so the final add_user below records the clarifier

        # Persist turns in context regardless of outcome.
        if parsed is None:
            # Give up: return current best so engine terminates with MUTATOR_NOOP.
            log.error("Overseer produced no parseable prompt after retry; halting mutation.")
            return MutationProposal(
                new_prompt=best_prompt,
                rationale="Overseer failed to produce a valid <prompt> block after retry.",
                analysis=raw_text[:500],
            )

        self.context.add_user(user_msg)
        self.context.add_assistant(raw_text)

        return MutationProposal(
            new_prompt=parsed.new_prompt,
            rationale=parsed.hypothesis,
            analysis=parsed.analysis,
        )

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _system_prompt(self) -> str:
        return OVERSEER_SYSTEM_TEMPLATE.format(
            task_md=self.task.task_md.strip() or "(no task.md provided)",
            metric_summary=_format_metric(self.task.metric),
            history_summary=self.context.summary.strip() or "(none yet — this is the start of the run)",
        )

    async def _call_overseer(self, user_text: str) -> tuple[OverseerResponse | None, str]:
        messages = list(self.context.turns) + [AnthropicMessage(role="user", content=user_text)]
        result = await self.client.complete(
            system=self._system_prompt(),
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens_per_call,
        )
        parsed = parse_overseer_response(result.text)
        return parsed, result.text

    async def _maybe_summarize(self, *, pending_user: str) -> None:
        if not self.context.needs_summarization(pending_user):
            return
        old, _recent = self.context._fold_candidates()
        if not old:
            return
        log.info("Summarizing %d old turns to stay under token budget.", len(old))
        # Build a single user message that walks the summarizer through the dropped turns.
        walk_parts: list[str] = []
        if self.context.summary:
            walk_parts.append("PRIOR SUMMARY:\n" + self.context.summary)
        for t in old:
            walk_parts.append(f"{t.role.upper()}:\n{t.content}")
        walk = "\n\n---\n\n".join(walk_parts)
        result = await self.client.complete(
            system=SUMMARIZATION_SYSTEM,
            messages=[AnthropicMessage(role="user", content=walk)],
            temperature=0.2,
            max_tokens=1024,
        )
        self.context.apply_summary(result.text)
