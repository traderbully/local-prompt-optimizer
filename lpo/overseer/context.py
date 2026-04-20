"""Overseer conversation context: turn formatting, token budgeting, summarization.

See `LPO_SDP.md` §5.3. The overseer maintains a running conversation thread
per run. When total token estimate exceeds ``max_tokens``, the oldest turn pairs
beyond ``keep_recent`` are folded into a single summary block that is prepended
to the next user message (via the system prompt's ``<history_summary>`` section).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lpo.core.history import IterationRecord
from lpo.models.anthropic_client import AnthropicMessage


def estimate_tokens(text: str) -> int:
    """Cheap token estimate. Anthropic's real tokenizer would be more accurate,
    but ~4 chars/token is close enough for budgeting purposes."""
    return max(1, len(text) // 4)


@dataclass
class IterationTurn:
    """Summary of a single iteration, used to build overseer prompts."""

    index: int
    prompt: str
    aggregate_score: float
    delta: float
    decision: str
    per_scenario: dict[str, float] = field(default_factory=dict)
    failed_ids: list[str] = field(default_factory=list)
    failed_examples: list[dict] = field(default_factory=list)  # {id, input, output, score, rationale}
    user_feedback: str = ""

    @classmethod
    def from_record(
        cls,
        record: IterationRecord,
        *,
        failed_sample_limit: int = 3,
        user_feedback: str = "",
    ) -> "IterationTurn":
        failed: list[dict] = []
        for row in record.outputs:
            if row.get("id") in record.failed_ids and len(failed) < failed_sample_limit:
                failed.append(
                    {
                        "id": row.get("id"),
                        "input": row.get("input"),
                        "output": row.get("output"),
                        "score": row.get("score"),
                        "rationale": row.get("rationale"),
                    }
                )
        return cls(
            index=record.index,
            prompt=record.prompt,
            aggregate_score=record.aggregate_score,
            delta=record.delta,
            decision=record.decision,
            per_scenario=dict(record.per_scenario),
            failed_ids=list(record.failed_ids),
            failed_examples=failed,
            user_feedback=user_feedback,
        )


def format_iteration_turn(turn: IterationTurn) -> str:
    """Render an iteration as a user-message body. Matches `LPO_SDP.md` §5.3."""
    lines: list[str] = []
    lines.append(f"ITERATION {turn.index}")
    lines.append("")
    lines.append("Prompt used:")
    lines.append("```")
    lines.append(turn.prompt.rstrip())
    lines.append("```")
    lines.append("")
    lines.append("Eval results summary:")
    lines.append(f"- Aggregate score: {turn.aggregate_score:.2f} (delta {turn.delta:+.2f})")
    lines.append(f"- Ratchet decision: {turn.decision}")
    if turn.per_scenario:
        breakdown = ", ".join(f"{k}={v:.1f}" for k, v in sorted(turn.per_scenario.items()))
        lines.append(f"- Scenario breakdown: {breakdown}")
    if turn.failed_ids:
        lines.append(f"- Failed example ids: {', '.join(turn.failed_ids)}")
    if turn.failed_examples:
        lines.append("")
        lines.append("Failed example details:")
        for ex in turn.failed_examples:
            lines.append(f"  - id={ex['id']} score={ex['score']:.1f} rationale={ex['rationale']}")
            lines.append(f"    input:  {_oneline(ex['input'])}")
            lines.append(f"    output: {_oneline(ex['output'])}")
    if turn.user_feedback:
        lines.append("")
        lines.append(f"User feedback: {turn.user_feedback!r}")
    lines.append("")
    lines.append(
        "Your task: Analyze the failures, then propose a prompt edit that addresses "
        "them without regressing on passing scenarios. Respond in the required tag format."
    )
    return "\n".join(lines)


def _oneline(value: object, max_len: int = 240) -> str:
    s = value if isinstance(value, str) else str(value)
    s = s.replace("\n", " ").replace("\r", " ")
    if len(s) > max_len:
        s = s[: max_len - 1] + "…"
    return s


# ---------------------------------------------------------------------------
# ConversationContext
# ---------------------------------------------------------------------------


@dataclass
class ConversationContext:
    """Running overseer conversation with budgeted summarization.

    ``turns`` is the verbatim message log (user/assistant alternation).
    ``summary`` is a rolling natural-language summary of turns already folded
    out of the verbatim log. Both are fed into each new overseer call:
    ``summary`` inside the system prompt, ``turns`` as the messages.
    """

    max_tokens: int = 100_000
    keep_recent_pairs: int = 10  # number of user/assistant pairs to retain verbatim
    summary: str = ""
    turns: list[AnthropicMessage] = field(default_factory=list)

    def add_user(self, text: str) -> None:
        self.turns.append(AnthropicMessage(role="user", content=text))

    def add_assistant(self, text: str) -> None:
        self.turns.append(AnthropicMessage(role="assistant", content=text))

    def token_estimate(self, extra: str = "") -> int:
        return estimate_tokens(self.summary) + sum(
            estimate_tokens(t.content) for t in self.turns
        ) + estimate_tokens(extra)

    def needs_summarization(self, pending_user: str = "") -> bool:
        """True when adding ``pending_user`` would push us over the budget and
        there is at least one older turn pair we could fold away."""
        if len(self.turns) <= self.keep_recent_pairs * 2:
            return False
        return self.token_estimate(pending_user) > self.max_tokens

    def _fold_candidates(self) -> tuple[list[AnthropicMessage], list[AnthropicMessage]]:
        keep = self.keep_recent_pairs * 2
        return self.turns[:-keep], self.turns[-keep:]

    def apply_summary(self, new_summary: str) -> None:
        """Replace old verbatim turns with ``new_summary`` and keep only the
        most recent ``keep_recent_pairs`` pairs verbatim."""
        _, recent = self._fold_candidates()
        self.summary = new_summary.strip()
        self.turns = recent
