"""Ratchet engine (single target). See `LPO_SDP.md` §3.2."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Awaitable, Callable

from lpo.config.schema import StopConditions, TargetModelConfig
from lpo.core.cost import CostTracker
from lpo.core.history import IterationRecord, RunPaths, append_jsonl, atomic_write_text
from lpo.core.iteration import IterationRunner
from lpo.core.mutator import MutationProposal, NullMutator, PromptMutator
from lpo.core.task import TaskBundle
from lpo.models.base import ModelClient
from lpo.scoring.base import Scorer

log = logging.getLogger("lpo.engine")


class StopReason(str, Enum):
    TARGET_SCORE = "target_score_reached"
    MAX_ITERATIONS = "max_iterations"
    PLATEAU = "plateau"
    USER_STOP = "user_stop"
    COST_CAP = "cost_cap"
    MUTATOR_NOOP = "mutator_returned_same_prompt"


@dataclass
class EngineResult:
    best_prompt: str
    best_score: float
    iterations: list[IterationRecord] = field(default_factory=list)
    stop_reason: StopReason = StopReason.MAX_ITERATIONS
    total_cost_usd: float = 0.0


IterationCallback = Callable[[IterationRecord], None]


@dataclass
class GateDecision:
    """Result of a user-gate callback.

    Returned by the UI's WebSocket handler (or any caller driving the engine
    in Supervised / Manual mode). The engine inspects each field after every
    iteration and reacts accordingly:

    * ``stop=True`` → break with :attr:`StopReason.USER_STOP`.
    * ``feedback`` → non-empty string is passed into the next mutator's
      ``user_feedback=`` argument; the Overseer surfaces it in its next turn.
    * ``mode`` → when set, the engine's current mode is swapped in-place so
      subsequent gate invocations see the new mode. (Autonomous gates are
      no-ops, so switching *to* Autonomous effectively disables gating for
      the rest of the run.)
    """

    stop: bool = False
    feedback: str = ""
    mode: str | None = None  # "autonomous" | "supervised" | "manual" | "visual"


UserGate = Callable[[IterationRecord, str], Awaitable[GateDecision]]
"""(iteration_record, current_mode) -> GateDecision. Mode is passed so the
gate implementation can short-circuit in Autonomous without allocating a
decision object."""


class RatchetEngine:
    """Single-target optimization loop.

    The engine is mode-agnostic: modes (autonomous / supervised / manual / visual)
    are implemented by composing different ``PromptMutator`` and
    ``iteration_callback`` values. Stage 1 wires ``NullMutator`` so the loop
    terminates after the seed prompt is evaluated once.
    """

    def __init__(
        self,
        *,
        task: TaskBundle,
        target_cfg: TargetModelConfig,
        client: ModelClient,
        scorer: Scorer,
        mutator: PromptMutator | None = None,
        stop_conditions: StopConditions | None = None,
        iteration_callback: IterationCallback | None = None,
        cost_tracker: CostTracker | None = None,
        user_gate: UserGate | None = None,
        initial_mode: str | None = None,
    ) -> None:
        self.task = task
        self.target_cfg = target_cfg
        self.client = client
        self.scorer = scorer
        self.mutator: PromptMutator = mutator or NullMutator()
        self.stop = stop_conditions or task.config.stop_conditions
        self.on_iteration = iteration_callback
        self.cost = cost_tracker or CostTracker()
        self.user_gate = user_gate
        # Modes "visual" and "autonomous" do not pause the loop. Supervised
        # and Manual do. The gate callback decides; this field is just the
        # engine's current opinion which the gate can switch.
        self.mode = (initial_mode or task.config.mode or "autonomous").lower()

        self.paths = RunPaths(task_root=task.root, model_slug=target_cfg.slug)
        self.runner = IterationRunner(
            client=client,
            target_cfg=target_cfg,
            scorer=scorer,
            concurrency=task.config.eval_concurrency,
        )

        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True

    async def run(self) -> EngineResult:
        self.paths.ensure()

        current_prompt = self._initial_prompt()
        atomic_write_text(self.paths.current_prompt, current_prompt)

        best_prompt = current_prompt
        best_score = float("-inf")
        history: list[IterationRecord] = []
        stop_reason = StopReason.MAX_ITERATIONS
        iter_since_improve = 0

        for i in range(1, self.stop.max_iterations + 1):
            if self._stop_requested:
                stop_reason = StopReason.USER_STOP
                break

            result = await self.runner.run(
                iteration_index=i,
                prompt=current_prompt,
                eval_records=self.task.eval_records,
                gold_standard=self.task.gold_standard,
                task_name=self.task.config.task_name,
            )
            record = result.record

            delta = record.aggregate_score - (best_score if best_score != float("-inf") else 0.0)
            if record.aggregate_score > best_score:
                record.decision = "accepted" if history else "initial"
                record.delta = delta
                best_prompt = current_prompt
                best_score = record.aggregate_score
                atomic_write_text(self.paths.best_prompt, best_prompt)
                iter_since_improve = 0
            else:
                record.decision = "rejected"
                record.delta = record.aggregate_score - best_score
                iter_since_improve += 1
                # Revert working prompt to best
                current_prompt = best_prompt
                atomic_write_text(self.paths.current_prompt, current_prompt)

            record.cost_usd = self.cost.total_usd
            record.write(self.paths)
            append_jsonl(
                self.paths.log_file,
                {
                    "iter": i,
                    "score": record.aggregate_score,
                    "decision": record.decision,
                    "delta": record.delta,
                    "per_scenario": record.per_scenario,
                    "failed_ids": record.failed_ids,
                    "timings": record.timings,
                    "cost_usd": record.cost_usd,
                },
            )
            history.append(record)
            if self.on_iteration is not None:
                try:
                    self.on_iteration(record)
                except Exception:  # pragma: no cover - UI callback safety
                    log.exception("iteration_callback raised")

            log.info(
                "iter=%d score=%.2f decision=%s delta=%+.2f best=%.2f cost=$%.4f",
                i,
                record.aggregate_score,
                record.decision,
                record.delta,
                best_score,
                self.cost.total_usd,
            )

            if best_score >= self.stop.target_score:
                stop_reason = StopReason.TARGET_SCORE
                break
            if iter_since_improve >= self.stop.plateau_patience:
                stop_reason = StopReason.PLATEAU
                break
            if self.cost.over_cap(self.stop.cost_cap_usd):
                stop_reason = StopReason.COST_CAP
                break
            if i >= self.stop.max_iterations:
                stop_reason = StopReason.MAX_ITERATIONS
                break

            # User gate — fires only when a callback is attached AND the
            # engine is in a mode that cares. Autonomous short-circuits so
            # headless runs have zero overhead.
            pending_feedback = ""
            if self.user_gate is not None and self.mode != "autonomous":
                decision = await self.user_gate(record, self.mode)
                if decision.mode is not None and decision.mode != self.mode:
                    log.info("mode switch: %s -> %s", self.mode, decision.mode)
                    self.mode = decision.mode.lower()
                if decision.stop:
                    stop_reason = StopReason.USER_STOP
                    break
                pending_feedback = decision.feedback

            proposal = await self.mutator.propose(
                current_prompt=current_prompt,
                best_prompt=best_prompt,
                history=history,
                user_feedback=pending_feedback,
            )
            _persist_overseer_analysis(self.paths.iteration_dir(i), proposal)

            if self.cost.over_cap(self.stop.cost_cap_usd):
                stop_reason = StopReason.COST_CAP
                break
            if proposal.new_prompt.strip() == current_prompt.strip():
                stop_reason = StopReason.MUTATOR_NOOP
                break
            current_prompt = proposal.new_prompt
            atomic_write_text(self.paths.current_prompt, current_prompt)

        # Write winner artifacts
        self.paths.winner_root.mkdir(parents=True, exist_ok=True)
        atomic_write_text(self.paths.winner_root / "prompt.txt", best_prompt)
        atomic_write_text(
            self.paths.winner_root / "report.md",
            _render_report(
                task_name=self.task.config.task_name,
                model_slug=self.target_cfg.slug,
                best_score=best_score if best_score != float("-inf") else 0.0,
                iterations=len(history),
                stop_reason=stop_reason,
                best_prompt=best_prompt,
            ),
        )

        return EngineResult(
            best_prompt=best_prompt,
            best_score=best_score if best_score != float("-inf") else 0.0,
            iterations=history,
            stop_reason=stop_reason,
            total_cost_usd=self.cost.total_usd,
        )

    # ------------------------------------------------------------------
    def _initial_prompt(self) -> str:
        # Resume semantics: prefer existing best, then current, then seed.
        for p in (self.paths.best_prompt, self.paths.current_prompt):
            if p.exists():
                text = p.read_text(encoding="utf-8")
                if text.strip():
                    return text
        if self.task.seed_prompt.strip():
            return self.task.seed_prompt
        raise FileNotFoundError(
            f"No prompt found. Create '{self.task.root / 'prompt_seed.txt'}' or place a "
            f"prompt at '{self.paths.current_prompt}'."
        )


def _persist_overseer_analysis(iteration_dir: Path, proposal: MutationProposal) -> None:
    """Write the overseer's analysis + hypothesis alongside the iteration it analyzed."""
    if not (proposal.analysis or proposal.rationale):
        return
    body = (
        "# Overseer analysis\n\n"
        f"{proposal.analysis.strip() or '(no analysis provided)'}\n\n"
        "## Hypothesis for next iteration\n\n"
        f"{proposal.rationale.strip() or '(no hypothesis provided)'}\n\n"
        "## Proposed next prompt\n\n"
        f"```\n{proposal.new_prompt.rstrip()}\n```\n"
    )
    atomic_write_text(iteration_dir / "overseer_analysis.md", body)


def _render_report(
    *,
    task_name: str,
    model_slug: str,
    best_score: float,
    iterations: int,
    stop_reason: StopReason,
    best_prompt: str,
) -> str:
    return (
        f"# Run report — {task_name} / {model_slug}\n\n"
        f"- **Best score:** {best_score:.2f}\n"
        f"- **Iterations:** {iterations}\n"
        f"- **Stop reason:** {stop_reason.value}\n\n"
        f"## Winning prompt\n\n```\n{best_prompt}\n```\n"
    )
