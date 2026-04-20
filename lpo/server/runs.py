"""Live run management: spawns engine tasks, brokers gate signals, pushes
WebSocket events.

There is one global :class:`RunManager` per FastAPI app. It owns:

* a dict of ``run_id -> LiveRun`` records for every optimization triggered via
  the UI (both active and recently completed),
* an asyncio task per active run driving the engine,
* a per-run broadcast fanout (subscriber list of ``asyncio.Queue``) that
  every connected WebSocket client consumes,
* a per-slug signal future that the engine's user-gate awaits; the
  ``submit_signal`` endpoint resolves it.

Runs persist everything through the filesystem as the engine already does.
The server-side in-memory record is just for *live* viewing — once a run
finishes, its disk artifacts are the source of truth and the REST readers in
:mod:`lpo.server.tasks` take over.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lpo.core.cost import CostTracker
from lpo.core.engine import GateDecision, RatchetEngine, UserGate
from lpo.core.history import IterationRecord
from lpo.core.multi_engine import UNIFIED_SLUG, run_multi, validate_runtime
from lpo.core.target_factory import build_target_context
from lpo.core.task import TaskBundle
from lpo.server.schemas import (
    IterationSummary,
    LiveRunInfo,
    StartRunRequest,
    WsEvent,
)
from lpo.server.tasks import resolve_task_path

log = logging.getLogger("lpo.server.runs")


# ---------------------------------------------------------------------------
# Helpers — shared with tasks.py mapping logic
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _summary_from_record(rec: IterationRecord) -> IterationSummary:
    """Shrink the full in-memory IterationRecord to the flat summary the UI renders."""
    per_model: dict[str, float] | None = None
    # Strategy-C records smuggle per-model aggregates into ``notes``; the
    # disk reader re-reads them from scores.json, but for live push we parse
    # the outputs list (each row carries ``model_slug``).
    if any("model_slug" in row for row in rec.outputs):
        aggs: dict[str, list[float]] = {}
        for row in rec.outputs:
            slug = row.get("model_slug")
            score = row.get("score")
            if slug is None or score is None:
                continue
            aggs.setdefault(slug, []).append(float(score))
        if aggs:
            per_model = {slug: sum(vals) / len(vals) for slug, vals in aggs.items()}

    return IterationSummary(
        index=rec.index,
        aggregate_score=rec.aggregate_score,
        decision=rec.decision,
        delta=rec.delta,
        cost_usd=rec.cost_usd,
        timestamp=rec.timestamp,
        failed_ids=list(rec.failed_ids),
        per_scenario={k: float(v) for k, v in rec.per_scenario.items()},
        per_model=per_model,
    )


# ---------------------------------------------------------------------------
# LiveRun record
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class LiveRun:
    run_id: str
    task_name: str
    task_path: Path
    strategy: str
    slugs: list[str]
    mutator: str
    initial_mode: str

    status: str = "starting"
    started_at: str = dataclasses.field(default_factory=_now_iso)
    finished_at: str | None = None
    error: str | None = None
    current_mode: str = "autonomous"

    # Per-slug latest published iteration index — used so reconnects can
    # include the right hello snapshot.
    latest_iterations: dict[str, int] = dataclasses.field(default_factory=dict)

    # Subscriber queues for WebSocket fanout.
    subscribers: list[asyncio.Queue] = dataclasses.field(default_factory=list)

    # Per-slug pending gate signal future. Created on-demand when the gate
    # is about to block, resolved by submit_signal.
    pending_signals: dict[str, asyncio.Future] = dataclasses.field(default_factory=dict)

    # The driver asyncio.Task so we can cancel it.
    driver: asyncio.Task | None = None

    # Most recently seen iteration per slug (for hello on connect).
    last_iteration: dict[str, IterationSummary] = dataclasses.field(default_factory=dict)

    def info(self) -> LiveRunInfo:
        return LiveRunInfo(
            run_id=self.run_id,
            task_name=self.task_name,
            strategy=self.strategy,
            status=self.status,  # type: ignore[arg-type]
            started_at=self.started_at,
            finished_at=self.finished_at,
            error=self.error,
            current_mode=self.current_mode,
            slugs=list(self.slugs),
            latest_iterations=dict(self.latest_iterations),
        )


# ---------------------------------------------------------------------------
# RunManager
# ---------------------------------------------------------------------------


class RunManager:
    """Owns every UI-initiated run. Lifetime: the FastAPI app."""

    def __init__(self, tasks_root: Path) -> None:
        self.tasks_root = Path(tasks_root)
        self.runs: dict[str, LiveRun] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_runs(self) -> list[LiveRunInfo]:
        return [r.info() for r in self.runs.values()]

    def get_run(self, run_id: str) -> LiveRun:
        if run_id not in self.runs:
            raise KeyError(run_id)
        return self.runs[run_id]

    async def start(self, req: StartRunRequest) -> LiveRun:
        task_path = resolve_task_path(self.tasks_root, req.task_name)
        task = TaskBundle.load(task_path)
        validate_runtime(task.config)

        run = LiveRun(
            run_id=uuid.uuid4().hex[:12],
            task_name=task.config.task_name,
            task_path=task_path,
            strategy=task.config.target_strategy,
            slugs=(
                [UNIFIED_SLUG]
                if task.config.target_strategy == "unified_portable"
                else [m.slug for m in task.config.target_models]
            ),
            mutator=req.mutator,
            initial_mode=(req.initial_mode or task.config.mode),
            current_mode=(req.initial_mode or task.config.mode),
        )
        self.runs[run.run_id] = run

        run.driver = asyncio.create_task(
            self._drive(run, task, fresh=req.fresh),
            name=f"lpo-run-{run.run_id}",
        )
        return run

    async def stop(self, run_id: str) -> None:
        run = self.get_run(run_id)
        # Resolve any blocking gate with a stop decision; engine breaks next.
        for fut in run.pending_signals.values():
            if not fut.done():
                fut.set_result(GateDecision(stop=True))
        if run.driver and not run.driver.done():
            run.driver.cancel()

    async def submit_signal(
        self,
        run_id: str,
        *,
        slug: str | None,
        mode: str | None,
        feedback: str,
        stop: bool,
    ) -> None:
        run = self.get_run(run_id)
        if mode is not None:
            run.current_mode = mode
            await self._broadcast(run, WsEvent(type="mode_changed", data={"mode": mode}))

        if not run.pending_signals:
            # No gate is currently blocked; stash a mode change but nothing
            # else to resolve. Supervised/manual signals require a waiting gate.
            if not stop and not feedback and mode is not None:
                return
            # If the run is not waiting on any signal but the user hit stop,
            # fall through to driver cancellation.
            if stop:
                await self.stop(run_id)
            return

        targets = (
            [run.pending_signals[slug]]
            if slug is not None and slug in run.pending_signals
            else list(run.pending_signals.values())
        )
        decision = GateDecision(stop=stop, feedback=feedback, mode=mode)
        for fut in targets:
            if not fut.done():
                fut.set_result(decision)

    # Subscriber management ----------------------------------------------

    def subscribe(self, run_id: str) -> asyncio.Queue:
        run = self.get_run(run_id)
        q: asyncio.Queue = asyncio.Queue(maxsize=256)
        run.subscribers.append(q)
        return q

    def unsubscribe(self, run_id: str, q: asyncio.Queue) -> None:
        if run_id not in self.runs:
            return
        try:
            self.runs[run_id].subscribers.remove(q)
        except ValueError:
            pass

    def hello_snapshot(self, run: LiveRun) -> WsEvent:
        return WsEvent(
            type="hello",
            data={
                "run": run.info().model_dump(),
                "last_iteration": {
                    slug: it.model_dump() for slug, it in run.last_iteration.items()
                },
            },
        )

    # ------------------------------------------------------------------
    # Engine driver
    # ------------------------------------------------------------------

    async def _drive(self, run: LiveRun, task: TaskBundle, *, fresh: bool) -> None:
        """One coroutine per run. Chooses single vs multi engine, wires the
        iteration callback and gate factory to push WebSocket events."""
        try:
            if fresh:
                self._wipe_run_artifacts(task.root, run.slugs, run.strategy)

            run.status = "running"
            await self._broadcast(run, WsEvent(type="status", data={"status": "running"}))

            cost = CostTracker()

            def iteration_cb(slug: str, rec: IterationRecord) -> None:
                summary = _summary_from_record(rec)
                run.latest_iterations[slug] = rec.index
                run.last_iteration[slug] = summary
                # Fire-and-forget fanout; queues are bounded so a slow
                # subscriber cannot block the engine.
                asyncio.create_task(
                    self._broadcast(
                        run,
                        WsEvent(
                            type="iteration",
                            data={"slug": slug, "iteration": summary.model_dump()},
                        ),
                    )
                )

            def gate_factory(slug: str) -> UserGate:
                return self._build_gate(run, slug)

            if run.strategy == "single":
                await self._run_single_engine(run, task, cost, iteration_cb, gate_factory)
            else:
                result = await run_multi(
                    task,
                    cost=cost,
                    mutator_mode=run.mutator,
                    iteration_callback=iteration_cb,
                    user_gate_factory=gate_factory,
                    initial_mode=run.initial_mode,
                )
                # The CLI path writes the cross-model comparison file after
                # run_multi returns; mirror that here so the UI's Comparison
                # tab is populated as soon as the run ends.
                from lpo.core.comparison import write_comparison_report
                write_comparison_report(task.root, task.config.task_name, result)

            run.status = "done"
            run.finished_at = _now_iso()
            await self._broadcast(run, WsEvent(type="done", data={"status": run.status}))

        except asyncio.CancelledError:
            run.status = "stopped"
            run.finished_at = _now_iso()
            log.info("run %s cancelled", run.run_id)
            await self._broadcast(run, WsEvent(type="done", data={"status": "stopped"}))
        except Exception as e:  # noqa: BLE001
            log.exception("run %s failed", run.run_id)
            run.status = "error"
            run.error = f"{type(e).__name__}: {e}"
            run.finished_at = _now_iso()
            await self._broadcast(
                run,
                WsEvent(type="error", data={"error": run.error}),
            )
            await self._broadcast(run, WsEvent(type="done", data={"status": "error"}))

    async def _run_single_engine(
        self,
        run: LiveRun,
        task: TaskBundle,
        cost: CostTracker,
        iteration_cb,
        gate_factory,
    ) -> None:
        """Strategy A path — one engine, one target."""
        target = task.config.target_models[0]
        ctx = build_target_context(task, target, cost, mutator_mode=run.mutator)
        try:
            engine = RatchetEngine(
                task=task,
                target_cfg=target,
                client=ctx.client,
                scorer=ctx.scorer,
                mutator=ctx.mutator,
                cost_tracker=cost,
                iteration_callback=lambda rec, s=target.slug: iteration_cb(s, rec),
                user_gate=gate_factory(target.slug),
                initial_mode=run.initial_mode,
            )
            await engine.run()
        finally:
            await ctx.aclose()

    # ------------------------------------------------------------------
    # Gate plumbing
    # ------------------------------------------------------------------

    def _build_gate(self, run: LiveRun, slug: str) -> UserGate:
        """Return a user-gate closure for ``slug``. On each invocation the
        gate registers an ``asyncio.Future`` under ``pending_signals[slug]``,
        broadcasts ``awaiting_signal`` so the UI can light up the relevant
        tab, and awaits the future. The UI resolves it via ``submit_signal``.
        """

        async def gate(record: IterationRecord, mode: str) -> GateDecision:
            # Autonomous short-circuits — engine already guards on this but
            # mirror here so the in-flight mode is authoritative.
            if mode == "autonomous":
                return GateDecision()

            fut: asyncio.Future[GateDecision] = asyncio.get_event_loop().create_future()
            run.pending_signals[slug] = fut
            run.status = "awaiting_signal"
            try:
                await self._broadcast(
                    run,
                    WsEvent(
                        type="awaiting_signal",
                        data={
                            "slug": slug,
                            "mode": mode,
                            "iteration": _summary_from_record(record).model_dump(),
                        },
                    ),
                )
                return await fut
            finally:
                run.pending_signals.pop(slug, None)
                run.status = "running"
                await self._broadcast(run, WsEvent(type="status", data={"status": "running"}))

        return gate

    # ------------------------------------------------------------------
    # Broadcast
    # ------------------------------------------------------------------

    async def _broadcast(self, run: LiveRun, event: WsEvent) -> None:
        payload = event.model_dump()
        dead: list[asyncio.Queue] = []
        for q in run.subscribers:
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            try:
                run.subscribers.remove(q)
            except ValueError:
                pass

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _wipe_run_artifacts(task_root: Path, slugs: list[str], strategy: str) -> None:
        """--fresh equivalent. Wipes only the specific slugs involved (plus
        logs + comparison) so parallel runs on sibling tasks are unaffected."""
        import shutil

        for slug in slugs:
            d = task_root / "runs" / slug
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
        for d in (task_root / "logs", task_root / "comparison"):
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
