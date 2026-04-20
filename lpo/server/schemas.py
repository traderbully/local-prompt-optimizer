"""Pydantic models for the UI <-> backend API.

These are deliberately separate from :mod:`lpo.config.schema` — the config
schema models validate on-disk YAML/JSONL and are the authoritative source
for the engine. The API schemas are thin DTOs shaped for the React frontend
(flat, all-optional where reasonable, JSON-serializable without custom
encoders).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Task listing & detail
# ---------------------------------------------------------------------------


class TargetSummary(BaseModel):
    slug: str
    provider: str
    model_id: str


class TaskSummary(BaseModel):
    """One row in the task browser."""

    name: str
    path: str
    strategy: str  # single | parallel_independent | unified_portable
    mode: str
    output_type: str
    targets: list[TargetSummary]
    has_runs: bool
    has_comparison: bool


class TaskDetail(BaseModel):
    """Everything the UI needs to render a task page without another round-trip."""

    summary: TaskSummary
    task_md: str
    seed_prompt: str
    config_yaml: str  # raw YAML for the config editor
    metric_yaml: str  # raw YAML for the metric editor
    eval_records: list[dict[str, Any]]  # pass-through; UI may render flexibly
    gold_count: int
    metric_type: str  # deterministic | rubric | conversational


# ---------------------------------------------------------------------------
# Run / iteration state
# ---------------------------------------------------------------------------


class IterationSummary(BaseModel):
    index: int
    aggregate_score: float
    decision: str
    delta: float
    cost_usd: float
    timestamp: str
    failed_ids: list[str] = Field(default_factory=list)
    per_scenario: dict[str, float] = Field(default_factory=dict)
    # Strategy-C-only: per-model breakdown pulled from scores.json.
    per_model: dict[str, float] | None = None


class EvalOutputRow(BaseModel):
    """One row from outputs.jsonl. Shape is intentionally open."""

    id: str
    input: Any
    output: Any
    score: float | None = None
    rationale: str | None = None
    scenario: str | None = None
    # Strategy-C-only: which model produced this row.
    model_slug: str | None = None

    class Config:
        extra = "allow"


class IterationDetail(BaseModel):
    summary: IterationSummary
    prompt: str
    outputs: list[EvalOutputRow]
    overseer_analysis_md: str | None = None
    scores_full: dict[str, Any] = Field(default_factory=dict)
    decision_full: dict[str, Any] = Field(default_factory=dict)


class RunState(BaseModel):
    """Per-model run state, reconstructed from the filesystem."""

    slug: str
    exists: bool
    best_score: float | None = None
    best_prompt: str | None = None
    iteration_count: int = 0
    latest_iteration: int | None = None
    iterations: list[IterationSummary] = Field(default_factory=list)
    winner_ready: bool = False


class ComparisonView(BaseModel):
    """Matches the on-disk comparison/summary.json shape, plus the rendered md."""

    present: bool
    summary: dict[str, Any] | None = None
    report_md: str | None = None


# ---------------------------------------------------------------------------
# Live runs (WebSocket + control)
# ---------------------------------------------------------------------------


class StartRunRequest(BaseModel):
    task_name: str
    mutator: Literal["auto", "overseer", "null"] = "auto"
    fresh: bool = False
    initial_mode: Literal["autonomous", "supervised", "manual", "visual"] | None = None


class StartRunResponse(BaseModel):
    run_id: str
    task_name: str
    strategy: str
    slugs: list[str]


class LiveRunInfo(BaseModel):
    run_id: str
    task_name: str
    strategy: str
    status: Literal["starting", "running", "awaiting_signal", "done", "error", "stopped"]
    started_at: str
    finished_at: str | None = None
    error: str | None = None
    current_mode: str
    slugs: list[str]
    # Per-slug latest published iteration index (for reconnect/catch-up).
    latest_iterations: dict[str, int] = Field(default_factory=dict)


class SignalRequest(BaseModel):
    """Payload for ``POST /api/runs/{run_id}/signal``.

    At least one of ``feedback``, ``mode``, ``stop`` should be set; the
    engine applies them atomically on the next gate check. ``slug`` targets
    a specific model in Strategy B / C; when omitted the signal is broadcast
    to whichever gate is awaiting.
    """

    slug: str | None = None
    mode: Literal["autonomous", "supervised", "manual", "visual"] | None = None
    feedback: str = ""
    stop: bool = False


# ---------------------------------------------------------------------------
# WebSocket events
# ---------------------------------------------------------------------------


class WsEvent(BaseModel):
    """Envelope for every WebSocket message the server pushes."""

    type: Literal[
        "hello",            # initial state dump on connect
        "iteration",        # new IterationSummary published
        "status",           # run status change
        "awaiting_signal",  # gate is blocked on user input
        "mode_changed",     # current_mode changed (push confirmation)
        "error",            # recoverable run error
        "done",             # terminal (status transitions to done/stopped/error)
    ]
    data: dict[str, Any]
