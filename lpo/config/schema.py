"""Pydantic schema for task bundle configuration files.

See `LPO_SDP.md` §4.1, §4.2, §4.4, §4.7.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SLUG_RE = re.compile(r"^[a-z0-9_-]+$")


# ---------------------------------------------------------------------------
# Eval / gold records
# ---------------------------------------------------------------------------


class EvalRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    input: Any
    scenario: str | None = None
    weight: float = 1.0


class GoldRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    output: Any


# ---------------------------------------------------------------------------
# Run config (config.yaml)
# ---------------------------------------------------------------------------


class TargetModelConfig(BaseModel):
    slug: str
    provider: Literal["lmstudio", "openai_compatible", "openrouter", "stub"] = "lmstudio"
    base_url: str = "http://localhost:1234/v1"
    model_id: str
    api_key_env: str | None = None  # env var name for Authorization; optional for LM Studio
    temperature: float = 0.2
    # Raised from 2048 to 4096 on Apr 21 after the windows_file_ops_reliability
    # / gemma-4-26b-local forensic investigation: the 2048 default caused
    # silent completion-token exhaustion on reasoning-heavy tasks (three
    # scenarios scored 0 because the model consumed the entire budget on
    # hidden CoT before emitting a single JSON token). Bumping to 4096
    # produced +27 aggregate points on that task with zero regression
    # elsewhere. The cost of the higher ceiling is bounded (providers
    # still only bill actual completion tokens) so the conservative
    # move is to make the new floor the default.
    max_tokens: int = 4096
    seed_policy: Literal["fixed", "fixed_per_example", "unlocked"] = "fixed_per_example"
    fixed_seed: int = 0
    weight: float | None = None

    # OpenRouter-specific optional headers (for OpenRouter's attribution /
    # analytics). Ignored by other providers.
    http_referer: str | None = None
    x_title: str | None = None

    # Stub-provider knobs. Only consulted when provider="stub". See
    # `lpo.models.stub` for semantics.
    stub_mode: Literal["fixed", "echo", "truncate", "prefix"] | None = None
    stub_fixed_text: str | None = None
    stub_prefix: str | None = None
    stub_truncate_chars: int | None = None

    @model_validator(mode="after")
    def _provider_specific_checks(self) -> "TargetModelConfig":
        if self.provider == "openrouter":
            # OpenRouter model_ids are always of the form ``org/model`` (e.g.
            # ``google/gemma-4-31b-it``). Reject an obviously-wrong shape
            # up-front rather than failing at call time.
            if "/" not in self.model_id:
                raise ValueError(
                    f"provider=openrouter requires model_id of the form 'org/model'; got {self.model_id!r}"
                )
        if self.provider == "stub" and self.stub_mode is None:
            # Fall through with default 'fixed' for convenience.
            object.__setattr__(self, "stub_mode", "fixed")
        return self

    @field_validator("slug")
    @classmethod
    def _slug_safe(cls, v: str) -> str:
        if not SLUG_RE.match(v):
            raise ValueError(f"slug {v!r} must match {SLUG_RE.pattern}")
        return v


class OverseerModelConfig(BaseModel):
    provider: Literal["anthropic"] = "anthropic"
    model_id: str = "claude-opus-4-5"
    api_key_env: str = "ANTHROPIC_API_KEY"


class StopConditions(BaseModel):
    target_score: float = 95.0
    max_iterations: int = 50
    plateau_patience: int = 5
    cost_cap_usd: float = 10.0


class MetricEvolution(BaseModel):
    enabled: bool = True
    check_every_n_iterations: int = 5
    require_user_approval: bool = True


class RunConfig(BaseModel):
    task_name: str
    mode: Literal["autonomous", "supervised", "manual", "visual"] = "autonomous"

    target_strategy: Literal["single", "parallel_independent", "unified_portable"] = "single"
    target_models: list[TargetModelConfig]
    unified_aggregation: Literal["min", "mean", "weighted_mean"] = "min"
    parallel_execution: Literal["sequential", "parallel"] = "sequential"

    overseer_model: OverseerModelConfig = Field(default_factory=OverseerModelConfig)
    stop_conditions: StopConditions = Field(default_factory=StopConditions)
    metric_evolution: MetricEvolution = Field(default_factory=MetricEvolution)
    output_type: Literal["text", "json", "image", "structured"] = "text"

    eval_concurrency: int = 4

    # Stage 8 — postmortem tuning. Opt-in at invocation time (no enabled
    # flag here per Apr 21 review); these fields govern thresholds, budget,
    # and confidence bars *if* the operator invokes the postmortem phase.
    # The nested model is defined in lpo.postmortem.schemas, which has no
    # back-reference to this module, so a direct import is safe.
    postmortem: "PostmortemConfig" = Field(  # noqa: F821  (resolved via model_rebuild below)
        default_factory=lambda: _default_postmortem_config(),
        description=(
            "Postmortem phase tuning (Stage 8). Defaults applied when omitted. "
            "See lpo.postmortem.schemas.PostmortemConfig."
        ),
    )

    @model_validator(mode="after")
    def _validate_strategy(self) -> "RunConfig":
        n = len(self.target_models)
        if n == 0:
            raise ValueError("target_models must have at least one entry")
        if self.target_strategy == "single" and n != 1:
            raise ValueError("target_strategy=single requires exactly one target_models entry")
        if self.target_strategy in ("parallel_independent", "unified_portable") and n < 2:
            raise ValueError(f"target_strategy={self.target_strategy} requires at least two target_models")
        if (
            self.target_strategy == "unified_portable"
            and self.unified_aggregation == "weighted_mean"
            and any(m.weight is None for m in self.target_models)
        ):
            raise ValueError("weighted_mean aggregation requires every target_model to set 'weight'")
        slugs = [m.slug for m in self.target_models]
        if len(set(slugs)) != len(slugs):
            raise ValueError("target_models slugs must be unique")
        return self


# ---------------------------------------------------------------------------
# Metric config (metric.yaml)
# ---------------------------------------------------------------------------


class DeterministicRule(BaseModel):
    name: str
    weight: float
    check: str
    params: list[Any] | dict[str, Any] | None = None


class DeterministicMetric(BaseModel):
    type: Literal["deterministic"]
    rules: list[DeterministicRule]


class RubricCriterion(BaseModel):
    name: str
    weight: float
    description: str
    anchors: dict[int, str] | None = None


class RubricMetric(BaseModel):
    type: Literal["rubric"]
    judge_model: str
    criteria: list[RubricCriterion]


class ConversationalMetric(BaseModel):
    type: Literal["conversational"]
    overseer_model: str
    stated_goal: str


MetricConfig = DeterministicMetric | RubricMetric | ConversationalMetric


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_run_config(path: Path) -> RunConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return RunConfig.model_validate(data)


def load_metric_config(path: Path) -> MetricConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    kind = data.get("type")
    if kind == "deterministic":
        return DeterministicMetric.model_validate(data)
    if kind == "rubric":
        return RubricMetric.model_validate(data)
    if kind == "conversational":
        return ConversationalMetric.model_validate(data)
    raise ValueError(f"Unknown metric type: {kind!r}")


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{i}: invalid JSON ({e})") from e
    return out


def load_eval_set(path: Path) -> list[EvalRecord]:
    records = [EvalRecord.model_validate(r) for r in _iter_jsonl(path)]
    if len(records) < 1:
        raise ValueError(f"{path}: eval set is empty (minimum 1, recommended >=5)")
    ids = [r.id for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate ids in eval set")
    return records


def load_gold_standard(path: Path) -> dict[str, GoldRecord]:
    if not path.exists():
        return {}
    records = [GoldRecord.model_validate(r) for r in _iter_jsonl(path)]
    return {r.id: r for r in records}


# ---------------------------------------------------------------------------
# Stage 8 postmortem — deferred forward-reference wiring
# ---------------------------------------------------------------------------
#
# RunConfig declares a ``postmortem: "PostmortemConfig"`` field. The concrete
# class lives in lpo.postmortem.schemas (one-way import: that module does not
# import from here). We resolve the forward reference and provide a default
# factory at module load time so that RunConfig is fully usable without the
# caller having to touch lpo.postmortem directly.


def _default_postmortem_config() -> "PostmortemConfig":  # noqa: F821
    """Default-factory for :class:`RunConfig.postmortem`. Defined here
    rather than inline because Pydantic's default_factory is evaluated
    per-instance and we want the import cost paid once."""
    from lpo.postmortem.schemas import PostmortemConfig
    return PostmortemConfig()


# Resolve the forward reference ``"PostmortemConfig"`` on RunConfig. After
# this call, ``RunConfig.model_fields["postmortem"].annotation`` is the
# real class rather than a string, which is what downstream serializers
# (notably the MCP transport layer) require.
from lpo.postmortem.schemas import PostmortemConfig  # noqa: E402  (intentional late import)

RunConfig.model_rebuild()
