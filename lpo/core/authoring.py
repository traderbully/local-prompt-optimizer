"""Programmatic task-bundle authoring.

Creates a task bundle on disk (``config.yaml``, ``metric.yaml``, ``eval_set.jsonl``,
``prompt_seed.txt``, ``task.md``) from a plain-language description + example
inputs, and — via :func:`generate_gold_standard` — fills in the matching
``gold_standard.jsonl`` using the Anthropic Gold Standard Source.

Used by both the MCP server (:mod:`lpo.server.mcp_server`) and by any CLI
one-shot scripts that need to scaffold bundles. Keeping authoring in its own
module (not the CLI) lets the tests exercise it directly without spawning a
subprocess or monkey-patching Typer.

See `LPO_SDP.md` §3.2 (Task specification), §4.2 (Evaluation Set), §4.8
(Invocation interfaces).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import yaml

from lpo.config.schema import load_eval_set, load_run_config, load_metric_config
from lpo.core.cost import CostTracker
from lpo.core.task import TaskBundle


log = logging.getLogger("lpo.core.authoring")


# ---------------------------------------------------------------------------
# Task creation
# ---------------------------------------------------------------------------


SLUG_RE = re.compile(r"^[A-Za-z0-9_\-]+$")
_NAME_SAFE = re.compile(r"[^A-Za-z0-9_\-]+")


def _safe_dirname(name: str) -> str:
    """Slugify a user-supplied task name into a filesystem-safe directory name."""
    slug = _NAME_SAFE.sub("_", name.strip()).strip("_")
    if not slug:
        raise ValueError(f"Task name {name!r} slugifies to empty; choose a different name.")
    return slug


@dataclass
class TargetSpec:
    """Minimal target-model spec accepted by :func:`create_task_bundle`. The
    full TargetModelConfig has many provider-specific knobs; the MCP surface
    exposes only the common subset (slug/provider/model_id/base_url). Extra
    keys are accepted and passed through verbatim for power users."""

    slug: str
    provider: str = "lmstudio"
    model_id: str = "local-model"
    base_url: str = "http://localhost:1234/v1"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_config(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "slug": self.slug,
            "provider": self.provider,
            "model_id": self.model_id,
            "base_url": self.base_url,
        }
        out.update(self.extra)
        return out


DEFAULT_SEED_PROMPT = (
    "You are a helpful assistant. Follow the task instructions carefully and "
    "produce the expected output format exactly.\n"
)


def _build_metric_yaml(
    *,
    output_type: str,
    task_description: str,
    judge_model: str,
    required_json_fields: list[str] | None,
) -> str:
    """Pick a sensible starter metric.

    * ``json`` output — deterministic rule set (``is_valid_json`` +, when the
      caller named the required fields, ``has_keys`` + ``exact_match_against_gold``).
      Deterministic metrics need no judge, so the bundle can be scored offline.
    * All other output types — single-criterion rubric using the overseer
      model as judge. The Overseer can evolve this via metric_evolution.
    """
    if output_type == "json":
        rules: list[dict[str, Any]] = [
            {"name": "json_valid", "weight": 30, "check": "is_valid_json"},
        ]
        if required_json_fields:
            rules.append(
                {
                    "name": "required_fields_present",
                    "weight": 40,
                    "check": "has_keys",
                    "params": list(required_json_fields),
                }
            )
            rules.append(
                {
                    "name": "field_exact_match",
                    "weight": 30,
                    "check": "exact_match_against_gold",
                }
            )
        else:
            # Without a known schema the best we can do deterministically is
            # reward valid JSON + gold match.
            rules.append(
                {
                    "name": "exact_match_against_gold",
                    "weight": 70,
                    "check": "exact_match_against_gold",
                }
            )
        return yaml.safe_dump({"type": "deterministic", "rules": rules}, sort_keys=False)

    # Default rubric for text / structured / image outputs.
    return yaml.safe_dump(
        {
            "type": "rubric",
            "judge_model": judge_model,
            "criteria": [
                {
                    "name": "overall_quality",
                    "weight": 100,
                    "description": (
                        "Rate how well the output matches the task description and the "
                        "gold-standard answer. Task: "
                        + task_description.strip().replace("\n", " ")[:400]
                    ),
                    "anchors": {
                        0: "Unusable — wrong format or off-topic.",
                        50: "Partially correct but missing important aspects.",
                        100: "Matches the gold answer in meaning and format.",
                    },
                }
            ],
        },
        sort_keys=False,
    )


def create_task_bundle(
    tasks_root: Path,
    *,
    name: str,
    task_description: str,
    example_inputs: list[Any],
    output_type: Literal["text", "json", "image", "structured"] = "text",
    targets: list[TargetSpec] | None = None,
    strategy: Literal["single", "parallel_independent", "unified_portable"] = "single",
    mode: Literal["autonomous", "supervised", "manual", "visual"] = "autonomous",
    required_json_fields: list[str] | None = None,
    seed_prompt: str | None = None,
    judge_model: str = "claude-opus-4-7",
    overseer_model: str = "claude-opus-4-7",
    scenario_tags: list[str | None] | None = None,
    overwrite: bool = False,
) -> Path:
    """Scaffold a complete task bundle on disk and return the resulting
    directory.

    The bundle is fully loadable by :class:`TaskBundle.load` — every file the
    loader requires (``config.yaml``, ``metric.yaml``, ``eval_set.jsonl``) is
    written. ``gold_standard.jsonl`` is left absent until
    :func:`generate_gold_standard` populates it; downstream consumers handle
    the empty-gold case (deterministic scoring on non-gold rules still works).

    Raises :class:`ValueError` on validation problems and :class:`FileExistsError`
    when a bundle under ``name`` already exists and ``overwrite=False``.
    """
    if not name or not name.strip():
        raise ValueError("name is required")
    if not task_description or not task_description.strip():
        raise ValueError("task_description is required")
    if not example_inputs:
        raise ValueError("example_inputs must contain at least one input")

    safe = _safe_dirname(name)
    task_path = tasks_root / safe
    if task_path.exists():
        if not overwrite:
            raise FileExistsError(f"Task bundle already exists: {task_path}")
    tasks_root.mkdir(parents=True, exist_ok=True)
    task_path.mkdir(parents=True, exist_ok=True)

    # --- Resolve strategy vs. target count ---------------------------------
    if targets is None:
        targets = [TargetSpec(slug="local-target")]
    if strategy == "single" and len(targets) != 1:
        raise ValueError("strategy=single requires exactly one target")
    if strategy in ("parallel_independent", "unified_portable") and len(targets) < 2:
        raise ValueError(f"strategy={strategy} requires at least two targets")

    # --- config.yaml --------------------------------------------------------
    cfg: dict[str, Any] = {
        "task_name": name.strip(),
        "mode": mode,
        "target_strategy": strategy,
        "target_models": [t.to_config() for t in targets],
        "overseer_model": {
            "provider": "anthropic",
            "model_id": overseer_model,
            "api_key_env": "ANTHROPIC_API_KEY",
        },
        "stop_conditions": {
            "target_score": 95.0,
            "max_iterations": 20,
            "plateau_patience": 5,
            "cost_cap_usd": 5.0,
        },
        "metric_evolution": {"enabled": True, "require_user_approval": True},
        "output_type": output_type,
        "eval_concurrency": 4,
    }
    (task_path / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    # --- metric.yaml --------------------------------------------------------
    (task_path / "metric.yaml").write_text(
        _build_metric_yaml(
            output_type=output_type,
            task_description=task_description,
            judge_model=judge_model,
            required_json_fields=required_json_fields,
        ),
        encoding="utf-8",
    )

    # --- eval_set.jsonl -----------------------------------------------------
    if scenario_tags is not None and len(scenario_tags) != len(example_inputs):
        raise ValueError("scenario_tags length must match example_inputs length")
    eval_lines: list[str] = []
    for i, inp in enumerate(example_inputs, start=1):
        rec: dict[str, Any] = {"id": f"ex{i:03d}", "input": inp}
        if scenario_tags is not None and scenario_tags[i - 1]:
            rec["scenario"] = scenario_tags[i - 1]
        eval_lines.append(json.dumps(rec, ensure_ascii=False))
    (task_path / "eval_set.jsonl").write_text("\n".join(eval_lines) + "\n", encoding="utf-8")

    # --- prompt_seed.txt + task.md -----------------------------------------
    (task_path / "prompt_seed.txt").write_text(
        (seed_prompt or DEFAULT_SEED_PROMPT).rstrip() + "\n",
        encoding="utf-8",
    )
    (task_path / "task.md").write_text(
        f"# {name.strip()}\n\n{task_description.strip()}\n",
        encoding="utf-8",
    )

    # --- Validate by round-tripping through the real loaders ---------------
    # Fail loudly now rather than producing a broken bundle.
    try:
        load_run_config(task_path / "config.yaml")
        load_metric_config(task_path / "metric.yaml")
        load_eval_set(task_path / "eval_set.jsonl")
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"Generated bundle failed validation: {e}") from e

    log.info("Created task bundle at %s", task_path)
    return task_path


# ---------------------------------------------------------------------------
# Gold standard generation
# ---------------------------------------------------------------------------


class _GoldSource(Protocol):
    """Narrow interface around :class:`AnthropicClient` so tests can inject
    a deterministic replacement without monkey-patching the real client."""

    async def complete(self, *, system: str, messages: list[Any], **kw: Any) -> Any: ...
    async def aclose(self) -> None: ...


_GOLD_SYSTEM = (
    "You are the Gold Standard Source for an automated prompt-optimization "
    "system. You will be shown a task description and one example input. "
    "Produce the single ideal output that a perfect downstream model should "
    "emit for that input. Respond with only the output itself — no preamble, "
    "no commentary, no markdown fences."
)


def _format_input_for_gold(inp: Any) -> str:
    if isinstance(inp, str):
        return inp
    try:
        return json.dumps(inp, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        return str(inp)


async def generate_gold_standard(
    task_path: Path,
    *,
    model_id: str | None = None,
    cost_tracker: CostTracker | None = None,
    client: _GoldSource | None = None,
    overwrite: bool = False,
) -> int:
    """Populate ``<task_path>/gold_standard.jsonl`` by asking the Gold Standard
    Source to produce an ideal output for every record in ``eval_set.jsonl``.

    Returns the number of records written.

    * ``client`` — inject a pre-built source (used by tests). When ``None``,
      a real :class:`AnthropicClient` is constructed with ``model_id`` (or
      the bundle's configured overseer model if omitted) and will be closed
      before return.
    * ``overwrite`` — when False and ``gold_standard.jsonl`` already exists,
      this call is a no-op and returns the existing record count.
    """
    task = TaskBundle.load(task_path)
    gold_path = task_path / "gold_standard.jsonl"

    if gold_path.exists() and not overwrite:
        return len(task.gold_standard)

    own_client = False
    if client is None:
        # The Gold Standard source is pluggable via env vars (see
        # lpo.models.gold_source). Defaults preserve the historical behavior
        # of going straight to Anthropic — but operators with OpenRouter
        # credits and no direct Anthropic billing can route through
        # OpenRouter by setting GOLD_STANDARD_PROVIDER=openrouter.
        from lpo.models.gold_source import build_gold_standard_source

        client = build_gold_standard_source(
            model_id=model_id or task.config.overseer_model.model_id,
            cost_tracker=cost_tracker,
        )
        own_client = True

    # Import here too so modules that only need task creation don't drag in
    # the anthropic SDK.
    from lpo.models.anthropic_client import AnthropicMessage

    records: list[dict[str, Any]] = []
    try:
        description = (task.task_md or "").strip() or "No task description was provided."
        for rec in task.eval_records:
            user_text = (
                f"TASK DESCRIPTION:\n{description}\n\n"
                f"INPUT:\n{_format_input_for_gold(rec.input)}\n\n"
                "Produce the ideal output now."
            )
            result = await client.complete(
                system=_GOLD_SYSTEM,
                messages=[AnthropicMessage(role="user", content=user_text)],
                temperature=0.0,
                max_tokens=1024,
            )
            text = (result.text or "").strip()
            # For JSON tasks, try to parse — keep structured value when valid.
            out_value: Any = text
            if task.config.output_type == "json":
                try:
                    out_value = json.loads(text)
                except json.JSONDecodeError:
                    out_value = text
            records.append({"id": rec.id, "output": out_value})
            log.debug("gold[%s] generated (%d chars)", rec.id, len(text))
    finally:
        if own_client:
            await client.aclose()

    gold_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )
    return len(records)
