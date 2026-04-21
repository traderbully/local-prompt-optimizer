"""MCP stdio server exposing LPO to agentic IDEs (Windsurf).

See `LPO_SDP.md` §4.8 + §7.3.

Architecture
------------

This module has two layers:

1. :class:`LpoMcpHandlers` — a **pure-Python** dispatcher with one async
   method per tool. It takes a ``tasks_root`` on construction and has no
   knowledge of stdio, JSON-RPC, or the mcp SDK. Tests drive the handlers
   directly; there is no need to spin up a subprocess.

2. :func:`build_mcp_server` + :func:`run_stdio` — the thin glue around the
   ``mcp`` SDK that turns the handlers into a registered MCP server and runs
   it over stdio. The CLI entry point (``lpo mcp``) calls :func:`run_stdio`.

Headless contract (SDP §4.8): MCP mode defaults to Autonomous. Manual and
Visual modes are rejected; Supervised is permitted only if the caller is
polling status between iterations.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import dotenv_values, find_dotenv, load_dotenv

from lpo.core.authoring import TargetSpec, create_task_bundle, generate_gold_standard
from lpo.core.comparison import write_comparison_report
from lpo.core.cost import CostTracker
from lpo.core.engine import RatchetEngine
from lpo.core.multi_engine import run_multi, validate_runtime
from lpo.core.target_factory import build_target_context
from lpo.core.task import TaskBundle
from lpo.server.tasks import (
    read_all_tasks,
    read_comparison,
    read_run_state,
    read_task_detail,
    read_winner,
    resolve_task_path,
)

log = logging.getLogger("lpo.server.mcp")


# ---------------------------------------------------------------------------
# Tool metadata
# ---------------------------------------------------------------------------


# Keep the schemas in Python (not duck-typed) so they're both the source of
# truth for the MCP wire format and a checklist for the handlers. Each entry
# is dispatched to a method called ``_tool_<name>``.
TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "lpo_create_task",
        "description": (
            "Create a new task bundle on disk from a task description and a list "
            "of example inputs. Returns the task_id (== directory name). Writes "
            "config.yaml, metric.yaml, eval_set.jsonl, prompt_seed.txt, task.md. "
            "The gold standard is NOT generated — call lpo_generate_gold_standard "
            "next."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Human-readable task name (slugified for the directory)."},
                "task_description": {"type": "string", "description": "Plain-language description of what the model should do."},
                "example_inputs": {
                    "type": "array",
                    "description": "List of representative inputs (strings or JSON objects). >=5 recommended.",
                    "items": {},
                    "minItems": 1,
                },
                "output_type": {
                    "type": "string",
                    "enum": ["text", "json", "image", "structured"],
                    "default": "text",
                },
                "required_json_fields": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "For output_type=json, the fields every output must contain (enables deterministic scoring).",
                },
                "seed_prompt": {"type": "string", "description": "Optional starting prompt. Defaults to a generic assistant preamble."},
                "target_models": {
                    "type": "array",
                    "description": "Targets. When omitted a single lmstudio target is used.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "slug": {"type": "string"},
                            "provider": {"type": "string"},
                            "model_id": {"type": "string"},
                            "base_url": {"type": "string"},
                        },
                        "required": ["slug"],
                    },
                },
                "strategy": {
                    "type": "string",
                    "enum": ["single", "parallel_independent", "unified_portable"],
                    "default": "single",
                },
                "scenario_tags": {
                    "type": "array",
                    "items": {"type": ["string", "null"]},
                    "description": "Optional scenario tag per example_inputs entry.",
                },
                "overwrite": {"type": "boolean", "default": False},
            },
            "required": ["name", "task_description", "example_inputs"],
        },
    },
    {
        "name": "lpo_generate_gold_standard",
        "description": (
            "Call the Gold Standard Source to produce ideal outputs for every "
            "entry in the task's eval_set.jsonl and write gold_standard.jsonl. "
            "Defaults to direct Anthropic (requires ANTHROPIC_API_KEY). Can be "
            "routed through OpenRouter by setting GOLD_STANDARD_PROVIDER="
            "openrouter in .env (see .env.example for all routing vars)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "overwrite": {"type": "boolean", "default": False},
            },
            "required": ["task_id"],
        },
    },
    {
        "name": "lpo_run_optimization",
        "description": (
            "Run the ratchet loop headlessly until stop conditions are met and "
            "return a summary that includes the winning prompt(s). Blocks for "
            "the duration of the optimization. mode must be 'autonomous' "
            "(default) — manual/visual/supervised modes are UI-only. By "
            "default runs every target in config.yaml; pass target_slugs to "
            "restrict to a subset (e.g. LM-Studio-only to avoid paid API "
            "spend) without editing config.yaml."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "mode": {
                    "type": "string",
                    "enum": ["autonomous"],
                    "default": "autonomous",
                },
                "mutator": {
                    "type": "string",
                    "enum": ["auto", "overseer", "null"],
                    "default": "auto",
                },
                "target_slugs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional subset of target slugs from config.yaml to "
                        "run. Omit or pass null to run every configured "
                        "target. Useful for local-only dry runs that skip "
                        "paid providers."
                    ),
                },
                "fresh": {"type": "boolean", "default": False},
                "stop_conditions": {
                    "type": "object",
                    "description": "Optional override for stop conditions in config.yaml.",
                    "properties": {
                        "target_score": {"type": "number"},
                        "max_iterations": {"type": "integer"},
                        "plateau_patience": {"type": "integer"},
                        "cost_cap_usd": {"type": "number"},
                    },
                },
            },
            "required": ["task_id"],
        },
    },
    {
        "name": "lpo_get_status",
        "description": (
            "Return on-disk run status for every target slug of the task: "
            "iteration count, best score, history summary, and per-model status."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"task_id": {"type": "string"}},
            "required": ["task_id"],
        },
    },
    {
        "name": "lpo_get_winner",
        "description": (
            "Return the winning prompt + report for a task. For Strategy A "
            "(single) model_slug is optional. For Strategy B (parallel "
            "independent) model_slug is required to disambiguate. For Strategy "
            "C (unified portable) the shared winner is returned regardless."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "model_slug": {"type": "string"},
            },
            "required": ["task_id"],
        },
    },
    {
        "name": "lpo_get_comparison",
        "description": (
            "Return the cross-model comparison report for parallel_independent "
            "/ unified_portable runs. Returns {present: false} if no comparison "
            "has been written yet."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"task_id": {"type": "string"}},
            "required": ["task_id"],
        },
    },
    {
        "name": "lpo_list_tasks",
        "description": "List every task bundle visible under tasks_root.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "lpo_reload_env",
        "description": (
            "Re-read .env into this MCP process's environment, overriding "
            "any stale values. Call this after editing .env so you do not "
            "need to restart the IDE. Returns the list of env var names "
            "whose value changed plus obfuscated fingerprints (never the "
            "full secret). No task or run state is affected."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "lpo_run_postmortem",
        "description": (
            "Run the Stage 8 postmortem phase on a completed task run: a "
            "frontier-model Analyst reads the full run artifacts, proposes "
            "structured interventions (findings with evidence, interventions "
            "with provenance), and — in autonomous mode — runs a focused "
            "validation retry and auto-commits prompt/seed changes that "
            "clear three AND-semantics thresholds. Metric patches, eval "
            "additions, and model-swap suggestions are always surfaced for "
            "human review per the Apr 21 design review. Opt-in only; not "
            "invoked automatically at the end of lpo_run_optimization. "
            "Uses a separate cost budget (config.yaml: postmortem.cost_cap_usd, "
            "default $2.50). Requires ANTHROPIC_API_KEY."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "slug": {
                    "type": "string",
                    "description": (
                        "Target slug whose run to analyze. Defaults to the "
                        "first target_model in config.yaml."
                    ),
                },
                "mode": {
                    "type": "string",
                    "enum": ["autonomous", "propose_only"],
                    "default": "autonomous",
                    "description": (
                        "autonomous: diagnose -> patch -> retry -> decide; "
                        "may auto-commit prompt/seed changes on threshold. "
                        "propose_only: stop after writing diagnosis + proposal."
                    ),
                },
                "allow_on_cost_cap": {
                    "type": "boolean",
                    "default": False,
                    "description": (
                        "Run the postmortem even when the main ratchet "
                        "terminated on cost_cap. Default False per design "
                        "\u00a79: if the main loop died on budget the operator "
                        "should fix the budget first."
                    ),
                },
            },
            "required": ["task_id"],
        },
    },
]


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


class McpToolError(Exception):
    """Raised by a tool handler; the dispatcher converts it to an MCP error
    payload instead of letting the whole server crash."""


@dataclass
class LpoMcpHandlers:
    """Pure async tool dispatcher. Keeps no long-lived state beyond the
    ``tasks_root`` and a dict of async run futures (for status checks)."""

    tasks_root: Path
    _runs: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Override point for tests that want to replace the real run_multi /
    # RatchetEngine plumbing with a stub. The production server uses the
    # defaults.
    _run_single_cb: Any = None
    _run_multi_cb: Any = None
    _run_postmortem_cb: Any = None  # Stage 8 test seam; see _tool_run_postmortem

    # --- dispatch ---------------------------------------------------------

    async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Dispatch by tool name. Returns a JSON-serializable dict — never
        raises; errors come back as ``{"error": "..."}``."""
        method = getattr(self, f"_tool_{name[4:]}", None) if name.startswith("lpo_") else None
        if method is None:
            return {"error": f"Unknown tool: {name}"}
        try:
            result = await method(arguments or {})
            return result if isinstance(result, dict) else {"result": result}
        except McpToolError as e:
            return {"error": str(e)}
        except FileNotFoundError as e:
            return {"error": f"Not found: {e}"}
        except ValueError as e:
            return {"error": f"Invalid argument: {e}"}
        except Exception as e:  # noqa: BLE001
            log.exception("tool %s crashed", name)
            return {"error": f"Internal error: {e.__class__.__name__}: {e}"}

    # --- helpers ----------------------------------------------------------

    def _resolve(self, task_id: str) -> Path:
        return resolve_task_path(self.tasks_root, task_id)

    # --- tools ------------------------------------------------------------

    async def _tool_create_task(self, args: dict[str, Any]) -> dict[str, Any]:
        targets_raw = args.get("target_models") or []
        targets: list[TargetSpec] | None = None
        if targets_raw:
            targets = []
            for t in targets_raw:
                if not isinstance(t, dict) or "slug" not in t:
                    raise McpToolError("each target_models entry must be an object with 'slug'")
                extra = {
                    k: v
                    for k, v in t.items()
                    if k not in {"slug", "provider", "model_id", "base_url"}
                }
                targets.append(
                    TargetSpec(
                        slug=t["slug"],
                        provider=t.get("provider", "lmstudio"),
                        model_id=t.get("model_id", "local-model"),
                        base_url=t.get("base_url", "http://localhost:1234/v1"),
                        extra=extra,
                    )
                )

        path = create_task_bundle(
            self.tasks_root,
            name=args["name"],
            task_description=args["task_description"],
            example_inputs=list(args["example_inputs"]),
            output_type=args.get("output_type", "text"),
            required_json_fields=args.get("required_json_fields"),
            seed_prompt=args.get("seed_prompt"),
            targets=targets,
            strategy=args.get("strategy", "single"),
            scenario_tags=args.get("scenario_tags"),
            overwrite=bool(args.get("overwrite", False)),
        )
        task = TaskBundle.load(path)
        return {
            "task_id": task.config.task_name,
            "path": str(path),
            "strategy": task.config.target_strategy,
            "n_examples": len(task.eval_records),
            "output_type": task.config.output_type,
            "gold_standard_ready": bool(task.gold_standard),
        }

    async def _tool_generate_gold_standard(self, args: dict[str, Any]) -> dict[str, Any]:
        path = self._resolve(args["task_id"])
        n = await generate_gold_standard(
            path,
            overwrite=bool(args.get("overwrite", False)),
        )
        return {"task_id": args["task_id"], "gold_records_written": n}

    async def _tool_run_optimization(self, args: dict[str, Any]) -> dict[str, Any]:
        task_id = args["task_id"]
        path = self._resolve(task_id)
        task = TaskBundle.load(path)
        validate_runtime(task.config)

        mode = args.get("mode", "autonomous")
        if mode != "autonomous":
            raise McpToolError(
                f"MCP mode={mode!r} not supported; Manual/Visual/Supervised are UI-only. "
                "Use the `lpo ui` command for interactive modes."
            )
        mutator = args.get("mutator", "auto")
        fresh = bool(args.get("fresh", False))

        # Optional per-call target filter. Lets the caller restrict a multi-
        # target task to, e.g., LM-Studio-only without editing config.yaml.
        # We mutate the in-memory TaskBundle only; disk is untouched.
        requested_slugs = args.get("target_slugs")
        if requested_slugs is not None:
            if not isinstance(requested_slugs, list) or not all(
                isinstance(s, str) for s in requested_slugs
            ):
                raise McpToolError(
                    "target_slugs must be an array of strings or omitted."
                )
            if not requested_slugs:
                raise McpToolError(
                    "target_slugs is empty; pass at least one slug or omit the arg."
                )
            available = {m.slug for m in task.config.target_models}
            unknown = [s for s in requested_slugs if s not in available]
            if unknown:
                raise McpToolError(
                    f"Unknown target_slugs {unknown!r}. Configured slugs for "
                    f"task {task_id!r}: {sorted(available)}."
                )
            # Preserve the order declared in config.yaml (deterministic).
            kept = [m for m in task.config.target_models if m.slug in set(requested_slugs)]
            task.config.target_models = kept

        # Apply stop_conditions overrides in memory (do NOT mutate config.yaml).
        override = args.get("stop_conditions") or {}
        if override:
            for k, v in override.items():
                if hasattr(task.config.stop_conditions, k):
                    setattr(task.config.stop_conditions, k, v)

        cost = CostTracker()
        run_id = uuid.uuid4().hex[:12]
        self._runs[run_id] = {"task_id": task_id, "status": "running"}

        # --- Fresh-wipe: scope to selected slugs when target_slugs is set ----
        # Pre-target_slugs, ``fresh=True`` meant "wipe the whole task's runs/
        # comparison/ logs". That's surprising when the caller is running only
        # a subset — they'd lose unrelated slugs' histories as collateral
        # damage. New behavior: when the caller restricted target_slugs, only
        # those slugs' ``runs/<slug>/`` dirs are wiped. ``comparison/`` stays
        # because a prior cross-model report is still meaningful for the
        # slugs the caller is *not* re-running right now.
        if fresh:
            import shutil
            if requested_slugs is not None:
                for slug in {m.slug for m in task.config.target_models}:
                    p = task.root / "runs" / slug
                    if p.exists():
                        shutil.rmtree(p, ignore_errors=True)
            else:
                for p in (task.root / "runs", task.root / "logs", task.root / "comparison"):
                    if p.exists():
                        shutil.rmtree(p, ignore_errors=True)

        # --- Path selection ---------------------------------------------------
        # If exactly one target survives (either the task is natively Strategy A
        # or the caller filtered down to one via target_slugs), we use the
        # single-target runner. Going through the multi path with a 1-element
        # target list produces a "comparison report with one winner" which is
        # vacuous and misleads the operator.
        try:
            if len(task.config.target_models) == 1:
                result = await (self._run_single_cb or _default_run_single)(
                    task, mutator_name=mutator, cost=cost, fresh=False  # wipe already handled above
                )
                summary: dict[str, Any] = {
                    "strategy": "single",
                    "best_score": result.best_score,
                    "iterations": len(result.iterations),
                    "stop_reason": result.stop_reason.value,
                    "total_cost_usd": result.total_cost_usd,
                    "winner": read_winner(path, task.config.target_models[0].slug),
                }
                # Make it obvious when the configured strategy was NOT single
                # but we collapsed because of a target_slugs filter. Downstream
                # tooling can use this flag to suppress "winning model" claims.
                if requested_slugs is not None:
                    summary["subset_run"] = True
                    summary["configured_strategy"] = task.config.target_strategy
                    summary["ran_slugs"] = [m.slug for m in task.config.target_models]
            else:
                multi = await (self._run_multi_cb or run_multi)(
                    task,
                    cost=cost,
                    mutator_mode=mutator,
                )
                # CLI path writes the comparison report; mirror it here.
                write_comparison_report(task.root, task.config.task_name, multi)
                summary = {
                    "strategy": multi.strategy,
                    "total_cost_usd": multi.total_cost_usd,
                    "per_model": [
                        {
                            "slug": r.slug,
                            "best_score": r.best_score,
                            "iterations": r.iterations,
                            "stop_reason": r.stop_reason.value,
                            "cost_usd": r.cost_usd,
                        }
                        for r in multi.per_model
                    ],
                    "comparison": read_comparison(path).model_dump(),
                }
                if requested_slugs is not None:
                    summary["subset_run"] = True
                    summary["ran_slugs"] = [m.slug for m in task.config.target_models]
            self._runs[run_id].update(status="done", summary=summary)
            return {"run_id": run_id, "task_id": task_id, "status": "done", **summary}
        except Exception as e:  # noqa: BLE001
            self._runs[run_id].update(status="error", error=str(e))
            raise

    async def _tool_get_status(self, args: dict[str, Any]) -> dict[str, Any]:
        path = self._resolve(args["task_id"])
        task = TaskBundle.load(path)
        slugs = (
            ["_unified"]
            if task.config.target_strategy == "unified_portable"
            else [m.slug for m in task.config.target_models]
        )
        per_model: dict[str, Any] = {}
        best_score: float | None = None
        iteration = 0
        for slug in slugs:
            st = read_run_state(path, slug)
            per_model[slug] = st.model_dump()
            if st.best_score is not None:
                best_score = st.best_score if best_score is None else max(best_score, st.best_score)
            iteration = max(iteration, st.iteration_count)
        return {
            "task_id": args["task_id"],
            "strategy": task.config.target_strategy,
            "iteration": iteration,
            "best_score": best_score,
            "per_model_status": per_model,
            "history_summary": [
                {
                    "slug": slug,
                    "iterations": per_model[slug].get("iteration_count", 0),
                    "best_score": per_model[slug].get("best_score"),
                    "winner_ready": per_model[slug].get("winner_ready", False),
                }
                for slug in slugs
            ],
        }

    async def _tool_get_winner(self, args: dict[str, Any]) -> dict[str, Any]:
        path = self._resolve(args["task_id"])
        task = TaskBundle.load(path)
        slug = args.get("model_slug")

        if task.config.target_strategy == "single":
            slug = slug or task.config.target_models[0].slug
        elif task.config.target_strategy == "parallel_independent" and not slug:
            raise McpToolError(
                "model_slug is required for strategy=parallel_independent; "
                f"choose one of: {[m.slug for m in task.config.target_models]}"
            )
        elif task.config.target_strategy == "unified_portable":
            slug = "_unified"

        winner = read_winner(path, slug)
        if not winner.get("present"):
            raise McpToolError(
                f"No winner on disk yet for slug={slug!r}. Run lpo_run_optimization first."
            )
        # Augment with the best score if we can read it off the run state.
        state = read_run_state(path, slug)
        return {
            "task_id": args["task_id"],
            "model_slug": slug,
            "prompt": winner["prompt"],
            "report": winner["report_md"],
            "best_score": state.best_score,
            "iterations": state.iteration_count,
        }

    async def _tool_get_comparison(self, args: dict[str, Any]) -> dict[str, Any]:
        path = self._resolve(args["task_id"])
        view = read_comparison(path)
        if not view.present:
            return {
                "task_id": args["task_id"],
                "present": False,
                "message": (
                    "No comparison report on disk. Run a parallel_independent or "
                    "unified_portable optimization first."
                ),
            }
        return {
            "task_id": args["task_id"],
            "present": True,
            "summary": view.summary,
            "report_md": view.report_md,
        }

    async def _tool_list_tasks(self, args: dict[str, Any]) -> dict[str, Any]:
        summaries = read_all_tasks(self.tasks_root)
        return {"tasks": [s.model_dump() for s in summaries]}

    async def _tool_run_postmortem(self, args: dict[str, Any]) -> dict[str, Any]:
        """Stage 8 opt-in postmortem. See lpo.postmortem.runner.run_postmortem.

        The real Anthropic-backed Analyst client is constructed here; tests
        inject a stub via ``_run_postmortem_cb`` on the handler, which
        bypasses the client-build path entirely and returns a pre-made
        :class:`PostmortemResult`.
        """
        task_id = args["task_id"]
        path = self._resolve(task_id)
        task = TaskBundle.load(path)

        slug = args.get("slug") or task.config.target_models[0].slug
        mode = args.get("mode", "autonomous")
        allow_on_cost_cap = bool(args.get("allow_on_cost_cap", False))

        # Test seam.
        if self._run_postmortem_cb is not None:
            result = await self._run_postmortem_cb(
                task_root=path,
                slug=slug,
                mode=mode,
                allow_on_cost_cap=allow_on_cost_cap,
                task=task,
            )
            return self._summarize_postmortem_result(result)

        # Production path: real AnthropicClient as Analyst.
        from lpo.core.cost import CostTracker
        from lpo.models.anthropic_client import AnthropicClient
        from lpo.postmortem.runner import run_postmortem

        cost = CostTracker()
        analyst_model = task.config.postmortem.analyst_model_id
        analyst_client = AnthropicClient(model_id=analyst_model, cost_tracker=cost)
        try:
            result = await run_postmortem(
                path,
                slug=slug,
                analyst_client=analyst_client,
                mode=mode,
                cost=cost,
                allow_on_cost_cap=allow_on_cost_cap,
            )
        finally:
            await analyst_client.aclose()

        return self._summarize_postmortem_result(result)

    def _summarize_postmortem_result(self, result: Any) -> dict[str, Any]:
        """Convert a :class:`PostmortemResult` into a JSON-safe MCP payload.

        Keeps the envelope small — full artifacts live on disk. Callers
        read them via the path in ``postmortem_root`` when they want
        detail beyond the headline numbers.
        """
        decision = result.decision
        plan = result.plan
        payload: dict[str, Any] = {
            "outcome": decision.outcome,
            "mode": result.mode,
            "postmortem_root": str(result.postmortem_root),
            "analyst_model_id": result.analyst_model_id,
            "analyst_retries": result.analyst_retries,
            "total_cost_usd": round(result.total_cost_usd, 4),
            "finding_ids": [f.id for f in plan.diagnosis.findings],
            "auto_applied_intervention_ids": list(decision.auto_applied_intervention_ids),
            "report_only_intervention_ids": list(decision.report_only_intervention_ids),
            "rationale": decision.rationale,
        }
        if decision.deltas is not None:
            payload["deltas"] = {
                "global": decision.deltas.global_delta,
                "remediation": decision.deltas.remediation_delta,
                "max_scenario_regression": decision.deltas.max_scenario_regression,
                "pre_best_score": decision.deltas.pre_best_score,
                "post_best_score": decision.deltas.post_best_score,
                "retry_iterations_run": decision.deltas.retry_iterations_run,
            }
        return payload

    async def _tool_reload_env(self, args: dict[str, Any]) -> dict[str, Any]:
        """Reload ``.env`` into ``os.environ`` with override. Returns the
        diff (fingerprints only — never full values)."""
        dotenv_path = find_dotenv(usecwd=True)
        if not dotenv_path:
            return {
                "reloaded": False,
                "reason": "No .env file found on disk (searched from cwd upward).",
                "changed": [],
            }
        # Snapshot before.
        before = {k: os.environ.get(k) for k in dotenv_values(dotenv_path).keys()}
        load_dotenv(dotenv_path, override=True)
        # Snapshot after.
        after = {k: os.environ.get(k) for k in dotenv_values(dotenv_path).keys()}

        changed: list[dict[str, Any]] = []
        for k, new_val in after.items():
            old_val = before.get(k)
            if new_val != old_val:
                changed.append({
                    "key": k,
                    "old_fingerprint": _fingerprint(old_val),
                    "new_fingerprint": _fingerprint(new_val),
                })
        return {
            "reloaded": True,
            "dotenv_path": dotenv_path,
            "keys_checked": sorted(after.keys()),
            "changed": changed,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fingerprint(value: str | None) -> str:
    """Obfuscate a secret for logging. Returns a short identifier that is
    safe to surface to the operator but useless to an attacker."""
    if value is None:
        return "(not set)"
    s = str(value)
    if not s:
        return "(empty)"
    if len(s) <= 8:
        return f"(len={len(s)}, short)"
    return f"{s[:4]}...{s[-4:]} (len={len(s)})"


# Default single-target runner. Factored out so tests can swap it. Mirrors
# the CLI's ``_run_single`` path minus the rich-console output.
async def _default_run_single(
    task: TaskBundle,
    *,
    mutator_name: str,
    cost: CostTracker,
    fresh: bool,
) -> Any:
    import shutil

    target = task.config.target_models[0]
    if fresh:
        for p in (task.root / "runs" / target.slug, task.root / "logs"):
            if p.exists():
                shutil.rmtree(p, ignore_errors=True)

    ctx = build_target_context(task, target, cost, mutator_mode=mutator_name)
    try:
        engine = RatchetEngine(
            task=task,
            target_cfg=target,
            client=ctx.client,
            scorer=ctx.scorer,
            mutator=ctx.mutator,
            cost_tracker=cost,
        )
        return await engine.run()
    finally:
        await ctx.aclose()


# ---------------------------------------------------------------------------
# MCP SDK glue
# ---------------------------------------------------------------------------


def build_mcp_server(tasks_root: Path):
    """Return a configured mcp ``Server`` instance backed by
    :class:`LpoMcpHandlers`. Imported lazily so module load doesn't require
    the ``mcp`` package to be present (it's in the ``mcp`` extras)."""
    from mcp import types
    from mcp.server import Server

    handlers = LpoMcpHandlers(tasks_root=tasks_root)
    server: Server = Server("lpo")

    @server.list_tools()
    async def _list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name=spec["name"],
                description=spec["description"],
                inputSchema=spec["inputSchema"],
            )
            for spec in TOOL_SPECS
        ]

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
        result = await handlers.call(name, arguments)
        # MCP clients expect a list[Content]. We encode the JSON payload as a
        # single TextContent so Windsurf can show/parse it. The agent side
        # will json.loads() the text; see README snippet.
        return [types.TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]

    # Attach the handlers for tests that want to poke internal state.
    server._lpo_handlers = handlers  # type: ignore[attr-defined]
    return server


async def run_stdio(tasks_root: Path) -> None:
    """Serve the MCP protocol over stdio until the client disconnects."""
    from mcp.server.stdio import stdio_server

    server = build_mcp_server(tasks_root)
    log.info("LPO MCP server ready. tasks_root=%s", tasks_root)
    async with stdio_server() as (read_stream, write_stream):
        init_options = server.create_initialization_options()
        await server.run(read_stream, write_stream, init_options)
