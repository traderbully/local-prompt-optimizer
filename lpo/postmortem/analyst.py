"""Postmortem Analyst + Planner.

A single frontier-model call that reads a completed run's artifacts
(:class:`RunHistoryBundle`) and emits a validated :class:`PostmortemPlan`
(diagnosis + proposal).

Two invariants from the Apr 21 design review are enforced here at two
layers — belt and suspenders:

1. **Prompt layer.** The system prompt states each invariant in plain
   language and shows a worked example that satisfies them.
2. **Schema layer.** The Analyst's JSON output is validated against
   :class:`PostmortemPlan` before we return. If validation fails, we
   feed the Pydantic error back to the Analyst and ask for a re-emit.
   One retry only — a model that can't satisfy the schema after being
   told exactly what's wrong is unlikely to succeed with a second try,
   and we don't want to burn the postmortem budget on loops.

The Analyst client is abstracted behind :class:`AnalystClient` so tests
can inject a deterministic stub. The concrete implementation wraps
:class:`lpo.models.anthropic_client.AnthropicClient` — the same frontier
binding the Overseer uses, but with a fresh conversation so the
Overseer's iteration-local context doesn't contaminate the bird's-eye
read.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import ValidationError

from lpo.postmortem.artifacts import (
    IterationArtifact,
    IterationScores,
    RunHistoryBundle,
)
from lpo.postmortem.schemas import PostmortemConfig, PostmortemPlan

log = logging.getLogger("lpo.postmortem.analyst")


# ---------------------------------------------------------------------------
# Client protocol — lets tests inject a stub without pulling in the Anthropic
# SDK. The real client is lpo.models.anthropic_client.AnthropicClient, whose
# .complete() signature matches this.
# ---------------------------------------------------------------------------


class AnalystClient(Protocol):
    """Narrow interface around :class:`AnthropicClient` that the Analyst
    needs. Kept shape-compatible so AnthropicClient can be passed directly."""

    async def complete(
        self,
        *,
        system: str,
        messages: list[Any],
        temperature: float = 0.0,
        max_tokens: int = 8192,
    ) -> Any: ...


@dataclass
class AnalystResult:
    """What :func:`run_analyst` returns on success."""

    plan: PostmortemPlan
    raw_response: str
    retries: int
    model_id: str


# ---------------------------------------------------------------------------
# System prompt — stable, version-controlled text. Changes to this prompt
# should be made deliberately; the prompt IS the Analyst's specification.
# ---------------------------------------------------------------------------


_SYSTEM_PROMPT_TEMPLATE = """\
You are the Postmortem Analyst for an automated prompt-optimization system
called LPO. Your job is to read the full artifacts of a completed
optimization run — the task description, eval set, gold standard, metric
definition, every iteration's prompt + outputs + scores + Overseer analysis
— and produce a structured diagnosis plus a concrete remediation proposal.

You are NOT the Overseer. The Overseer has already made iteration-local
decisions on this run and those decisions are in the history you are about
to read. Your value is the bird's-eye view the Overseer never gets. Look
for structural patterns: whole classes of failure the Overseer never
names, metric-shape issues that only appear when comparing across
examples, local optima the Overseer kept drifting toward.

# Hard invariants (non-negotiable; outputs that violate them are rejected)

## Invariant 1 — Evidence
Every `finding` in your `diagnosis.findings` list MUST carry an
`evidence` object with:

- `iterations`: non-empty list of iteration indices (1-based) the
  finding is drawn from.
- `example_ids`: non-empty list of eval-example IDs the finding cites.
- `scenarios`: list of scenario tags. May be empty for findings that
  are not scenario-scoped (e.g. a `prompt_gap` applying across all).
- `score_breakdown`: non-empty map of example_id to per-iteration
  score, where each per-iteration score is keyed by `iter_1`, `iter_2`,
  etc. matching the indices in `iterations`. Every example key must
  appear in `example_ids`; every iter-label must correspond to an
  iteration in `iterations`.

Findings without concrete evidence are rejected by the downstream
schema validator. Do NOT emit narrative analysis without the numbers
that support it.

## Invariant 2 — Intervention provenance
Every `intervention` in your `proposal.interventions` list MUST carry
a non-empty `fixes` list referencing the IDs of the findings it
addresses. Example: `"fixes": ["F1", "F3"]`. An intervention that
does not tie back to specific findings is rejected.

## Invariant 3 — Differential evidence for strong claims
A finding of type `metric_mismatch` or `model_fit_issue` MUST additionally
include a `differential_evidence` string arguing why the diagnosis is
the metric/model rather than simply the prompt. These are the two types
most prone to hallucination; make the claim cost something.

## Invariant 4 — Metric patches are report-only
You may propose `metric_patch` interventions but they will NEVER be
auto-applied, regardless of their confidence. They always go to human
review. Propose them only when the evidence is strong enough that a
human should see them. Do not water down your real prompt diagnosis
to hide it behind a metric excuse.

# Closed sets

Allowed `finding.type` values:
- `scenario_blindspot` — scenarios consistently scoring near 0.
- `prompt_gap` — the prompt lacks a rule that would plausibly fix observed failures.
- `metric_mismatch` — the metric gave credit for wrong outputs OR failed to penalize them.
- `eval_coverage_gap` — the eval set underrepresents a category that the task description implies matters.
- `overseer_local_optimum` — the Overseer kept making the same class of mutation while the real problem was elsewhere.
- `model_fit_issue` — the failure pattern is characteristic of a specific model's tokenizer/alignment limitations.

Allowed `intervention.type` values and their REQUIRED `patch` shape:

- `prompt_patch` — append/prepend/replace specific rules in the current best prompt.
  Required patch shape: `{"mode": "append"|"prepend"|"replace", "content": "<text>"}`
- `seed_reset` — replace the seed prompt entirely (bigger intervention; use only when the best prompt is so warped that in-place patching won't recover).
  Required patch shape: `{"new_seed": "<complete new seed prompt>"}`
- `metric_patch` — add/modify rules in metric.yaml. Always report-only.
  Required patch shape: `{"rationale": "<why the metric is flawed>"}`
- `eval_addition` — propose new eval examples stressing a missing category. Advisory; never applied.
  Required patch shape: `{"new_examples": [{"input": "<text>", "expected": "<text>", "scenario": "<tag>"}, ...]}`  (non-empty list)
- `model_swap_suggestion` — flag that a different model may be needed. Advisory only.
  Required patch shape: `{"rationale": "<why the current model is inadequate>", "suggested_models": ["<model_id>", ...]}`  (non-empty list)

The `patch` field MUST be a JSON object with the keys listed above for
the chosen type. It is NEVER null, NEVER a string, and NEVER an empty
object. Interventions that omit required patch keys are rejected at
validation time, so emit every key explicitly even when the value is
short.

# Output format

Respond with exactly one JSON object — no preamble, no postamble, no
markdown fences. The object has two top-level keys, `diagnosis` and
`proposal`. Finding IDs are `F1`, `F2`, ... Intervention IDs are `I1`,
`I2`, ...

Here is a minimal example that satisfies every invariant:

```json
{
  "diagnosis": {
    "findings": [
      {
        "id": "F1",
        "type": "scenario_blindspot",
        "severity": "high",
        "confidence": 0.92,
        "summary": "Three content-escaping scenarios scored 0 in every iteration.",
        "evidence": {
          "iterations": [1, 2, 3],
          "example_ids": ["ex001", "ex002", "ex008"],
          "scenarios": ["content_special_chars"],
          "score_breakdown": {
            "ex001": {"iter_1": 0.0, "iter_2": 0.0, "iter_3": 0.0},
            "ex002": {"iter_1": 0.0, "iter_2": 0.0, "iter_3": 0.0},
            "ex008": {"iter_1": 0.0, "iter_2": 0.0, "iter_3": 0.0}
          }
        },
        "root_cause_hypothesis": "The prompt has no rules covering PowerShell content escaping."
      }
    ],
    "metric_observations": [],
    "overseer_drift_observations": [],
    "analyst_model_id": "__ANALYST_MODEL_ID__",
    "task_id": "__TASK_ID__",
    "slug": "__SLUG__"
  },
  "proposal": {
    "interventions": [
      {
        "id": "I1",
        "type": "prompt_patch",
        "fixes": ["F1"],
        "confidence": 0.88,
        "summary": "Add three content-escaping rules.",
        "expected_impact": {"global": [5, 12], "remediation": [15, 35]},
        "patch": {
          "mode": "append",
          "content": "- Content with single quotes: double each quote.\\n- Content with double quotes: use single-quoted here-strings.\\n- Content with newlines: use backtick-n inside double-quoted strings."
        }
      }
    ],
    "rationale": "All three failing scenarios share one root cause; one patch addresses them.",
    "human_review_summary": ""
  }
}
```

Be conservative: if the evidence does not support a finding, do not
invent one. An empty or short diagnosis is better than a padded one.
Your findings feed into an automated decision gate that can modify the
prompt without human review — so accuracy matters more than volume.
"""


def _system_prompt(
    *,
    analyst_model_id: str,
    task_id: str,
    slug: str,
) -> str:
    # Use str.replace rather than str.format so literal braces in the
    # JSON example don't have to be doubled (which was a footgun — the
    # moment a new instance appeared in a nested key, the template
    # would KeyError). Named sentinel tokens keep the substitution
    # explicit.
    return (
        _SYSTEM_PROMPT_TEMPLATE
        .replace("__ANALYST_MODEL_ID__", analyst_model_id)
        .replace("__TASK_ID__", task_id)
        .replace("__SLUG__", slug)
    )


# ---------------------------------------------------------------------------
# Context builder — turns a RunHistoryBundle into the user-message body.
# ---------------------------------------------------------------------------


def build_analyst_context(bundle: RunHistoryBundle) -> str:
    """Serialize the run history into the prompt body the Analyst reads.

    We use compact JSON rather than Markdown: structured data is easier
    for the model to cite from (it can quote example IDs and iter labels
    verbatim) and cheaper per token. The layout mirrors the on-disk
    artifacts so a future reader can reconstruct "what the Analyst saw"
    by diffing against the same files.
    """
    task = bundle.task
    payload: dict[str, Any] = {
        "task_id": task.config.task_name,
        "slug": bundle.slug,
        "task_description": task.task_md,
        "seed_prompt": task.seed_prompt,
        "metric": _metric_to_dict(task.metric),
        "eval_set": [
            {
                "id": r.id,
                "input": r.input,
                "scenario": r.scenario,
                "weight": r.weight,
            }
            for r in task.eval_records
        ],
        "gold_standard": {
            gid: (g.output if hasattr(g, "output") else g)
            for gid, g in (task.gold_standard or {}).items()
        },
        "iterations": [_iteration_to_dict(it) for it in bundle.iterations],
        "winner": {
            "prompt": bundle.winner_prompt,
            "report": bundle.winner_report,
        },
    }
    # ensure_ascii=False so non-ASCII eval inputs (e.g. content-escaping
    # examples with nested quotes) survive round-trip without \uXXXX noise.
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def _metric_to_dict(metric: Any) -> dict[str, Any]:
    # MetricConfig is a discriminated union of Pydantic models. model_dump
    # gives us a clean dict; fallbacks for anything else.
    if hasattr(metric, "model_dump"):
        return metric.model_dump()
    if isinstance(metric, dict):
        return metric
    return {"raw": str(metric)}


def _iteration_to_dict(it: IterationArtifact) -> dict[str, Any]:
    return {
        "index": it.index,
        "prompt": it.prompt,
        "outputs": it.outputs,
        "scores": _scores_to_dict(it.scores),
        "decision": {
            "decision": it.decision.decision,
            "delta": it.decision.delta,
            "cost_usd": it.decision.cost_usd,
            "timings": it.decision.timings,
            "notes": it.decision.notes,
        },
        "overseer_analysis": it.overseer_analysis,
    }


def _scores_to_dict(s: IterationScores) -> dict[str, Any]:
    return {
        "aggregate": s.aggregate,
        "per_example": s.per_example,
        "per_scenario": s.per_scenario,
        "failed_ids": s.failed_ids,
        "per_model": s.per_model,
    }


# ---------------------------------------------------------------------------
# Entry point — run the Analyst and return a validated plan.
# ---------------------------------------------------------------------------


class AnalystError(RuntimeError):
    """Raised when the Analyst produced output we cannot recover into a
    valid PostmortemPlan even after one retry with feedback."""


async def run_analyst(
    bundle: RunHistoryBundle,
    *,
    cfg: PostmortemConfig,
    client: AnalystClient,
    client_model_id: str | None = None,
    max_tokens: int = 8192,
) -> AnalystResult:
    """Call the Postmortem Analyst and return a validated plan.

    Flow:
    1. Build system prompt + run-history context.
    2. Ask the client once with ``temperature=0``.
    3. Parse JSON. If parsing or schema validation fails, ask once more
       with the error fed back. Any second failure raises AnalystError.

    ``client_model_id`` is only used to stamp the ``analyst_model_id``
    field on the returned :class:`PostmortemPlan` (when the Analyst
    omits it or emits a placeholder). Real clients expose their model
    id on the returned result object; we prefer that when available.
    """
    model_label = client_model_id or cfg.analyst_model_id
    system = _system_prompt(
        analyst_model_id=model_label,
        task_id=bundle.task.config.task_name,
        slug=bundle.slug,
    )
    user_context = build_analyst_context(bundle)

    messages: list[Any] = [_user_message(user_context)]

    # First attempt.
    raw, model_id_used = await _call_client(client, system, messages, max_tokens)
    plan, err = _try_parse(raw, analyst_model_id=model_id_used or model_label)
    if plan is not None:
        return AnalystResult(plan=plan, raw_response=raw, retries=0, model_id=model_id_used or model_label)

    log.info("Analyst output failed validation on first attempt: %s", err)
    # One retry with the error fed back — the model has seen the rubric
    # and now it sees exactly where it went wrong.
    retry_messages = messages + [
        _assistant_message(raw),
        _user_message(_retry_feedback(err)),
    ]
    raw2, model_id_used2 = await _call_client(client, system, retry_messages, max_tokens)
    plan2, err2 = _try_parse(raw2, analyst_model_id=model_id_used2 or model_label)
    if plan2 is not None:
        return AnalystResult(plan=plan2, raw_response=raw2, retries=1, model_id=model_id_used2 or model_label)

    raise AnalystError(
        "Analyst produced invalid output after one retry. "
        f"First error: {err}\nSecond error: {err2}\n"
        f"Last raw response (truncated): {raw2[:500]}"
    )


def _user_message(text: str) -> Any:
    """Build a message whose shape matches AnthropicClient.complete's
    expected ``messages`` element. We import lazily to keep this module
    importable from tests that stub the client."""
    try:
        from lpo.models.anthropic_client import AnthropicMessage
        return AnthropicMessage(role="user", content=text)
    except Exception:  # pragma: no cover — tests use a stub client that
        # accepts the dict shape. Real AnthropicClient rejects dicts but
        # never hits this branch because the import above always succeeds
        # when the anthropic SDK is installed (which is a hard runtime dep).
        return {"role": "user", "content": text}


def _assistant_message(text: str) -> Any:
    try:
        from lpo.models.anthropic_client import AnthropicMessage
        return AnthropicMessage(role="assistant", content=text)
    except Exception:  # pragma: no cover
        return {"role": "assistant", "content": text}


async def _call_client(
    client: AnalystClient,
    system: str,
    messages: list[Any],
    max_tokens: int,
) -> tuple[str, str | None]:
    """Invoke the client and extract ``(text, model_id_if_known)``."""
    result = await client.complete(
        system=system,
        messages=messages,
        temperature=0.0,
        max_tokens=max_tokens,
    )
    text = getattr(result, "text", None)
    if text is None and isinstance(result, dict):
        text = result.get("text", "")
    text = text or ""
    model_id = getattr(result, "model_id", None)
    return text, model_id


def _try_parse(
    raw: str,
    *,
    analyst_model_id: str,
) -> tuple[PostmortemPlan | None, str | None]:
    """Attempt to turn ``raw`` into a :class:`PostmortemPlan`. Returns
    ``(plan, None)`` on success or ``(None, error_message)`` otherwise."""
    stripped = _strip_fences(raw)
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError as e:
        return None, f"json decode error at char {e.pos}: {e.msg}"

    if not isinstance(data, dict):
        return None, f"expected a JSON object at top level, got {type(data).__name__}"

    # Stamp analyst_model_id with the authoritative value from the API
    # response (served model id). The model's self-reported identity in
    # its JSON output is unreliable: Claude models often misidentify
    # themselves (e.g. an opus-4-5 endpoint may self-report as
    # "claude-sonnet-4-20250514" because that's the latest snapshot the
    # model was trained to know about). We overwrite unconditionally so
    # that the envelope (which also uses the API-served id) and the
    # diagnosis.json payload agree. The prior setdefault behaviour
    # shipped the model's guess to disk and created a confusing
    # provenance discrepancy surfaced during the Stage-8 validation run.
    diagnosis = data.get("diagnosis")
    if isinstance(diagnosis, dict):
        diagnosis["analyst_model_id"] = analyst_model_id

    try:
        plan = PostmortemPlan.model_validate(data)
    except ValidationError as e:
        return None, _format_validation_error(e)
    return plan, None


def _strip_fences(raw: str) -> str:
    """Extract the JSON object from a raw model response.

    Handles three cases in order of preference:

    1. Pure JSON (happy path) — returned as-is.
    2. JSON wrapped in a markdown fence — strip the fence.
    3. JSON embedded in prose ("Here's the analysis:" + object + trailing
       commentary) — isolate the outermost balanced-brace object via a
       string-aware scan.

    Case 3 is the one :func:`_try_parse` used to fail on. Claude Opus
    frequently prefaces structured output with a sentence or two of
    narration despite the "respond with exactly one JSON object"
    instruction; we recover instead of burning the single retry budget
    on a cosmetic issue.
    """
    s = raw.strip()

    # Case 2: strip markdown fences first so case 3's scan works on the
    # bare payload regardless of whether the fence itself had prose
    # around it.
    if s.startswith("```"):
        first_newline = s.find("\n")
        if first_newline != -1:
            s = s[first_newline + 1 :]
        if s.endswith("```"):
            s = s[:-3]
        s = s.strip()

    # Case 1: already a JSON object. Fast path.
    if s.startswith("{") and s.endswith("}"):
        return s

    # Case 3: scan for the outermost JSON object. Track string state so a
    # `{` inside a JSON string doesn't throw off the depth counter.
    extracted = _extract_first_json_object(s)
    return extracted if extracted is not None else s


def _extract_first_json_object(s: str) -> str | None:
    """Return the substring of ``s`` containing the first balanced
    top-level JSON object, or None if no such object is found.

    String-aware: ``{`` and ``}`` inside JSON strings (including escaped
    quotes) don't count toward depth. This is the minimum needed to
    reliably extract a model's structured reply from surrounding prose;
    anything more elaborate and we'd pull in a real JSON parser.
    """
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None


def _format_validation_error(e: ValidationError) -> str:
    # Pydantic's default formatting is verbose; for the Analyst retry
    # message we surface the first few errors with path + message only.
    errors = e.errors()
    head = errors[:5]
    lines = []
    for err in head:
        loc = ".".join(str(p) for p in err.get("loc", ()))
        msg = err.get("msg", "invalid")
        lines.append(f"- {loc}: {msg}")
    more = len(errors) - len(head)
    if more > 0:
        lines.append(f"- ... and {more} more errors")
    return "schema validation failed:\n" + "\n".join(lines)


def _retry_feedback(error: str | None) -> str:
    return (
        "Your previous response did not validate against the required "
        "PostmortemPlan schema. The errors were:\n\n"
        f"{error}\n\n"
        "Please re-emit the JSON object with the errors corrected. "
        "Respond with exactly one JSON object — no preamble, no "
        "postamble, no markdown fences. Common fix checklist:\n"
        "\n"
        "- EVERY finding needs `root_cause_hypothesis` (non-empty string) — "
        "not just F1 but F2, F3, etc.\n"
        "- EVERY finding needs the full `evidence` block: `iterations`, "
        "`example_ids`, and `score_breakdown` (all non-empty).\n"
        "- EVERY intervention needs a non-empty `fixes` list of finding "
        "IDs like `[\"F1\"]`.\n"
        "- EVERY intervention needs a `patch` object matching its type's "
        "required shape:\n"
        "  * `prompt_patch` -> {mode, content}\n"
        "  * `seed_reset` -> {new_seed}\n"
        "  * `metric_patch` -> {rationale}\n"
        "  * `eval_addition` -> {new_examples: [...]}\n"
        "  * `model_swap_suggestion` -> {rationale, suggested_models: [...]}\n"
        "  The `patch` field is NEVER null, empty, or a string.\n"
        "- `metric_mismatch` and `model_fit_issue` findings also need "
        "`differential_evidence`.\n"
    )
