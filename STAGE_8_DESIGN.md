# Stage 8 — Postmortem Analysis & Remediation (Design)

**Status:** Approved with adjustments (Apr 21 2026). Implementation in progress.
**Scope:** Design doc for the Stage 8 postmortem phase.

**Review adjustments applied:**
- Postmortem is **opt-in only**. No automatic invocation at the end of `run_single` / `run_multi`. MCP tool call or explicit config flag required.
- Cost is governed by a **separate budget**: `postmortem.cost_cap_usd`, default `$2.50`. Does not share the task's `cost_cap_usd`.
- **Cross-slug findings** ("this intervention would help slug X but hurt slug Y") are out of scope for v1. Per-slug independent only. Revisit after real-world usage surfaces the need.
- **Metric-patch interventions are never auto-committed.** The Analyst may propose them and the meta-check sanity pass is still run for reporting, but the decision gate treats every `metric_patch` as report-only awaiting human approval. Only `prompt_patch` and `seed_reset` can be auto-committed on threshold.
- **Evidence-based findings only.** Every finding in `diagnosis.json` must cite specific iteration numbers, scenario IDs, and score breakdowns. Every intervention in `proposal.md` must reference the finding IDs it addresses. Free-floating narrative is rejected at schema validation.

**Relation to existing system:** New *phase* bolted onto the end of the main optimization loop; uses the existing `RatchetEngine` for validation but adds a new model role (the *Postmortem Analyst*) and new on-disk artifacts under `runs/<slug>/postmortem/`.

---

## 1. Problem

The Overseer (`lpo.overseer.mutator`) is iteration-local. Each mutation decision looks at the most recent iteration's outputs, scores, and the accumulating short-term conversation with the previous few iterations. This works well for local refinement — "output ex004 is missing the `verify_command` field, add a rule" — but is structurally blind to bird's-eye patterns. Three symptoms from the April 20 bake-off illustrate this:

1. **Concentrated failure classes the Overseer never names.** Scenarios `gui_launch_with_content`, `json_content`, and `content_special_chars` scored 0 in every iteration. All three share a root cause (PowerShell content-escaping rules absent from the prompt) but the Overseer's mutations drifted elsewhere because each iteration's failure looked like a different surface problem.
2. **Metric structure smells the Overseer can't see.** Uniform 43.33 across all 10 examples in iter 1 was diagnostic of a metric that wasn't content-sensitive — but this smell only appears when you compare per-scenario scores across examples, not within one iteration.
3. **Plateau-as-false-success.** The ratchet happily stops when `plateau_patience` fires, even though the plateau is often a *solvable* problem, not a local maximum.

A 30-second human read of the full run artifacts surfaces all three. The Overseer, watching iteration-by-iteration, does not.

## 2. Proposal at a glance

A dedicated **postmortem phase**, invoked only on explicit request, with its own model role:

```
(postmortem explicitly invoked — never auto-runs)
        │
        ▼
[Postmortem Analyst]  ← reads full run artifacts from disk, single frontier call
        │   emits diagnosis.json — every finding MUST cite iter numbers + scenario IDs + scores
        ▼
[Remediation Planner]  ← same model, same conversation — emits proposal.md
        │   every intervention MUST reference the finding IDs it addresses
        │   (prompt/seed patches auto-applicable; metric patches ALWAYS report-only; eval additions proposed not applied)
        ▼
[Focused Validation]   ← reuses RatchetEngine for 3 iters with the patched prompt
        │   measures global delta + remediation delta + regression risk
        │   (skipped entirely when the only applicable interventions are metric_patch or eval_addition)
        ▼
[Decision Gate]        ← accept (prompt/seed only) / reject / abstain (metric or eval interventions)
        │   separate postmortem.cost_cap_usd budget (default $2.50)
        ▼
runs/<slug>/postmortem/{diagnosis.json, proposal.md, retry/, decision.json, report.md}
```

**Opt-in only.** Two entry points: the MCP tool `lpo_run_postmortem` and the CLI `lpo postmortem <task_dir>`. There is no `postmortem.auto_run_after_ratchet` flag — the Apr 21 review explicitly rejected auto-invocation. `config.yaml` still carries a `postmortem:` block, but its fields govern thresholds and budget, not whether the phase runs.

---

## 3. Where it plugs in

Bolts on after the main ratchet, not into it. Two entry points (per Apr 21 review — no automatic invocation):

- **MCP tool** — `lpo_run_postmortem(task_id, slug?, mode="autonomous"|"propose_only")`. The primary entry point. Callable post-hoc on any completed run; can also be invoked from an agent conversation right after `lpo_run_optimization` returns.
- **CLI** — `lpo postmortem <task_dir> [--slug X]` for manual invocation from an operator's shell.

`run_single` / `run_multi` do **not** invoke the postmortem phase internally. A deliberately-left option `config.yaml: postmortem.enabled` was considered and rejected during review: opt-in by explicit call only, to keep the cost model predictable and avoid surprise billing on completed ratchet runs.

The engine itself is untouched. `RatchetEngine` only needs one new capability — the ability to start from a patched prompt and a restricted iteration budget — which it already supports via existing `prompt_override` and `max_iterations` parameters.

## 4. What "diagnosis" means

A single frontier-model call (same binding as the Overseer — Claude by default) with a clean context loaded with the full run artifacts: `task.md`, `eval_set.jsonl`, `gold_standard.jsonl`, `metric.yaml`, and every iteration's `prompt.txt`, `outputs.jsonl`, `scores.json`, `overseer_analysis.md`. The Overseer's own context is deliberately *not* inherited — we want a fresh read.

### Evidence invariant (Apr 21 review)

**Every finding must cite concrete evidence; free-floating narrative is rejected.** Schema-enforced via Pydantic validators on the diagnosis JSON: a finding without populated `evidence.iterations`, `evidence.example_ids`, and `evidence.score_breakdown` fails validation and the Analyst is asked to re-emit. This turns the postmortem into a debuggable audit: when a diagnosis is later judged wrong, we can walk back through the exact evidence the Analyst cited and see where it reasoned incorrectly.

Output is `diagnosis.json`:

```json
{
  "findings": [
    {
      "id": "F1",
      "type": "scenario_blindspot",
      "severity": "high",
      "confidence": 0.92,
      "summary": "Three scenarios involving content escaping scored 0 in every iteration.",
      "evidence": {
        "iterations": [1, 2, 3, 4],
        "scenarios": ["gui_launch_with_content", "json_content", "content_special_chars"],
        "example_ids": ["ex001", "ex002", "ex008"],
        "score_breakdown": {
          "ex001": {"iter_1": 0.0, "iter_2": 0.0, "iter_3": 0.0, "iter_4": 0.0},
          "ex002": {"iter_1": 0.0, "iter_2": 0.0, "iter_3": 0.0, "iter_4": 0.0},
          "ex008": {"iter_1": 0.0, "iter_2": 0.0, "iter_3": 0.0, "iter_4": 0.0}
        }
      },
      "root_cause_hypothesis": "The prompt has no rules covering PowerShell content-escaping for nested quotes, newlines, or special characters. Outputs consistently emit unquoted content that breaks when the shell evaluates it."
    }
  ],
  "metric_observations": [...],
  "overseer_drift_observations": [...]
}
```

Finding types (closed set): `scenario_blindspot`, `prompt_gap`, `metric_mismatch`, `eval_coverage_gap`, `overseer_local_optimum`, `model_fit_issue`. All six require the full `evidence` block. `metric_mismatch` and `model_fit_issue` additionally require evidence that distinguishes the diagnosis from easier alternatives (e.g., a `metric_mismatch` claim must cite examples where the metric gave the *wrong* score, not merely a low one).

## 5. What "remediation proposal" looks like

The same model call (or an immediate follow-up turn) produces `proposal.md` plus a machine-readable `proposal.json` with one or more **typed interventions**. Every intervention carries a `fixes` list referencing the finding IDs it addresses — an intervention with no finding references fails schema validation (Apr 21 evidence invariant).

| Intervention type | Effect | Auto-applicable on threshold? |
|---|---|---|
| `prompt_patch` | Append/replace specific rules in `prompt.txt.best` | **Yes** — written to retry-scratch; auto-committed if the decision gate accepts |
| `seed_reset` | Replace the seed prompt with a new one that bakes in missing rules | **Yes** — same treatment as `prompt_patch` (higher confidence bar; see §9) |
| `metric_patch` | Add/modify rules in `metric.yaml` | **No** — per Apr 21 review, **always** report-only. A `metric.postmortem.yaml` sidecar is written so operators can inspect + apply manually; the decision gate never commits it |
| `eval_addition` | Propose new eval examples stressing the missing category | **No** — logged for human review only |
| `model_swap_suggestion` | Flag that this failure pattern is characteristic of model X's limitations | **No** — advisory |

Example intervention with the mandatory `fixes` field:

```json
{
  "id": "I1",
  "type": "prompt_patch",
  "fixes": ["F1", "F3"],
  "confidence": 0.88,
  "expected_impact": {"global": [5, 12], "remediation": [15, 35]},
  "patch": {
    "mode": "append",
    "after_section": "## Rules",
    "content": "- Content containing single quotes: replace each with two single quotes before interpolating.\n- Content containing double quotes: wrap the whole string in single-quoted here-strings @'...'@.\n- Content containing newlines: use `\n` sequences inside double-quoted strings only."
  }
}
```

Interventions can be composed — a realistic proposal might bundle one `prompt_patch` and one `metric_patch`. When a proposal contains both auto-applicable and report-only interventions, the auto-applicable ones proceed to focused validation; the report-only ones are copied verbatim into `report.md` for human review.

**Never auto-modified on disk:** `eval_set.jsonl`, `gold_standard.jsonl`, the original `prompt.txt.best`, the original `metric.yaml`. The postmortem only ever writes into `runs/<slug>/postmortem/` until the decision gate explicitly promotes a retry winner (prompt/seed only).

## 6. How the focused retry scopes down

**Full eval set, short iteration budget, dual-score measurement.** Three iterations (configurable) with the patched prompt against the *whole* eval set. This gives us:

- **Global score delta** = `new_best - old_best` on the aggregate metric.
- **Remediation delta** = improvement on scenarios previously scoring ≤ `failure_threshold` (default 20/100).
- **Regression risk** = max per-scenario score drop on scenarios previously scoring ≥ `success_threshold` (default 70/100).

Running the full eval set is deliberate — the Goodhart failure mode of "the intervention helps what it was designed for but silently breaks everything else" is the single biggest risk. We need to see regressions, not just improvements.

First retry iteration uses `mutator: "null"` (measure the patch alone, without further Overseer drift). Iterations 2-3 allow Overseer refinement but with a constrained system prompt that tells it "you are validating a specific intervention, do not make unrelated changes."

## 7. What counts as "meaningful improvement"

Commit the intervention when **all three** hold:

```
global_delta           >= config.postmortem.accept_threshold_global         (default +5)
remediation_delta      >= config.postmortem.accept_threshold_remediation    (default +15)
max_scenario_regression <= config.postmortem.regression_tolerance           (default 3)
```

`AND` semantics chosen deliberately: any one threshold alone can be gamed (global alone misses concentrated regressions; remediation alone allows Goodharting the failing subset at the cost of the rest; regression tolerance alone gives credit for doing nothing).

**The gate applies only to `prompt_patch` and `seed_reset` interventions.** Per Apr 21 review, `metric_patch` is always report-only — the thresholds do not authorize committing a metric change, ever. If a proposal contains only `metric_patch` and/or `eval_addition` interventions, the focused retry is skipped entirely and the decision is `abstained`.

Four decision outcomes written to `decision.json`:

- **`accepted`** — thresholds met on auto-applicable interventions. Postmortem promotes the retry winner: copies `retry/prompt.txt.best` to `runs/<slug>/prompt.txt.best`, copies `retry/winner/` into a new `winner/` with provenance field `source: "postmortem"`, leaves the original winner in `winner.pre_postmortem/` for rollback.
- **`rejected`** — thresholds not met. Original winner untouched. `proposal.md` and `diagnosis.json` remain on disk; operator can inspect and optionally apply manually.
- **`abstained`** — proposal contained no auto-applicable interventions (only `metric_patch`, `eval_addition`, and/or `model_swap_suggestion`). No retry was run. Report lists the human-approval items in priority order.
- **`partial`** — proposal mixed auto-applicable and report-only interventions. The auto-applicable subset was validated; if it passed thresholds the prompt/seed change is committed and the report-only interventions are listed for human review separately.

## 8. Relationship to the main ratchet

**Inherits, does not replicate.** The focused retry is a short `RatchetEngine` run with the same scoring stack, same cost tracker, same mutator infrastructure. The postmortem phase adds three things and *only* three things:

1. A new model role (Postmortem Analyst) with one new prompt.
2. A new intervention schema + patch-application logic.
3. A decision gate.

No new engine, no new scorer, no fork of the ratchet. If a future Stage expands the Analyst into a persistent watcher that runs alongside the main loop instead of after it, that's a separate design.

---

## 9. Resolved review items & remaining design calls

**Resolved in Apr 21 review:**

1. *Autonomous MCP default?* — **Opt-in only.** MCP tool + CLI are the only entry points. No auto-invocation.
2. *Cost ceiling?* — **Separate budget.** `postmortem.cost_cap_usd` (default `$2.50`), independent of the task's own cap.
3. *Cross-slug findings?* — **Out of scope for v1.** Per-slug independent. Revisit after real-world usage.
4. *Metric-patch auto-commit?* — **Never.** `metric_patch` interventions are always report-only regardless of thresholds. The decision gate authorizes `prompt_patch` and `seed_reset` only.
5. *Evidence requirement?* — **Enforced at schema layer.** Findings without iteration/scenario/score evidence fail validation; interventions without `fixes` references fail validation.

**Remaining design calls I'll make during implementation unless told otherwise:**

- **`seed_reset` confidence bar.** Resetting the prompt is a bigger intervention than patching. Default confidence floor: `≥ 0.85` for `seed_reset` vs `≥ 0.70` for `prompt_patch`. Configurable via `postmortem.min_confidence_seed_reset` and `postmortem.min_confidence_prompt_patch`.
- **Cost-cap-terminated main runs.** Skip the postmortem by default — if the task's main loop died on budget, the operator should fix the budget first. Overridable via MCP tool argument `allow_on_cost_cap=True`.
- **Mode default.** MCP tool defaults to `mode="autonomous"` (diagnose → patch → retry → decide). CLI defaults to `mode="propose_only"` (diagnose → emit proposal → stop before retry). This mirrors the existing headless/interactive split across the codebase.

## 10. Risks

- **Hallucinated patterns.** The Analyst can invent a structural story that's coherent but wrong. The decision gate (regression tolerance + AND-semantics on thresholds) is the main defense; a bad diagnosis produces a patch that fails validation and gets rejected. But bad *metric* patches could still slip through if they inflate scores artificially. Mitigation: `metric_patch` interventions get an extra validation pass — compare new-metric scores *on the old prompt* to old-metric scores; if they diverge sharply, the metric patch is scoring-system manipulation rather than measurement improvement.
- **Goodhart on the retry metric.** See above — the full-eval-set retry + regression check is the main defense.
- **Stacking postmortems.** Running the postmortem twice in a row on the same task risks ratcheting the prompt toward the Analyst's biases. Proposal: refuse to run a postmortem if the current best winner has `source: "postmortem"` and no intervening main-ratchet run, unless `--force`.
- **Making rejection useful.** A rejected proposal is still information. The rejected `proposal.md` should be preserved (not deleted), and the final report should quote the Analyst's hypothesis verbatim so the operator can assess it manually.

## 11. Explicitly out of scope for Stage 8

- No UI surface. The web UI can read the artifacts after the fact; no live postmortem dashboard.
- No multi-iteration postmortems. Exactly one Analyst call → one proposal → one retry → one decision, per invocation.
- No auto-editing of `eval_set.jsonl`. Proposals only.
- No model-swap automation. `model_fit_issue` findings are advisory only; changing `target_models` in `config.yaml` stays a human decision.
- No cross-task learning. Each task's postmortem is independent; we don't train a meta-model on accepted/rejected proposals across tasks.

---

**Awaiting review.** Implementation plan will follow the same batching pattern as Stage 7 (artifacts + Analyst → proposal schema + patch application → focused retry + decision gate → MCP/CLI surface + docs), but I won't start until you've signed off on the design.
