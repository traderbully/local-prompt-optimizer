# Stage 8 — Postmortem Analysis & Remediation (Design)

**Status:** Proposal, awaiting review.
**Scope:** Design only. No implementation until approved.
**Relation to existing system:** New *phase* bolted onto the end of the main optimization loop; uses the existing `RatchetEngine` for validation but adds a new model role (the *Postmortem Analyst*) and new on-disk artifacts under `runs/<slug>/postmortem/`.

---

## 1. Problem

The Overseer (`lpo.overseer.mutator`) is iteration-local. Each mutation decision looks at the most recent iteration's outputs, scores, and the accumulating short-term conversation with the previous few iterations. This works well for local refinement — "output ex004 is missing the `verify_command` field, add a rule" — but is structurally blind to bird's-eye patterns. Three symptoms from the April 20 bake-off illustrate this:

1. **Concentrated failure classes the Overseer never names.** Scenarios `gui_launch_with_content`, `json_content`, and `content_special_chars` scored 0 in every iteration. All three share a root cause (PowerShell content-escaping rules absent from the prompt) but the Overseer's mutations drifted elsewhere because each iteration's failure looked like a different surface problem.
2. **Metric structure smells the Overseer can't see.** Uniform 43.33 across all 10 examples in iter 1 was diagnostic of a metric that wasn't content-sensitive — but this smell only appears when you compare per-scenario scores across examples, not within one iteration.
3. **Plateau-as-false-success.** The ratchet happily stops when `plateau_patience` fires, even though the plateau is often a *solvable* problem, not a local maximum.

A 30-second human read of the full run artifacts surfaces all three. The Overseer, watching iteration-by-iteration, does not.

## 2. Proposal at a glance

After the main ratchet terminates, run a dedicated **postmortem phase** with its own model role:

```
main ratchet exits (any stop reason except cost_cap)
        │
        ▼
[Postmortem Analyst]  ← reads full run artifacts from disk, single frontier call
        │   emits diagnosis.json (structured findings)
        ▼
[Remediation Planner]  ← same model, same conversation — emits proposal.md
        │   (patches to prompt / metric / seed; eval additions proposed but not applied)
        ▼
[Focused Validation]   ← reuses RatchetEngine for 3 iters with the patched prompt
        │   measures raw delta + remediation delta + regression risk
        ▼
[Decision Gate]        ← accept / reject / abstain against configurable thresholds
        │
        ▼
runs/<slug>/postmortem/{diagnosis.json, proposal.md, retry/, decision.json, report.md}
```

Opt-in. Defaults off. Can be triggered three ways: `config.yaml` flag (runs automatically after the main loop), MCP tool `lpo_run_postmortem`, or CLI `lpo postmortem <task_dir>`.

---

## 3. Where it plugs in

Bolts on after the main ratchet, not into it. Three entry points:

- **Automatic** — `config.yaml: postmortem.enabled: true` causes `run_single` / `run_multi` to invoke the phase once the main loop returns (any `stop_reason` except `cost_cap`, since budget exhaustion signals a different problem).
- **MCP tool** — `lpo_run_postmortem(task_id, slug?, mode="autonomous"|"propose_only")`. Lets an agent trigger it post-hoc on any completed run without re-running the main loop.
- **CLI** — `lpo postmortem <task_dir> [--slug X]` for manual invocation.

The engine itself is untouched. `RatchetEngine` only needs one new capability — the ability to start from a patched prompt and a restricted iteration budget — which it already supports via existing `prompt_override` and `max_iterations` parameters.

## 4. What "diagnosis" means

A single frontier-model call (same binding as the Overseer — Claude by default) with a clean context loaded with the full run artifacts: `task.md`, `eval_set.jsonl`, `gold_standard.jsonl`, `metric.yaml`, and every iteration's `prompt.txt`, `outputs.jsonl`, `scores.json`, `overseer_analysis.md`. The Overseer's own context is deliberately *not* inherited — we want a fresh read.

Output is `diagnosis.json` with a typed finding list:

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
        "scenarios": ["gui_launch_with_content", "json_content", "content_special_chars"],
        "iterations_affected": [1, 2, 3],
        "example_ids": ["ex001", "ex002", "ex008"]
      },
      "root_cause_hypothesis": "The prompt has no rules covering PowerShell content-escaping for nested quotes, newlines, or special characters. Outputs consistently emit unquoted content that breaks when the shell evaluates it."
    }
  ],
  "metric_observations": [...],
  "overseer_drift_observations": [...]
}
```

Finding types (closed set): `scenario_blindspot`, `prompt_gap`, `metric_mismatch`, `eval_coverage_gap`, `overseer_local_optimum`, `model_fit_issue`. Each finding cites evidence by iteration/example id so the report can link back to the artifacts.

## 5. What "remediation proposal" looks like

The same model call (or an immediate follow-up turn) produces `proposal.md` with one or more **typed interventions**:

| Intervention type | Effect | Applied by postmortem? |
|---|---|---|
| `prompt_patch` | Append/replace specific rules in `prompt.txt.best` | Yes, into a retry-scratch copy |
| `seed_reset` | Replace the seed prompt with a new one that bakes in missing rules | Yes, into retry-scratch |
| `metric_patch` | Add/modify rules in `metric.yaml` | Yes, into a `metric.postmortem.yaml` sidecar used only for the retry |
| `eval_addition` | Propose new eval examples stressing the missing category | **No** — logged for human review only |
| `model_swap_suggestion` | Flag that this failure pattern is characteristic of model X's limitations | **No** — advisory |

Each intervention carries a confidence score and an `expected_impact` range (e.g. "+10 to +25 on remediation-weighted score"). Interventions can be composed — a realistic proposal might bundle one `prompt_patch` and one `metric_patch`.

**Never auto-modified on disk:** `eval_set.jsonl`, `gold_standard.jsonl`, the original `prompt.txt.best`, the original `metric.yaml`. The postmortem only ever writes into `runs/<slug>/postmortem/` until the decision gate says otherwise.

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

Three decision outcomes written to `decision.json`:

- **`accepted`** — thresholds met. Postmortem promotes the retry winner: copies `retry/prompt.txt.best` to `runs/<slug>/prompt.txt.best`, copies `retry/winner/` into a new `winner/` with provenance field `source: "postmortem"`, leaves the original winner in `winner.pre_postmortem/` for rollback.
- **`rejected`** — thresholds not met. Original winner untouched. `proposal.md` and `diagnosis.json` remain on disk; operator can inspect and optionally apply manually.
- **`abstained`** — proposal contained only interventions the postmortem can't apply unilaterally (e.g. pure `eval_addition`). No retry was run. Report lists the human-approval items.

## 8. Relationship to the main ratchet

**Inherits, does not replicate.** The focused retry is a short `RatchetEngine` run with the same scoring stack, same cost tracker, same mutator infrastructure. The postmortem phase adds three things and *only* three things:

1. A new model role (Postmortem Analyst) with one new prompt.
2. A new intervention schema + patch-application logic.
3. A decision gate.

No new engine, no new scorer, no fork of the ratchet. If a future Stage expands the Analyst into a persistent watcher that runs alongside the main loop instead of after it, that's a separate design.

---

## 9. Open questions for review

1. **Autonomous vs propose-only as the MCP default.** Current proposal: autonomous in MCP mode (a caller who asks for a postmortem expects a verdict), propose-only in UI. Is that the right split, or should MCP also default to propose-only with an explicit `--apply` flag?
2. **How aggressive should `seed_reset` be?** Resetting the prompt to a new seed is a bigger intervention than a patch — it potentially throws away iterations of Overseer work. Should `seed_reset` require higher confidence than `prompt_patch` (e.g. ≥0.85 vs ≥0.7)?
3. **Cost ceiling per postmortem.** Rough estimate: 1 Opus-grade diagnostic call (~$0.50–$2.00 depending on run-history size) + 3 retry iterations (~3× a normal iteration cost). A realistic per-postmortem budget is $2–$10. Should this inherit the task's existing `cost_cap_usd` as remaining headroom, or have its own separate cap?
4. **What about cost_cap-terminated runs?** Current proposal skips postmortem when the main loop died on budget. Reasonable because the problem is "we ran out of money," not "the prompt can be improved." But a postmortem on a budget-killed run could still surface whether the cost was buying real progress or churning. Worth including with a shorter retry budget?
5. **Interactions with Strategy B/C.** For `parallel_independent`, postmortem runs per-slug and each slug's findings are independent. For `unified_portable`, one postmortem per task with per-model evidence. Confirmed, or do you want a cross-slug finding type ("this intervention would help slug X but hurt slug Y")?

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
