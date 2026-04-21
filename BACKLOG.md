# LPO Backlog

Non-urgent polish, refactors, and architectural improvements. Items are in
rough priority order within each section. File an entry here instead of
silently losing context when a good idea surfaces during a run.

## Priority: polish

### Async/polling API for `run_optimization` (2026-04-21)

**Problem:** `mcp2_lpo_run_optimization` blocks the MCP tool call for 5-30 min.
This is fine when invoked from a CLI or a long-lived script, but causes three
failure modes when driven through Cascade / Windsurf MCP:

1. Users instinctively cancel long-blocking calls, which drops client-side
   visibility while work continues orphaned server-side.
2. If the IDE session goes idle, the MCP transport can drop mid-run and the
   final results never reach the assistant.
3. There's no "walk away and come back" pattern — when the user returns, the
   assistant has to reconstruct state from disk artifacts via shell tools.

**Proposed fix:** split the single blocking tool into three:

- `mcp2_lpo_run_optimization_start(task_id, target_slugs, ...)` returns a
  `run_id` immediately after spawning a background worker process.
- `mcp2_lpo_run_optimization_status(run_id)` returns
  `{iteration, best_score, is_running, elapsed_s, eta_s, last_decision}`.
  Cheap, always fast.
- `mcp2_lpo_run_optimization_wait(run_id, max_wait_s=60)` blocks up to N
  seconds then returns latest status. Lets Cascade choose its own cadence.

Keep the existing `run_optimization` as a thin sync-wrapper over these three
for CLI / script users.

Rule file `.windsurf/rules/lpo-runs.md` already encodes the operational
workaround; this ticket removes the need for the workaround entirely.

## Priority: would-be-nice

### Auto-postmortem trigger on failed `target_score` (2026-04-21)

`mcp2_lpo_run_postmortem` is explicitly opt-in per the design doc. Operationally
the Cascade rule now always *offers* it when `target_score` isn't reached — but
that's still a manual confirmation step. Consider a `postmortem: auto` option
in `config.yaml` that triggers it automatically within the existing cost cap
when a run terminates on `plateau_patience` / `max_iterations` without hitting
`target_score`. Keep `auto` off by default so API spend stays predictable.

### `DONE.json` completion marker (2026-04-21)

Even with the async API above, having the worker write
`runs/<slug>/DONE.json` with `{ended_at, best_score, stop_reason, cost}` when
it finishes would make post-hoc detection trivial for any client, not just
the MCP-aware ones. Cheap.
