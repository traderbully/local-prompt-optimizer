---
description: Operating discipline for LPO optimization runs via the MCP server
---

# LPO Run Handling

These rules govern how Cascade should handle Local Prompt Optimizer (LPO) runs
invoked through the `mcp2_lpo_*` MCP tools. They exist because
`mcp2_lpo_run_optimization` blocks synchronously for the full duration of an
optimization (typically 5-30 min), and cancelling the tool call from the client
side does NOT stop the server-side work — it only drops visibility.

## Before invoking `mcp2_lpo_run_optimization`

1. **State the expected duration upfront.** Tell the user "this tool call will
   block for approximately N minutes while the optimization runs. If you
   cancel the call, the run will continue orphaned in the background but I
   will lose visibility into progress."
2. **Do NOT cancel `run_optimization` on the user's behalf.** If the user
   indicates they want to walk away, remind them the tool will complete and
   return results. Only kill worker processes if the user explicitly asks to
   stop the run.
3. **Verify preflight state with cheap tools:**
   - `mcp2_lpo_list_tasks` to confirm the task bundle exists with expected
     targets.
   - Shell-check LM Studio (`http://localhost:1234/v1/models`) if the target
     is `provider: lmstudio`.
   - `mcp2_lpo_reload_env` if any `.env` changes happened this session.

## During a blocked `run_optimization` call

- Don't fire repeated polling tool calls while the main call is blocked. The
  same MCP transport is in use and parallel calls may fight for resources.
- If you must show progress, poll filesystem artifacts via `run_command`:
  `tasks/<task_id>/runs/<slug>/history/iter_NNNN/scores.json` and
  `decision.json` update per iteration.

## When the user returns mid-run or after a cancelled call

Reconstruct state **from disk, not from tool calls**:

1. List iterations: `tasks/<task_id>/runs/<slug>/history/iter_NNNN/`.
2. Best score so far: scan `scores.json` files; max `aggregate` is the best.
3. Still running? Check for a python process whose path contains
   `Prompt Optimizer` AND whose CPU time is advancing (CPU=0.0 with no new
   iter artifact for >5 min AND LM Studio idle = likely stalled overseer).
4. Never re-fire `run_optimization` assuming it died without confirming via
   process inspection.

## After a run completes

Always run these, in order, before declaring the task done:

1. `mcp2_lpo_get_comparison(task_id)` to surface the final per-model table
   and winning prompt(s).
2. Tell the user the ranking, best scores, stop reasons, and costs.
3. **Offer `mcp2_lpo_run_postmortem` explicitly.** This is opt-in per the
   tool design. Recommended whenever: (a) target_score was not reached,
   (b) the winning score is less than 2 points above the second-best run, or
   (c) the user says something suggests they want deeper analysis.

## LM Studio quirks

- `/v1/models` timing out on a short (2-3s) timeout often means LM Studio is
  **busy serving inference**, not dead. Retry with an 8-10s timeout before
  declaring it unreachable.
- LM Studio activity is bursty during optimization: ~1-2 min of eval calls
  per iteration, then ~2-3 min of overseer + judge cloud calls (LM Studio
  completely idle). Idle between iters is normal, not a stall.
- Multiple `LM Studio` processes in task manager is normal (worker per model).

## MCP server restart semantics

- The LPO MCP server is a long-running Python process. Code changes on disk
  do NOT take effect until it's restarted — modules are read into memory
  once at startup.
- If an LPO source file was modified in this session, tell the user: "you'll
  need to restart the LPO MCP server for this change to take effect."
  Restart pattern: kill the MCP python process (DO NOT kill the one running
  the optimization if different), then reconnect via the Cascade MCP panel.

## Don'ts

- Don't proliferate new LPO task bundles when the user says "try another
  model" — add a target to the existing bundle and use `target_slugs` to
  filter at run time.
- Don't delete `runs/<slug>/history/` directories without explicit
  confirmation — they're the source of truth for postmortem analysis.
- Don't cancel a `run_optimization` call to "start fresh" — it wastes
  compute that the user has already paid for. If a run is bad, let it
  finish, record the result, then start a new one with different params.
