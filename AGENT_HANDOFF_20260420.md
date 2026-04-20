# LPO Agent Handoff — 2026-04-20

Field report from an integration attempt. The user was trying to run a parallel_independent bake-off (4 targets: gemma-4-26b-local, gemma-4-31b-or, gpt-4.1-mini, haiku-4.5) on a new `windows_file_ops_reliability` task. Task bundle was created successfully; gold-standard generation was blocked by an Anthropic 401. Debugging that 401 exposed three distinct LPO issues worth fixing — the Anthropic key validity itself is out of scope (user action item).

Task bundle already exists and is valid; no need to recreate:
- `E:\CascadeProjects\Local Prompt Optimizer\tasks\windows_file_ops_reliability\`

---

## Issue #1 — `load_dotenv()` silently ignored, stale shell env wins

**Severity:** High. Causes a confusing class of "I updated the key and it still fails" reports that are very hard to diagnose from the outside.

**Location:** `lpo/cli/main.py:45`

```python
@app.callback()
def _root(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    load_dotenv()   # <-- override=False is the python-dotenv default
    _configure_logging(verbose)
```

**Observed behavior:**
1. User updated `E:\CascadeProjects\Local Prompt Optimizer\.env` with a fresh `ANTHROPIC_API_KEY`.
2. User restarted Windsurf → MCP server process PIDs confirmed new (fresh spawn timestamps).
3. The respawned MCP process inherited the **old** `ANTHROPIC_API_KEY` from the User-scope Windows environment variable (set weeks ago).
4. `load_dotenv()` ran, saw `ANTHROPIC_API_KEY` was already populated in `os.environ`, and — per python-dotenv's default `override=False` — refused to overwrite it.
5. Result: `.env` and `os.environ` disagreed; the process used the stale value; Anthropic returned 401; nothing in LPO surfaced the mismatch.

This is the LPO equivalent of "silently trust whatever contaminated state the parent process had."

**Proposed fix (one-line):**

```python
load_dotenv(override=True)
```

Rationale: the repo-local `.env` should be the authoritative source for a project-scoped tool. Tools that want shell-env precedence can explicitly unset the `.env` value. Silent loss-of-update is worse than explicit override.

**Optional enhancement (defense in depth):** also emit a warning when `.env` and shell env disagree for any secret key, regardless of which wins. A one-liner in `_root` that diffs `dotenv_values(find_dotenv())` against `os.environ` for keys ending in `_API_KEY` and logs a `rich` warning panel.

---

## Issue #2 — Opaque auth errors don't identify the key source

**Severity:** Medium. Wastes operator time during key-rotation incidents.

**Location:** `lpo/models/anthropic_client.py` (the 401 bubbles up to the MCP tool unhandled).

**Observed error (verbatim from `lpo_generate_gold_standard`):**

```
Internal error: ModelError: Anthropic API error: Error code: 401 -
{'type': 'error', 'error': {'type': 'authentication_error', 'message': 'invalid x-api-key'},
 'request_id': 'req_011CaF5zTx7JuNA54mFZfKqk'}
```

**Missing context the operator needs:**
- Key fingerprint that was actually sent (first 4 + last 4 chars + length)
- Which source populated `os.environ["ANTHROPIC_API_KEY"]` — `.env`, shell inheritance, or per-task override
- Whether `.env` and shell env disagree (see Issue #1)

**Proposed fix:** wrap the `AnthropicClient` init to catch `AuthenticationError` and re-raise with enriched context, e.g.:

```
AnthropicClient auth failed (401 invalid x-api-key).
  key fingerprint sent : sk-a...kwAA  (len=108, prefix=sk-ant-api03-)
  source              : os.environ (populated at process start; not overridden by .env)
  .env value          : sk-a...kwAA  (matches — both sources agree, key is simply invalid at Anthropic)
  request_id          : req_011CaF5zTx7JuNA54mFZfKqk

Next steps: verify key at https://console.anthropic.com/settings/keys
            check billing at https://console.anthropic.com/settings/billing
```

That one diagnostic block would have collapsed a 90-minute debugging session to 30 seconds.

---

## Issue #3 — No hot-reload for config; every `.env` edit costs a full Windsurf restart

**Severity:** Low but nagging. Quality-of-life for the developer tuning things.

**Observed:** To pick up a `.env` change, the user must fully quit and relaunch Windsurf because:
- 6 `lpo mcp` server processes run concurrently (one per IDE window)
- Each only calls `load_dotenv()` at startup
- No reload/refresh MCP tool exists

**Proposed fix:** add an MCP tool `lpo_reload_env` that calls `load_dotenv(override=True)` in the server process and returns the list of env keys whose values changed (fingerprints only, never full values). Advertised in the tool description as "call this after editing `.env` to avoid restarting Windsurf."

Implementation is trivial — 10 lines in `lpo/server/mcp_server.py`.

---

## Issue #4 — Gold Standard Source hardcoded to `api.anthropic.com`

**Severity:** Medium — limits who can use LPO out of the box.

**Context:** Per SDP §3.1, Gold Standard must be frontier-quality. Current implementation assumes direct Anthropic billing. But users with working `OPENROUTER_API_KEY` can already access `anthropic/claude-opus-4.6` and `anthropic/claude-sonnet-4.6` through OpenRouter — often with credit pools they've already provisioned for other work.

**Proposed fix:** make the Gold Standard Source respect a `GOLD_STANDARD_BASE_URL` + `GOLD_STANDARD_API_KEY_ENV` pair in `.env`, defaulting to Anthropic direct but transparently supporting OpenRouter:

```env
# Optional: route Gold Standard Source through OpenRouter instead of direct Anthropic
GOLD_STANDARD_BASE_URL=https://openrouter.ai/api/v1
GOLD_STANDARD_API_KEY_ENV=OPENROUTER_API_KEY
GOLD_STANDARD_MODEL=anthropic/claude-opus-4.6
```

Would unblock a whole class of users (including this one) who have healthy OpenRouter accounts but no standalone Anthropic billing relationship.

---

## Reproduction instructions for your agent

Once the Anthropic key issue is resolved OR Issue #4 is implemented:

1. Confirm `mcp2_lpo_list_tasks` shows `windows_file_ops_reliability` (already created).
2. `mcp2_lpo_generate_gold_standard(task_id="windows_file_ops_reliability")`.
3. `mcp2_lpo_run_optimization(task_id="windows_file_ops_reliability", stop_conditions={max_iterations: 15, cost_cap_usd: 5, target_score: 90, plateau_patience: 4})`.
4. `mcp2_lpo_get_comparison(task_id="windows_file_ops_reliability")` → surface the per-model winner scores.

The task was designed to stress-test the exact failure pattern we debugged in a separate system (the "greeting.txt" trace, documented in that project's logs): blocking GUI commands, unverified success claims, improperly-quoted Windows paths. Deterministic scoring is defined in the task's `metric.yaml`; no LLM judge is required for this bake-off.

---

## Priority suggestion

1. **Issue #1** first — it's a one-line change and prevents a whole class of silent failures.
2. **Issue #2** next — high leverage on debuggability.
3. **Issue #4** before shipping to other users — removes a hard dependency on one specific billing relationship.
4. **Issue #3** when bandwidth allows — polish.

All four are independent. All four are safe changes.
