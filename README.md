# Local Prompt Optimizer (LPO)

Autonomous prompt engineering for local open-weight models. See `LPO_SDP.md` for the full design proposal.

## Status

**Stage 6 (current):** MCP server — every LPO capability (task authoring, gold-standard generation, headless optimization, status, winner retrieval, cross-model comparison) is exposed as a tool to any MCP client (Windsurf, Claude Desktop, etc.). The full engine, scoring, overseer, multi-target strategies, web UI, and MCP surface are in place; remaining stages (seed control, scenario weighting, polish) track `LPO_SDP.md` §10.

## Install

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

## Quick start

Start [LM Studio](https://lmstudio.ai) with its local OpenAI-compatible server on `http://localhost:1234/v1` and load a model.

Run the example task:

```powershell
lpo run tasks\example_json_extract
```

This executes the seed prompt against the eval set, scores outputs deterministically against the gold standard, and writes the first iteration under `tasks\example_json_extract\runs\<model_slug>\history\iter_0001\`.

## Task layout

See `LPO_SDP.md` §4.1. Each task is a self-contained directory:

```
tasks/<task_name>/
├── task.md
├── eval_set.jsonl
├── gold_standard.jsonl
├── config.yaml
├── metric.yaml
├── prompt_seed.txt        # initial prompt, copied to runs/<slug>/prompt.txt on first iteration
└── runs/<slug>/...        # created by the engine
```

## CLI

- `lpo run <task_dir>` — run optimization on a task bundle (Autonomous by default; supervised/manual via UI)
- `lpo ui [--tasks-root DIR] [--port 8787]` — launch the FastAPI + React web UI
- `lpo mcp [--tasks-root DIR]` — run as an MCP stdio server (for Windsurf / Claude Desktop / any MCP client)
- `lpo task list [root]` — list tasks under a directory
- `lpo task show <task_dir>` — summarize a task and its latest run

## Register as an MCP server in Windsurf

LPO exposes seven tools to MCP clients: `lpo_create_task`, `lpo_generate_gold_standard`, `lpo_run_optimization`, `lpo_get_status`, `lpo_get_winner`, `lpo_get_comparison`, `lpo_list_tasks`. Full schemas are emitted by `Server.list_tools` on connect — inspect them in Windsurf's MCP panel.

### Quick registration (per `LPO_SDP.md` §7.3)

Add this entry to Windsurf's MCP config file (**Windsurf → Settings → MCP** opens it in the editor; it lives at `~/.codeium/windsurf/mcp_config.json` on macOS/Linux and `%USERPROFILE%\.codeium\windsurf\mcp_config.json` on Windows):

```json
{
  "mcpServers": {
    "lpo": {
      "command": "lpo",
      "args": ["mcp", "--tasks-root", "C:/path/to/your/tasks"],
      "env": {
        "ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY}"
      }
    }
  }
}
```

### Notes

- **`command`**: Must resolve on Windsurf's `PATH`. The cleanest setup is a dedicated virtualenv with LPO installed (`pip install -e .`) whose `Scripts/` (Windows) or `bin/` (macOS/Linux) directory is on `PATH`. Alternatively invoke the venv directly: `"command": "C:/path/to/.venv/Scripts/lpo.exe"` on Windows or `".venv/bin/lpo"` on macOS/Linux.
- **`--tasks-root`**: Points LPO at the directory where task bundles live. If omitted, LPO uses `./tasks/` relative to wherever Windsurf launched it — usually not what you want. Passing an absolute path makes LPO usable from *any* project.
- **`ANTHROPIC_API_KEY`**: Required for Overseer-driven mutation, rubric/conversational scoring, and `lpo_generate_gold_standard`. Pure deterministic-metric flows with `mutator: "null"` work without it.
- **Headless contract**: MCP mode defaults to Autonomous. Manual and Visual modes are rejected (they require the web UI). Supervised works if your agent polls status between iterations.
- **One LPO, many projects**: Point every Windsurf workspace at the same `--tasks-root` directory and all tasks become visible from any project. Or run multiple registrations with different roots (`"lpo-work"`, `"lpo-personal"`, …).

### Verifying the registration

After saving the JSON, restart Windsurf (or use **Reload MCP Servers** from the command palette). In any chat, ask:

> List my LPO tasks.

Windsurf should call `lpo_list_tasks` and return the empty list (first time) or your existing bundles.

Create your first task from the agent:

> Create an LPO task called "email-subject-liner" that writes a short subject line summarizing an email body. Use these three examples: "...", "...", "...".

This invokes `lpo_create_task`, then `lpo_generate_gold_standard`, then `lpo_run_optimization` end-to-end.

## Gotchas

### Reasoning models (Gemma-4, DeepSeek-R1, QwQ, Qwen3-Thinking, gpt-oss, …)

Reasoning models emit a hidden Chain-of-Thought into `message.reasoning_content`
before producing the final answer in `message.content`. LM Studio's OpenAI-
compatible endpoint counts **both** channels against the `max_tokens` budget.
If the CoT exhausts the budget first, `content` comes back empty, `finish_reason`
is `"length"`, and the iteration scores 0 through no fault of the prompt.

The LM Studio client logs a `WARNING` ("LM Studio returned empty content but
reasoning_content has N chars …") whenever this happens. If you see it:

1. In LM Studio, reload the model with a context window ≥ 8k (gear icon →
   Context Length).
2. In your task's `config.yaml`, set `target_models[*].max_tokens` to at
   least 4096. Smaller numbers may truncate mid-thought.

Non-reasoning instruct models (Llama-3.1-Instruct, Qwen2.5-Instruct, etc.) do
not have this problem and can run comfortably at `max_tokens: 512`.

## Tests

```powershell
pytest
```
