# Local Prompt Optimizer (LPO)

**Autonomous prompt engineering for local open-weight models.**

LPO closes the quality gap between a local model running on your own GPU and a frontier API like Claude or GPT-5 — not by fine-tuning, not by swapping models, but by letting a frontier model **rewrite the system prompt for you** until your local model's outputs match frontier quality on *your* task. The winning prompt is committed to your repo as a permanent asset. You pay the frontier API cost once, during optimization. Every inference after that runs on your hardware, for free, offline, with no data leaving the machine.

See [`LPO_SDP.md`](./LPO_SDP.md) for the full design proposal and v1.1 specification.

---

## Why this exists

### The economic asymmetry of frontier models

Frontier models (Claude, GPT-5, Gemini) succeed on ambiguous natural-language tasks out of the box, with almost no prompt engineering. Local open-weight models of comparable parameter count — Gemma-4, Llama-3, Qwen3, DeepSeek — can reach the same quality **on a well-scoped task**, but only with a carefully engineered system prompt. That engineering effort is where the real cost hides.

A "perfect prompt" for a local model typically takes **days or weeks** of manual iteration:

- Read outputs by hand, diagnose failure modes, edit the prompt, re-run the eval, repeat
- Every edit risks regressing the cases that already worked
- Every model family needs a different prompt, and swapping models means starting over
- The work is unglamorous, easy to defer, and hard to parallelize across humans

That friction is why engineers default back to paid APIs even when their hardware could run the workload for free. The cost is real, but the cost is *labor*, not GPU time.

### LPO eliminates the labor

You give LPO:

1. A **task description** — one paragraph of natural language
2. An **eval set** — a handful of representative inputs (5+ is enough to start)
3. A **target model** — whatever open-weight model you run locally

LPO generates reference outputs with Claude, then autonomously iterates a system prompt against your local model until its outputs match the references. A task that would have taken a week of manual tweaking typically resolves in an hour of autonomous optimization, and the resulting prompt is yours forever — a plain `.txt` file checked into your repo.

### The three-model architecture

LPO separates concerns across three *roles*, so the right tool does each job:

| Role | What it does | Default |
|---|---|---|
| **Target Model** | The local model being optimized. Executes the candidate prompt against eval inputs each iteration. | LM Studio on `localhost:1234` |
| **Overseer Model** | The frontier model doing the engineering. Reads failures, proposes prompt edits, manages metric evolution. | Claude (Anthropic API) |
| **Gold Standard Source** | The frontier model that generates reference outputs once, at setup. Ground truth for scoring. | Claude (Anthropic API) |

Each role is independently configurable. The Overseer and Gold Standard can share a provider or differ. All three can be stubbed for offline tests. Frontier cost is bounded and one-time; steady-state inference is free and local.

### The ratchet: monotonic improvement

LPO's optimization loop is a **ratchet**, not a search. It implements the pattern Karpathy calls the "autoresearch ratchet," adapted from ML training research to prompt optimization:

```
1. Execute current prompt against eval set → outputs
2. Score outputs → current_score
3. If current_score > best_score:
       Commit as new best
   Else:
       Revert to previous best
4. Overseer analyses failures, proposes prompt edit
5. Apply edit, loop to step 1
6. Terminate on target score, plateau, max iterations, user stop, or cost cap
```

**Why a ratchet and not, e.g., beam search or RL?**

- **It cannot regress.** Every iteration either improves the best-known prompt or is discarded. You can stop at any time and know the prompt on disk is the best LPO has produced so far.
- **It's inspectable.** Each iteration's prompt, outputs, score, and Overseer rationale are written to disk as human-readable files. No opaque gradient updates, no embeddings, no hidden state.
- **It's cheap.** One frontier API call per iteration (the Overseer's edit) plus however much local inference you want. Cost caps are enforced per-run.
- **It composes with humans.** Drop the Overseer, switch to Manual mode, and a human becomes the mutator. The engine doesn't care.

### Prompts as durable artifacts

Every task in LPO is a **task bundle** — a self-contained directory of plain text and JSONL files. No database, no binary format, no hidden state. You can:

- `diff` two iterations of a prompt
- `git log` the evolution of a winning prompt across months
- Copy a `prompt.txt.best` into a downstream service and know exactly what LPO measured it to do
- Hand a bundle to a teammate as a folder and they can reproduce the run bit-for-bit

This is deliberate. Prompts are the source code of LLM-powered software; they deserve to be treated like code.

### Cross-model portability (Strategy C)

If you run more than one local model — or swap models as hardware improves — you do not want to re-engineer prompts every time. LPO supports three target configurations:

| Strategy | Meaning | Use case |
|---|---|---|
| **A — Single target** | Optimize one prompt against one model. | You have one model, you want the best prompt for it. |
| **B — Parallel independent** | Optimize one prompt *per* model, in parallel. Produces a cross-model comparison report. | *"Which of these local models reaches frontier quality most efficiently on my task?"* |
| **C — Unified portable** | Optimize a **single shared prompt** that scores well across *every* target. Aggregation configurable (`min`, `mean`, `weighted_mean`). | *"Give me one prompt that works on any of these models, so I can swap them without rewriting."* |

The Overseer in Strategy C receives per-model score breakdowns, so it can target whichever model is currently the weak link.

### Objective measurement, not vibes

Scoring is taken seriously. Three metric families are available, stackable:

- **Deterministic** — field-exact match, numeric tolerance, regex, JSON schema conformance, string-similarity. Free, fast, reproducible, runs without any API key.
- **Rubric** — structured LLM-judge prompt with criteria and weights. Catches semantic failures deterministic metrics miss.
- **Conversational** — multi-turn judge with access to task description, gold standard, and the Overseer's last rationale. For tasks where "quality" can't be tabulated.

Metrics are declared in `metric.yaml`, not hardcoded. You can start with pure deterministic (fast, free, offline) and add a rubric layer only when you need it.

### Headless-first, MCP-native

LPO has a **web UI** (FastAPI + React + Tailwind) for visual inspection, live run monitoring, manual-mode feedback, and winner export. But the UI is strictly optional. The same engine is exposed as an **MCP stdio server** with seven tools:

- `lpo_create_task` — author a task bundle from a description and inputs
- `lpo_generate_gold_standard` — fill in reference outputs
- `lpo_run_optimization` — run the ratchet to completion
- `lpo_get_status` — iteration count, best score, cost
- `lpo_get_winner` — return the winning prompt and its report
- `lpo_get_comparison` — cross-model report for Strategies B and C
- `lpo_list_tasks` — discover bundles under a tasks root

Register LPO once in Windsurf or Claude Desktop and you can drive the entire optimization pipeline from agent chat: *"Create an LPO task that writes email subject lines, generate the gold standard, run it, and tell me the winning prompt."* The agent does every step, end-to-end, without ever leaving the conversation.

---

## Status

**Stage 6 of the SDP — MCP server complete.** The engine, scoring stack, Overseer, multi-target strategies (A/B/C), web UI, and MCP surface are all in place and covered by 111 tests running in under five seconds. Remaining polish (deterministic seed control, scenario weighting, dashboard refinements) tracks [`LPO_SDP.md`](./LPO_SDP.md) §10.

---

## Install

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

Requires Python 3.11+. On first run, set `ANTHROPIC_API_KEY` (for Overseer + gold standard) and optionally `OPENROUTER_API_KEY` (for OpenRouter targets). See `.env.example`.

---

## Quick start

Start [LM Studio](https://lmstudio.ai) with its OpenAI-compatible server on `http://localhost:1234/v1` and load a model.

Run the bundled example:

```powershell
lpo run tasks\example_json_extract
```

This executes the seed prompt against the eval set, scores outputs deterministically against the gold standard, and writes iteration artifacts under `tasks\example_json_extract\runs\<model_slug>\history\iter_0001\`. Each subsequent iteration writes a new directory and either advances or reverts `prompt.txt.best`.

---

## Task layout

Each task is a self-contained directory (full schema in [`LPO_SDP.md`](./LPO_SDP.md) §4.1):

```
tasks/<task_name>/
├── task.md              # Plain-language goal, constraints, output format
├── eval_set.jsonl       # Input examples with optional scenario tags
├── gold_standard.jsonl  # Reference outputs (generated once at setup)
├── config.yaml          # Run configuration: targets, strategy, stop conditions
├── metric.yaml          # Scoring definition (deterministic / rubric / conversational)
├── prompt_seed.txt      # Initial prompt, copied to runs/<slug>/prompt.txt on first iteration
└── runs/<slug>/         # Created by the engine: prompt.txt(.best), history/, winner/
```

Everything is plain text or JSONL. The bundle is the source of truth; LPO keeps no database.

---

## CLI

- `lpo run <task_dir>` — run optimization on a task bundle (Autonomous by default)
- `lpo ui [--tasks-root DIR] [--port 8787]` — launch the FastAPI + React web UI
- `lpo mcp [--tasks-root DIR]` — run as an MCP stdio server (for Windsurf / Claude Desktop / any MCP client)
- `lpo task list [root]` — list task bundles under a directory
- `lpo task show <task_dir>` — summarise a task and its latest run

---

## Register as an MCP server in Windsurf

LPO exposes seven tools over MCP stdio. Schemas are emitted by `Server.list_tools` on connect — inspect them in Windsurf's MCP panel.

### Registration (per [`LPO_SDP.md`](./LPO_SDP.md) §7.3)

Open Windsurf's MCP config (**Windsurf → Settings → MCP**; or `~/.codeium/windsurf/mcp_config.json` on macOS/Linux, `%USERPROFILE%\.codeium\windsurf\mcp_config.json` on Windows) and add:

```json
{
  "mcpServers": {
    "lpo": {
      "command": "lpo",
      "args": ["mcp", "--tasks-root", "C:/path/to/your/tasks"],
      "env": {
        "ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY}",
        "OPENROUTER_API_KEY": "${OPENROUTER_API_KEY}"
      }
    }
  }
}
```

### Notes

- **`command`**: Must resolve on Windsurf's `PATH`. Cleanest is a dedicated virtualenv with LPO installed (`pip install -e .`) whose `Scripts/` (Windows) or `bin/` (Linux/macOS) directory is on `PATH`. Alternatively invoke the venv directly: `"command": "C:/path/to/.venv/Scripts/lpo.exe"` on Windows or `".venv/bin/lpo"` elsewhere.
- **`--tasks-root`**: Absolute path to where task bundles live. Omit it and LPO uses `./tasks/` relative to wherever Windsurf spawned the process — almost never what you want.
- **`ANTHROPIC_API_KEY`**: Required for Overseer-driven mutation, rubric / conversational scoring, and `lpo_generate_gold_standard`. Pure deterministic flows with `mutator: "null"` work without it.
- **Headless contract**: MCP mode is Autonomous-only. Manual and Visual modes require the web UI and are rejected over MCP. Supervised works if your agent polls status between iterations.
- **One LPO, many projects**: Every workspace can point at the same `--tasks-root` so all tasks are visible everywhere. Or register multiple LPO servers with different roots (`"lpo-work"`, `"lpo-personal"`, …).

### Verifying

After saving, restart Windsurf (full quit + relaunch — new env vars are inherited by new processes only). In any Cascade chat:

> List my LPO tasks.

Windsurf should call `lpo_list_tasks` and return your bundles. End-to-end authoring works too:

> Create an LPO task called "email-subject-liner" that writes a short subject line summarising an email body. Here are three examples: …

That invokes `lpo_create_task`, then `lpo_generate_gold_standard`, then `lpo_run_optimization` — all without leaving the agent chat.

---

## Gotchas

### Reasoning models (Gemma-4, DeepSeek-R1, QwQ, Qwen3-Thinking, gpt-oss, …)

Reasoning models emit a hidden Chain-of-Thought into `message.reasoning_content` before producing the final answer in `message.content`. LM Studio's OpenAI-compatible endpoint counts **both** channels against the `max_tokens` budget. If the CoT exhausts the budget first, `content` comes back empty, `finish_reason` is `"length"`, and the iteration scores 0 — through no fault of the prompt.

The LM Studio client logs a `WARNING` ("LM Studio returned empty content but reasoning_content has N chars …") whenever this happens. If you see it:

1. In LM Studio, reload the model with a context window ≥ 8k (gear icon → Context Length).
2. In your task's `config.yaml`, set `target_models[*].max_tokens` to at least 4096. Smaller numbers may truncate mid-thought.

Non-reasoning instruct models (Llama-3.1-Instruct, Qwen2.5-Instruct, etc.) do not have this problem and run comfortably at `max_tokens: 512`.

### `${ENV_VAR}` substitution in MCP config

Windsurf expands `${ANTHROPIC_API_KEY}` from the **environment of the process that spawned Windsurf**, not from shell startup files. If you set a new user env var on Windows, quit Windsurf fully and relaunch — a reload is not enough.

---

## Tests

```powershell
pytest
```

111 tests, ~4.7s on a modern laptop. Stub clients let the full suite run with no network and no API keys.

---

## License & acknowledgements

Design pattern borrowed from Andrej Karpathy's description of the autoresearch ratchet loop. Anthropic's `claude-haiku-4-5` is the default Overseer and Gold Standard binding. LM Studio is the default local inference runtime. None of these projects are affiliated with LPO.
