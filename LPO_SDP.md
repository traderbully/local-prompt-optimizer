# Software Design Proposal: Local Prompt Optimizer (LPO)

**Version:** 1.1
**Target Implementer:** Claude Opus 4.7 via Windsurf
**Target Runtime:** Windows/Linux, Python 3.11+, local GPU (RTX 5090 reference)
**Status:** Ready for implementation

**Changelog:**
- 1.1 — Added multi-target-model support (Section 3.4, updated 4.7, 5.2, 9, 10)
- 1.0 — Initial specification

---

## 1. Executive Summary

Local Prompt Optimizer (LPO) is an autonomous prompt engineering system that closes the quality gap between local open-weight models and frontier API models for well-scoped tasks. The user supplies a task description and representative inputs; LPO iteratively rewrites a system prompt for a local target model until its outputs match frontier-model quality, then returns the winning prompt for permanent use in downstream workflows.

LPO is delivered as a single Python project exposing two interfaces to the same underlying engine: an MCP server for programmatic invocation by an agentic IDE (Windsurf), and a local web UI for interactive exploration, visual inspection, and human-in-the-loop iteration.

The architectural pattern derives from Karpathy's autoresearch ratchet loop, adapted from ML training research to prompt optimization. LPO extends that pattern with a conversational overseer agent, multiple evaluation modalities (automated, supervised, manual, visual), scenario-based eval sets, deterministic seed control, and dynamic metric adjustment.

---

## 2. Problem Statement

Frontier models (Claude, GPT-5, Gemini) succeed on ambiguous natural-language tasks without extensive prompt engineering. Local open-weight models of comparable parameter count can approach this quality with carefully engineered prompts, but the engineering effort is prohibitive — a "perfect prompt" can take days or weeks of manual iteration. This economic gap drives users back to paid APIs even when their hardware could handle the workload.

LPO eliminates the manual engineering effort. A task that previously required weeks of prompt iteration is resolved in an hour of autonomous optimization, and the resulting prompt is committed into the codebase as a permanent asset.

---

## 3. System Overview

### 3.1 Core Roles

LPO separates concerns across three model roles. These roles are architectural; the specific model bound to each role is configurable.

| Role | Purpose | Default Binding |
|------|---------|-----------------|
| **Target Model** | The local model being optimized. Executes the prompt under test against eval inputs. | LM Studio local server |
| **Overseer Model** | The frontier model driving the optimization. Analyzes failures, rewrites prompts, judges outputs in semantic mode, manages metric evolution. | Claude (Anthropic API) |
| **Gold Standard Source** | The frontier model that generates reference outputs at setup time, used as ground truth for scoring. | Claude (Anthropic API) |

The Overseer and Gold Standard Source may be the same provider but are logically distinct and must be configurable independently.

### 3.2 The Ratchet Loop

The optimization engine implements a monotonic ratchet:

```
1. Execute current prompt against eval set → outputs
2. Score outputs → current_score
3. If current_score > best_score:
       Commit prompt as new best
       Update best_score
   Else:
       Revert prompt to best
4. Overseer analyzes failures, proposes prompt edit
5. Apply edit, loop to step 1
6. Terminate on: target score reached, plateau detected, max iterations, user stop, or cost cap
```

The ratchet is implemented via file versioning on disk, not git. Each iteration writes `prompt.txt` to an experiment directory with a sequential index; `prompt.txt.best` is a copy of the current winner. No git dependency.

### 3.3 Evaluation Modes

Four modes, user-selectable per invocation, switchable mid-run:

1. **Autonomous** — Overseer judges every iteration. No human input after launch. Runs to completion.
2. **Supervised** — Overseer judges autonomously but pauses at each iteration for a lightweight user signal (thumbs up/down, numeric rating, tag selection). User signal is added to overseer context.
3. **Manual** — Loop pauses every iteration; user provides free-text feedback. Overseer interprets and translates to prompt changes.
4. **Visual** — Output rendered in UI for user inspection. Required for image/video outputs. Can combine with supervised or manual rating.

Mode selection is a runtime parameter, not a build-time configuration. The same engine handles all modes via a unified iteration callback.

### 3.4 Target Model Configurations

LPO supports optimization against a single target model or multiple target models concurrently. The user selects one of three strategies per task.

**Strategy A — Single Target (default):**
One target model. Standard single-pipeline optimization. All prior sections describe this case.

**Strategy B — Parallel Independent (comparison mode):**
A list of target models. LPO runs the full optimization loop independently for each model, producing one winning prompt per model and a comparison report.

- Each model gets its own `prompt.txt.best`, history, and winner directory, namespaced by model id
- Runs may execute sequentially (default, to avoid GPU contention on a single local host) or in parallel if the user has multiple endpoints
- Final artifact is a comparison table: model → final score, iteration count, time to target, winning prompt
- Use case: "Which local model reaches frontier quality most efficiently on this task?"

**Strategy C — Unified Portable (consensus mode):**
A list of target models. LPO optimizes a single shared prompt that must score well across all models. Each iteration executes the current prompt against every target, produces a per-model score, and aggregates.

- Aggregation function configurable: `min` (worst-case — prompt must work on all), `mean` (average performance), `weighted_mean` (per-model weights for priority)
- Ratchet decision uses the aggregate score
- The Overseer receives per-model breakdowns in its analysis context so it can target models that are underperforming
- Final artifact is one prompt plus a per-model score report
- Use case: "Give me a prompt that works across any of these models so I can swap them without rewriting prompts."

Strategy is declared in `config.yaml` (see 4.7). The engine treats Strategy A as the degenerate case of a one-element target list, so the core loop logic is unified.

---

## 4. Functional Requirements

### 4.1 Task Specification

Each optimization run is defined by a **task bundle** in a dedicated directory:

```
tasks/<task_name>/
├── task.md                    # Plain-language goal, constraints, output format
├── eval_set.jsonl             # Input examples with optional scenario tags
├── gold_standard.jsonl        # Reference outputs (generated at setup)
├── config.yaml                # Run configuration (see 4.7)
├── metric.yaml                # Scoring rubric (see 4.4)
├── runs/                      # One subdirectory per target model (single-element for Strategy A)
│   └── <model_slug>/
│       ├── prompt.txt         # Current prompt under optimization
│       ├── prompt.txt.best    # Best prompt found so far
│       ├── history/
│       │   ├── iter_0001/
│       │   │   ├── prompt.txt
│       │   │   ├── outputs.jsonl
│       │   │   ├── scores.json
│       │   │   ├── overseer_analysis.md
│       │   │   └── decision.json
│       │   └── iter_0002/...
│       └── winner/
│           ├── prompt.txt
│           └── report.md
└── comparison/                # Populated for Strategy B and C
    ├── summary.json           # Cross-model metrics and final scores
    └── report.md              # Human-readable comparison
```

For Strategy A (single target), `runs/` contains exactly one subdirectory. For Strategy B (parallel independent), one subdirectory per model, each with its own independent optimization. For Strategy C (unified portable), one subdirectory holding the shared prompt plus per-model scoring traces in each iteration's `scores.json`.

### 4.2 Eval Set Structure

`eval_set.jsonl` — one JSON object per line:

```json
{"id": "ex_001", "input": "<input text or structured data>", "scenario": "casual_tone", "weight": 1.0}
{"id": "ex_002", "input": "...", "scenario": "formal_tone", "weight": 1.5}
{"id": "ex_003", "input": "...", "scenario": "edge_case_ambiguous", "weight": 2.0}
```

- `id` — stable identifier for cross-iteration tracking
- `input` — the task input (text, JSON, base64 image, or URL)
- `scenario` — optional category tag; enables per-category scoring breakdown
- `weight` — optional per-example weight in aggregate scoring

Minimum eval set size: 5 examples. Recommended: 10–30. No hard upper limit, but iteration time scales linearly with set size.

### 4.3 Gold Standard Generation

At task setup, LPO calls the Gold Standard Source once per eval input with a zero-shot or lightly-prompted configuration to produce reference outputs. These are persisted in `gold_standard.jsonl` and not regenerated unless explicitly reset. This is the only mandatory frontier API cost per task.

For visual tasks (image generation), the gold standard is produced by the same local model under a curated "known-good" prompt provided by the user, OR by a frontier image model if available. The user's expectation of "frontier quality" for visuals is necessarily manual, which is why Visual mode exists.

### 4.4 Metric System

Metrics are defined in `metric.yaml` and are of three types:

**Type A — Deterministic (code-scored):**
```yaml
type: deterministic
rules:
  - name: json_valid
    weight: 20
    check: is_valid_json
  - name: required_fields_present
    weight: 30
    check: has_keys
    params: [name, date, location]
  - name: field_exact_match
    weight: 50
    check: exact_match_against_gold
```

**Type B — Rubric-scored (LLM judge with fixed rubric):**
```yaml
type: rubric
judge_model: claude-opus-4-7
criteria:
  - name: semantic_accuracy
    weight: 40
    description: "Does the output convey the same meaning as gold standard?"
    anchors:
      0: "Completely different meaning"
      50: "Partially correct, missing key points"
      100: "Semantically equivalent"
  - name: tone_match
    weight: 30
    description: "Does the output tone match the gold standard?"
    anchors:
      0: "Inappropriate tone"
      50: "Acceptable but mismatched"
      100: "Tone matches"
  - name: format_compliance
    weight: 30
    description: "Does the output follow required format constraints?"
```

**Type C — Conversational (overseer judges with accumulated context):**
```yaml
type: conversational
overseer_model: claude-opus-4-7
stated_goal: |
  <user's plain-language description of what "good" means>
```

The metric system is pluggable. Adding new scoring types is a matter of implementing a `Scorer` interface (see 5.3).

### 4.5 Metric Evolution

In any mode, the Overseer may propose a metric adjustment after observing N iterations (default N=5) or when explicitly triggered. Proposed changes are:

- Surfaced to the user as a diff against current `metric.yaml`
- Accompanied by a plain-language rationale
- Gated on user approval — even in Autonomous mode, metric changes require an explicit `approve_metric_change(iteration_id)` call

If the user is not present (headless MCP invocation), metric change proposals are logged but not applied; the run continues with the current metric.

### 4.6 Seed Control

All calls to the Target Model accept an optional `seed` parameter. Three seed policies:

- **Fixed** — Single seed used for all iterations. Eliminates sampling variance as a confound during optimization.
- **Fixed per example** — Each eval example has its own deterministic seed (derived from `id`), consistent across iterations but varied across inputs.
- **Unlocked** — No seed passed; natural sampling variance. Used for final validation after a winner is selected.

Default: Fixed per example during optimization; Unlocked for post-winner validation report.

### 4.7 Run Configuration

`config.yaml` per task:

```yaml
task_name: ebay_listing_generator
mode: supervised                    # autonomous | supervised | manual | visual

target_strategy: single             # single | parallel_independent | unified_portable

# target_models is always a list. For target_strategy: single, provide exactly one.
# For parallel_independent and unified_portable, provide two or more.
target_models:
  - slug: qwen-32b                  # used for run directory naming and reports
    provider: lmstudio
    base_url: http://localhost:1234/v1
    model_id: qwen2.5-32b-instruct
    temperature: 0.2
    seed_policy: fixed_per_example
    weight: 1.0                     # used only in unified_portable with weighted_mean

# Aggregation applies only when target_strategy is unified_portable.
unified_aggregation: min            # min | mean | weighted_mean

# Concurrency applies only when target_strategy is parallel_independent.
# sequential = run models one after another (safe for single-GPU local setups).
# parallel   = run models concurrently (requires separate endpoints or sufficient VRAM).
parallel_execution: sequential      # sequential | parallel

overseer_model:
  provider: anthropic
  model_id: claude-opus-4-7
stop_conditions:
  target_score: 95
  max_iterations: 50
  plateau_patience: 5               # iterations without improvement
  cost_cap_usd: 10.00
metric_evolution:
  enabled: true
  check_every_n_iterations: 5
  require_user_approval: true
output_type: text                   # text | json | image | structured
```

**Validation rules enforced at config load:**

- `target_strategy: single` requires exactly one entry in `target_models`
- `target_strategy: parallel_independent` and `unified_portable` require at least two entries
- `unified_aggregation: weighted_mean` requires every target model to have a non-null `weight`
- `parallel_execution: parallel` is only meaningful for `parallel_independent`; ignored otherwise
- `slug` values must be unique within the list and filesystem-safe (`[a-z0-9_-]`)

### 4.8 Invocation Interfaces

**MCP Server:**

Exposes the following tools to an MCP client (Windsurf):

- `lpo_create_task(name, task_description, example_inputs, output_type) → task_id`
- `lpo_generate_gold_standard(task_id) → status`
- `lpo_run_optimization(task_id, mode, stop_conditions) → run_id`
- `lpo_get_status(run_id) → {iteration, best_score, history_summary, per_model_status}`
- `lpo_get_winner(task_id, model_slug=None) → {prompt, score, report}`
- `lpo_get_comparison(task_id) → {strategy, per_model_results, winner_recommendation}`
- `lpo_list_tasks() → [task_summary]`

For Strategy A (single target), `lpo_get_winner` returns the single winner directly; `model_slug` is optional.
For Strategy B (parallel independent), `lpo_get_winner` requires `model_slug` to disambiguate; `lpo_get_comparison` returns the cross-model comparison report.
For Strategy C (unified portable), `lpo_get_winner` returns the single shared prompt plus per-model score breakdown.

MCP mode is headless. Manual and Visual modes cannot be executed over MCP (they require UI presence). MCP invocations default to Autonomous mode; Supervised mode may be used if the MCP client can handle prompt-back calls.

**Web UI:**

Local FastAPI + React app. Launched via `lpo ui` command. Exposes:

- Task browser / creator (with target strategy selection)
- Live iteration viewer with left/right/center layout (conversation / current output / history)
- Per-model tab view when Strategy B or C is active — switch between models or see unified dashboard
- Cross-model comparison view for Strategy B runs (side-by-side scores, prompts, iteration counts)
- Mode switcher (changeable mid-run)
- Metric editor with diff view for overseer proposals
- Visual output rendering (images, side-by-side comparison, grid view for scenario sets)
- Export winner to clipboard / file (single or batch export for parallel independent)

### 4.9 Cost Tracking

Every API call to the Overseer or Gold Standard Source is logged with token counts and estimated cost. A running total is maintained per run; hitting `cost_cap_usd` triggers graceful stop with partial-winner reporting.

Local Target Model calls are logged for timing but not costed.

---

## 5. Technical Architecture

### 5.1 Component Map

```
lpo/
├── core/
│   ├── engine.py              # Main ratchet loop orchestrator (single model)
│   ├── multi_engine.py        # Multi-model strategy orchestrator (B and C)
│   ├── task.py                # Task bundle load/save
│   ├── iteration.py           # Single iteration execution
│   └── history.py             # Iteration persistence
├── models/
│   ├── base.py                # ModelClient ABC
│   ├── lmstudio.py            # OpenAI-compatible local client
│   ├── anthropic_client.py    # Claude API client
│   └── registry.py            # Provider registry
├── scoring/
│   ├── base.py                # Scorer ABC
│   ├── deterministic.py       # Rule-based scorers
│   ├── rubric.py              # LLM-judge with rubric
│   ├── conversational.py      # Overseer-judge with context
│   └── aggregation.py         # Weighted aggregation, scenario breakdown
├── overseer/
│   ├── agent.py               # Overseer conversation management
│   ├── prompt_writer.py       # Prompt rewrite logic
│   ├── metric_proposer.py     # Metric evolution logic
│   └── context.py             # Persistent conversation context
├── server/
│   ├── mcp_server.py          # MCP interface
│   ├── api.py                 # FastAPI routes for UI
│   └── schemas.py             # Pydantic models
├── ui/
│   ├── frontend/              # React + Vite app
│   └── static/                # Built assets
├── cli/
│   └── main.py                # `lpo` command entry
├── config/
│   ├── defaults.yaml
│   └── schema.py
└── tests/
    ├── test_engine.py
    ├── test_scoring.py
    ├── test_overseer.py
    └── fixtures/
```

### 5.2 Key Interfaces

```python
# models/base.py
class ModelClient(ABC):
    @abstractmethod
    async def generate(
        self,
        system_prompt: str,
        user_input: str | list[ContentBlock],
        seed: int | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> GenerationResult: ...

# scoring/base.py
class Scorer(ABC):
    @abstractmethod
    async def score(
        self,
        output: str | bytes,
        gold: str | bytes,
        input_record: EvalRecord,
        context: ScoringContext,
    ) -> ScoreResult: ...
```

`GenerationResult` includes raw output, token counts, latency, seed used, and provider metadata. `ScoreResult` includes aggregate score, per-criterion breakdown, and a rationale string.

### 5.3 Overseer Context Management

The Overseer maintains a persistent conversation thread per run. Each iteration appends a structured turn:

```
ITERATION 7
Prompt used:
<prompt content>

Eval results summary:
- Aggregate score: 87.3 (delta +2.1)
- Scenario breakdown: casual_tone=92, formal_tone=85, edge_case=71
- Failed examples: ex_003, ex_011

User feedback (if supervised/manual): "<user message>"

Your task: Analyze the failures, propose a prompt edit that addresses
them without regressing on passing scenarios.
```

The Overseer responds with:
1. Failure analysis (free text)
2. Proposed prompt diff (structured)
3. Expected improvement hypothesis

Context is truncated intelligently past a token budget (default 100k): oldest iterations are summarized, most recent 10 kept verbatim, running summary maintained of accepted prompt changes and their effects.

**Multi-model context handling:**

For Strategy B (parallel independent), each model's optimization has its own isolated Overseer conversation. Contexts do not cross-contaminate; each prompt is tuned to its specific target model.

For Strategy C (unified portable), a single Overseer conversation governs the shared prompt. The iteration turn is augmented with per-model breakdowns:

```
ITERATION 7
Prompt used: <shared prompt content>

Per-model results:
- qwen-32b:     aggregate 91.2 (delta +1.8)  scenarios: casual=95, formal=90, edge=88
- gemma-27b:    aggregate 83.5 (delta +0.4)  scenarios: casual=89, formal=85, edge=76
- llama-70b:    aggregate 88.1 (delta +2.1)  scenarios: casual=93, formal=88, edge=83

Aggregation: min = 83.5, mean = 87.6
Weakest link: gemma-27b on edge scenarios
Your task: Improve the weakest model's performance without regressing others.
```

The Overseer is explicitly prompted to consider cross-model tradeoffs and avoid over-specialization to any single model.

### 5.4 Concurrency

Eval set execution is parallelized across examples (async fan-out, configurable concurrency, default 4). Overseer calls are sequential. UI updates are pushed via WebSocket during a run.

### 5.5 Persistence

All state is filesystem-based under the task bundle directory. No database. JSON and YAML for structured data, JSONL for append-only logs, raw text for prompts and analyses. Survives process crashes — runs can be resumed from last committed iteration.

---

## 6. Non-Functional Requirements

### 6.1 Reliability

- All file writes are atomic (write to temp, fsync, rename)
- API calls have exponential backoff retry (max 3 attempts) with separate handling for rate limits vs hard errors
- Run state is resumable after crash — on startup, detect in-progress runs and offer resume
- Overseer output parsing is defensive — malformed responses trigger a retry with a clarifying instruction before failing the iteration

### 6.2 Observability

- Structured logging (JSON) to `logs/<run_id>.jsonl`
- Per-iteration timing breakdown (generation, scoring, overseer, I/O)
- Cost counter updated after every billable call
- UI shows real-time progress; MCP status tool returns equivalent data

### 6.3 Security

- API keys loaded from `.env`, never committed, never logged
- LM Studio endpoint is local-only; no outbound traffic from Target Model
- Web UI binds to `127.0.0.1` by default; remote binding requires explicit flag
- No telemetry

### 6.4 Performance Targets

- Engine overhead per iteration: < 200ms excluding model calls
- UI responsive during active runs (WebSocket push, no polling)
- Parallelized eval execution should saturate the local model's throughput without queueing

---

## 7. Build & Deployment

### 7.1 Dependencies

Python: `anthropic`, `httpx`, `fastapi`, `uvicorn`, `pydantic`, `pyyaml`, `mcp`, `websockets`, `pytest`, `pytest-asyncio`

Frontend: React 18, Vite, TailwindCSS, shadcn/ui components. No state-management library — React Context is sufficient.

### 7.2 Packaging

- `pyproject.toml` with `uv` or `pip` install path
- Single entry point: `lpo` CLI with subcommands `ui`, `mcp`, `run`, `task`
- Frontend built once at install time and served as static assets by FastAPI

### 7.3 MCP Server Registration

Windsurf MCP configuration snippet provided in README:

```json
{
  "mcpServers": {
    "lpo": {
      "command": "lpo",
      "args": ["mcp"],
      "env": {
        "ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY}"
      }
    }
  }
}
```

### 7.4 First-Run Experience

`lpo init` creates `~/.lpo/` with default config, `.env` template, and an example task. User edits `.env`, runs `lpo ui`, creates their first task through the web interface.

---

## 8. Out of Scope (v1.0)

- Multi-user / cloud deployment
- Fine-tuning the target model (prompt-only optimization)
- Cross-task prompt transfer learning
- Target model **ensembling at inference time** (Strategies B and C are in scope; runtime routing between models at serve time is not)
- Git integration (explicitly replaced by file versioning)
- Authentication on the web UI (local-only assumed)

These are plausible v2 directions but must not influence v1 architecture.

---

## 9. Acceptance Criteria

The system is complete when:

1. A user can create a task through the UI with a plain-language description and 10 example inputs, generate a gold standard, and run an optimization to completion without editing any code or config files manually.
2. All four modes (Autonomous, Supervised, Manual, Visual) are functional end-to-end with matching behavior described in Section 3.3.
3. All three target strategies (Single, Parallel Independent, Unified Portable) are functional end-to-end with behavior matching Section 3.4.
4. An MCP client can invoke `lpo_run_optimization` headlessly and retrieve a winning prompt without human intervention. For multi-model runs, `lpo_get_comparison` returns the cross-model report.
5. A run that crashes mid-execution can be resumed from the last committed iteration with no data loss. For multi-model runs, per-model progress is independently resumable.
6. All three metric types (Deterministic, Rubric, Conversational) produce consistent scores across repeat runs with fixed seeds.
7. The UI renders image outputs correctly and supports batch rating across scenario sets.
8. The UI renders per-model tabs and comparison view for Strategy B runs and per-model breakdowns for Strategy C runs.
9. Seed policies produce the documented determinism: Fixed → bit-identical outputs across runs; Fixed-per-example → identical per input across iterations; Unlocked → natural variance.
10. Cost tracking matches actual API billing within 5% on a 50-iteration test run.
11. Unit tests cover the scoring, engine, multi-engine, and overseer components with > 80% line coverage.
12. A full end-to-end test (provided task fixture → gold standard → optimization → winner) passes in CI for each of the three strategies.

---

## 10. Implementation Sequencing

Suggested build order for incremental validation:

1. Core engine + deterministic scoring + LM Studio client (CLI only, single target) — validate ratchet loop end-to-end on a simple JSON extraction task
2. Overseer integration + conversational context — validate prompt rewriting works
3. Rubric scorer — validate LLM-as-judge consistency
4. Multi-engine layer (Strategies B and C) — validate parallel-independent and unified-portable orchestration against two local models. Strategy A becomes the one-element case of the multi-engine.
5. FastAPI backend + React UI shell — validate live iteration viewing with single and multi-model layouts
6. Mode implementations (Supervised, Manual, Visual) — validate human-in-loop paths
7. MCP server — validate headless invocation for all three strategies including `lpo_get_comparison`
8. Metric evolution — validate overseer-proposed metric changes
9. Seed control — validate determinism guarantees
10. Scenario sets + weighted aggregation — validate per-category and per-model breakdown
11. Cost tracking, resume-on-crash, polish, docs

Each stage is independently testable and independently valuable — if the build runs out of budget at any point, the user still has a working system.

**Rationale for sequencing multi-engine before the UI:** the UI needs to render per-model tabs and comparison views from day one. Building the UI against a single-model engine and retrofitting multi-model support later creates more rework than getting the orchestration layer right before the frontend consumes it. Stage 4 is the architectural inflection point; everything downstream benefits from it being correct.

---

## 11. Reference Concepts

LPO's ratchet loop pattern is inspired by the autoresearch methodology — a fixed-budget iteration loop with keep-or-discard semantics and a single file under modification. LPO generalizes this pattern from ML training research (where the metric is objective: validation loss) to prompt optimization (where the metric may be deterministic, rubric-based, or conversational).

No external repositories need to be referenced or downloaded. The pattern is fully specified in this document.

---

**End of SDP.**
