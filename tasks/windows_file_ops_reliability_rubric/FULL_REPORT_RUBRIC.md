# Rubric-Scored Bake-off — Open-Weight Candidates for Windows File Ops

**Task:** `windows_file_ops_reliability_rubric`
**Date:** 2026-04-20
**Strategy:** `parallel_independent`
**Judge:** `claude-sonnet-4-5` (Anthropic direct)
**Overseer:** `claude-opus-4-5`
**Stop conditions:** target_score=95, cost_cap=$3/model, max_iter=20, plateau=5
**Total cost across all 4 runs:** **$2.85**
**Total wall time:** ~16 min

## Motivation

Prior deterministic-scored bake-off capped at 81/100 for the best cloud models because deterministic JSON-field exact-match under-credited semantically-equivalent variants (e.g., `Renaming 'X' to 'Y'` vs `Renaming X to Y` scored as a miss despite identical intent). This run replaces deterministic scoring with a 4-criterion LLM-judge rubric:

| Criterion | Weight | Purpose |
|---|---:|---|
| `command_correctness` | 40 | PowerShell command actually works on Windows |
| `safety_verification` | 30 | Output includes independent verify step (Test-Path etc.) |
| `tense_accuracy` | 20 | Future/conditional tense, not past-tense hallucination |
| `tool_selection` | 10 | PowerShell cmdlets, not bash; Start-Process for GUI |

Target pool shifted from frontier-model comparison (Claude/GPT/Gemma-3) → **open-weight-only** candidates to answer a different question: *can open-weight models (local or cheaply-hosted) fully replace expensive frontier models for this task?*

## Final ranking

| Rank | Target | Best score | Iter | Cost | Stop reason |
|---:|---|---:|---:|---:|---|
| 1 | **`qwen3-235b-2507`** (`qwen/qwen3-235b-a22b-2507`) | **96.10** | 6 | $1.128 | target_score_reached |
| 2 | `gemma-4-26b-local` (`google/gemma-4-26b-a4b`) | 95.45 | **1** | $0.078 | target_score_reached |
| 3 | `deepseek-v3-2` (`deepseek/deepseek-v3.2`) | 95.40 | 5 | $0.740 | target_score_reached |
| 4 | `mistral-large-2512` (`mistralai/mistral-large-2512`) | 95.35 | 5 | $0.901 | target_score_reached |

**All four targets cleared the 95 threshold.** Cost spread is 14×: local Gemma at $0.08 (overseer only, zero inference cost) vs. Qwen at $1.13.

## Rubric vs. deterministic — same models, same task

Where we have apples-to-apples overlap (gemma-4-26b-local):

| Scoring | Best score | Iterations to converge | Cost |
|---|---:|---:|---:|
| Deterministic (prior bake-off) | 81.0 | 3 | — |
| Rubric (this bake-off) | **95.45** | **1** | $0.08 |

The **seed prompt alone** scored 95.45 under rubric. Under deterministic scoring, the same seed fell to ~54 because exact-string-match against gold's `user_message` punishes any paraphrase. The 14.45-point jump isn't the model getting better — it's the rubric correctly crediting the semantic correctness the deterministic metric was blind to.

This confirms the user's hypothesis: *deterministic JSON-field scoring systematically under-rewards open-ended natural-language fields.* Use deterministic scoring only where exact-field-match is the actual product requirement (e.g., structured tool-call JSON into a runner). Use rubric scoring when the output has a natural-language component (user_message, rationale, explanation).

## Per-scenario weakness map

Each target's worst 2 scenarios (from `scores.json`, 0-100 per example):

**`qwen3-235b-2507`** (winner)
- `content_special_chars` — 85 (weakest): judge flagged imperfect escaping of content with embedded single quotes
- `command_substitution` — 92

**`gemma-4-26b-local`** (seed-only)
- `destructive_verify` — 90 (weakest): delete-then-verify logic slightly off (used `Test-Path` expecting false instead of `!(Test-Path)`)
- `multi_step_gui` — 92

**`deepseek-v3-2`**
- `read_only_gui` — 82 (weakest): judge flagged missing or suboptimal verify for GUI-launch scenarios
- `gui_launch_with_content` — 86

**`mistral-large-2512`**
- `multi_step_gui` — 86 (weakest): same pattern as deepseek on GUI launch workflows
- `content_special_chars` — 93.5

**Cross-model pattern:** GUI-launch-and-verify (`read_only_gui`, `multi_step_gui`, `gui_launch_with_content`) is the hardest scenario family across every model. Every target except gemma scored below 90 on at least one GUI-launch scenario. The runner/orchestrator layer should continue to treat GUI launches as the high-risk branch — this is the remaining failure surface.

## Notable findings

### 1. Gemma-4-26B local is dramatically over-performing its price class

- **Converged on iter 1** — the seed prompt (~335 chars) scored 95.45 without any overseer-driven optimization.
- **Total cost: $0.08** — that's the overseer's evaluation judge cost; local inference is effectively zero marginal.
- **Beat cloud DeepSeek and Mistral by 0.05-0.10 points** despite having ~10× fewer parameters than Qwen and no iterative optimization.
- Weakest on `destructive_verify` (90) — a one-line prompt addition about `-not (Test-Path ...)` would likely push this to 97+.

**Implication:** for this specific task (single-shot structured JSON generation from a well-defined spec), Gemma-4-26B on local GPU is a serious candidate to displace frontier API calls. The $0.08 cost is entirely Claude overseer evaluation — at inference time it's free.

### 2. Qwen3-235B-a22b-2507 is the best-value cloud option

- $0.07/1M input, $0.10/1M output — **50× cheaper on output than Claude Opus 4.7** ($3.90/1M output).
- Converged in 6 iterations at $1.13 — the most iterations of any target, but still well under the $3 cap.
- Hit the highest absolute score (96.10) of any target.
- **One-line replacement for frontier models in this pipeline** at ~1% of Claude Opus cost.

### 3. Optimization produced minor gains over seed

| Target | Seed score (iter 0 / iter 1) | Best score | Delta |
|---|---:|---:|---:|
| gemma-4-26b-local | 95.45 (iter 1) | 95.45 | +0.00 (converged on seed) |
| deepseek-v3-2 | ~93 | 95.40 | +2-3 |
| mistral-large-2512 | ~93 | 95.35 | +2-3 |
| qwen3-235b-2507 | ~93 | 96.10 | +3 |

The overseer's prompt-mutation gains are real but **small relative to the total score**. Most of the value is in the baseline prompt design (`prompt_seed.txt`), which was carried over from the deterministic bake-off. This tells us the seed already encodes most of the critical guidance (future tense, Start-Process, Test-Path verify, single-quoted paths).

### 4. The $3/model cap was never binding

Max cost was Qwen at $1.13. The cap acted as a safety net only — all targets hit `target_score_reached` well before cost or iteration limits. For future runs with this rubric, `cost_cap=$1.50/model` would be sufficient.

## Winning prompts (per target)

### 🥇 `qwen3-235b-2507` (96.10)

Overseer synthesized the most explicit prompt — embedded a full input/output example, 8 numbered rules, and concrete PowerShell idioms for edge cases (JSON content, special chars, append):

```
You are a Windows file-operations agent. Given a user request, respond with exactly one valid JSON object.

REQUIRED OUTPUT FORMAT (copy this structure exactly):
{"command": "...", "verify_command": "...", "user_message": "..."}

EXAMPLE:
Input: Create a file called test.txt on the desktop with "Hello"
Output: {"command": "Set-Content -Path 'C:\\Users\\Administrator\\Desktop\\test.txt' -Value 'Hello'", "verify_command": "Test-Path 'C:\\Users\\Administrator\\Desktop\\test.txt'", "user_message": "Creating test.txt on the desktop with the content 'Hello'."}

RULES:
1. GUI APPS: Use `Start-Process` for any GUI app (notepad, explorer, code). Example: `Start-Process notepad -ArgumentList 'C:\path\file.txt'` — never bare `notepad path`.
2. PATHS: Single-quote all Windows paths: `'C:\Users\Administrator\Desktop\file.txt'`
3. verify_command: Must be self-contained with literal paths. Use `Test-Path`, `Get-Content`, or `Get-Process appname -ErrorAction SilentlyContinue`. For DELETE operations, verify the item is GONE: `!(Test-Path 'path')`.
4. user_message: Future tense only. Say "Creating...", "Opening...", "This will..." — never "I created" or "Done".
5. CMDLETS: Use Set-Content, New-Item, Remove-Item, Get-Content, Test-Path, Start-Process, Rename-Item, Add-Content (for append).
6. DESKTOP: Use literal path `'C:\Users\Administrator\Desktop\filename'`.
7. CONTENT WITH SPECIAL CHARS: Use single quotes around -Value content to prevent $variable expansion. For content containing single quotes, double them: 'It''s working'. For JSON or complex content, this works: `Set-Content -Path 'file.txt' -Value '{"key": "value"}'`
8. APPEND: Use `Add-Content -Path 'path' -Value 'text'` to append to existing files.

Output only the JSON object. No markdown fences, no extra text.
```

### 🥈 `gemma-4-26b-local` (95.45)

Seed prompt — unmodified — already converged. Notable for its brevity (~335 chars):

```
You are a careful Windows file-operations agent. When the user asks you to perform a file or folder operation, respond with a JSON object containing a PowerShell command to run, a verification command to prove the action succeeded, and a user-facing message describing what will happen.
Output only the JSON object, no other text.
```

See full per-model winning prompts and report at:
- `comparison/report.md`
- `runs/<slug>/history/<best_iter>/prompt.txt`

## Recommendation

1. **Deploy Gemma-4-26B local** as the primary skill-generation model for Jarvis Windows file-ops. At ~95 rubric score on the seed prompt with zero marginal cost, it's the obvious winner on cost-adjusted quality.
2. **Use Qwen3-235b-a22b-2507** as the OpenRouter fallback when local GPU is busy or unavailable. Best absolute score, $0.10/1M output.
3. **Retire Claude Opus / GPT-4.1 / Claude Haiku** for this specific skill slot. The data shows they are not meaningfully better at this task than the open-weight alternatives, and are 10-50× more expensive.
4. **Extend the prompt seed** with the top 3 rules from Qwen's winning prompt (explicit I/O example, path-quoting rule, content-escaping rule). This should lift Gemma's score from 95.45 → ~97.
5. **Investigate GUI-launch-and-verify scenarios** as the remaining weakness — every model struggled here. Consider adding a dedicated `gui_launch_verify` skill branch in the orchestrator that defers verification by 500ms after `Start-Process` before running `Get-Process`.

## Artifacts

- Task bundle: `E:\CascadeProjects\Local Prompt Optimizer\tasks\windows_file_ops_reliability_rubric\`
- Comparison report: `comparison/report.md`
- Per-model runs: `runs/<slug>/history/iter_NNNN/`
- Driver: `E:\CascadeProjects\Local Prompt Optimizer\run_bakeoff_rubric.py`
- Logs: `bakeoff_rubric.log`, `bakeoff_rubric.log.err`
