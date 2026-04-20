"""Render iteration-by-iteration trajectories for both bake-off winners."""
from __future__ import annotations
import json
import pathlib

DET_ROOT = pathlib.Path(r"E:\CascadeProjects\Local Prompt Optimizer\tasks\windows_file_ops_reliability\runs\gemma-4-31b-or\history")
RUB_ROOT = pathlib.Path(r"E:\CascadeProjects\Local Prompt Optimizer\tasks\windows_file_ops_reliability_rubric\runs\qwen3-235b-2507\history")
OUT = pathlib.Path(r"E:\CascadeProjects\Local Prompt Optimizer\tasks\windows_file_ops_reliability_rubric\TRAJECTORIES.md")

TRACK_ID = "ex001"
FENCE = "```"


def short(s: str, n: int = 500) -> str:
    s = s.strip()
    return s if len(s) <= n else s[:n] + "... [truncated]"


def load_iter(it_dir: pathlib.Path):
    prompt = (it_dir / "prompt.txt").read_text(encoding="utf-8", errors="replace").strip()
    scores = json.loads((it_dir / "scores.json").read_text(encoding="utf-8", errors="replace"))
    decision = json.loads((it_dir / "decision.json").read_text(encoding="utf-8", errors="replace"))
    ov = it_dir / "overseer_analysis.md"
    overseer = ov.read_text(encoding="utf-8", errors="replace").strip() if ov.exists() else ""
    tracked = None
    with (it_dir / "outputs.jsonl").open(encoding="utf-8", errors="replace") as f:
        for line in f:
            row = json.loads(line)
            if row.get("id") == TRACK_ID:
                tracked = row
                break
    return prompt, scores, decision, overseer, tracked


def render(root: pathlib.Path, header: str, scoring_kind: str, tracked_input: str) -> str:
    iters = sorted([d for d in root.iterdir() if d.is_dir() and d.name.startswith("iter_")], key=lambda p: p.name)
    L: list[str] = []
    L.append(f"## {header}\n")
    L.append(f"**Tracked example: `{TRACK_ID}`** (scenario: `gui_launch_with_content`)\n")
    L.append(f"**User input:** _{tracked_input}_\n")
    L.append(f"**Scoring:** {scoring_kind}\n")
    L.append(f"**Iterations:** {len(iters)}\n")
    prev_prompt = None
    for it in iters:
        prompt, scores, decision, overseer, tracked = load_iter(it)
        L.append(f"### {it.name}  (aggregate = {scores['aggregate']:.2f})\n")
        reason = (decision.get("reason") or "")[:200]
        L.append(f"- decision: `{decision.get('decision','?')}` — {reason}\n")
        if prompt == prev_prompt:
            L.append("**Prompt:** _(unchanged from previous iter)_\n")
        else:
            L.append("**Prompt fed to target:**\n")
            L.append(FENCE)
            L.append(short(prompt, 2000))
            L.append(FENCE + "\n")
        if tracked is not None:
            L.append(f"**Target output on `{TRACK_ID}`:**\n")
            L.append(FENCE)
            L.append(short(tracked.get("output", ""), 900))
            L.append(FENCE + "\n")
            sc = tracked.get("score", 0)
            pc = tracked.get("per_criterion", {})
            pc_str = ", ".join(f"{k}={float(v):.0f}" for k, v in pc.items())
            L.append(f"**Evaluator score: {sc:.1f}/100** — per-criterion: {pc_str}\n")
            rat = tracked.get("rationale", "") or ""
            if rat:
                L.append("**Evaluator rationale:**")
                for part in rat.split("; "):
                    L.append(f"- {part.strip()}")
                L.append("")
        if overseer:
            L.append("**Overseer analysis (truncated):**\n")
            L.append(FENCE)
            L.append(short(overseer, 1400))
            L.append(FENCE + "\n")
        prev_prompt = prompt
    return "\n".join(L)


# Pull the tracked input (same across both runs — shared eval_set)
tracked_input = ""
with (DET_ROOT / "iter_0001" / "outputs.jsonl").open(encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        if row.get("id") == TRACK_ID:
            tracked_input = row["input"]
            break

parts: list[str] = []
parts.append("# Winner Trajectories — Iteration-by-Iteration Back-and-Forth\n")
parts.append("This document shows how each winning prompt evolved, the actual target-model output on one tracked eval example, and the scorer/judge response per iteration.\n")
parts.append(f"Tracked input: _\"{tracked_input}\"_\n")
parts.append("---\n")
parts.append(render(DET_ROOT,
                    "Deterministic-Scored Bake-off Winner: `gemma-4-31b-or` (final score 81.00, 8 iters)",
                    "Deterministic rules — `json_valid` (30%), `required_fields_present` (40%), `field_exact_match` (30%)",
                    tracked_input))
parts.append("\n---\n")
parts.append(render(RUB_ROOT,
                    "Rubric-Scored Bake-off Winner: `qwen3-235b-2507` (final score 96.10, 6 iters)",
                    "LLM-judge rubric — `command_correctness` (40), `safety_verification` (30), `tense_accuracy` (20), `tool_selection` (10). Judge: `claude-sonnet-4-5`.",
                    tracked_input))

OUT.write_text("\n".join(parts), encoding="utf-8")
print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
