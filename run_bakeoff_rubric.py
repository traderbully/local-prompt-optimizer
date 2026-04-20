"""Headless driver for the rubric-scored windows_file_ops_reliability_rubric bake-off.

Same invocation pattern as run_bakeoff.py — avoids MCP transport quirks,
applies load_dotenv(override=True) up front, runs run_multi, writes comparison
report.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import traceback
from pathlib import Path

from dotenv import load_dotenv

LPO_ROOT = Path(r"E:\CascadeProjects\Local Prompt Optimizer")
TASK_PATH = LPO_ROOT / "tasks" / "windows_file_ops_reliability_rubric"

load_dotenv(LPO_ROOT / ".env", override=True)

_key = os.environ.get("ANTHROPIC_API_KEY", "")
if _key:
    print(
        f"[driver] ANTHROPIC_API_KEY fingerprint: {_key[:4]}...{_key[-4:]} len={len(_key)}",
        flush=True,
    )
else:
    print("[driver] ANTHROPIC_API_KEY not set", flush=True)
    sys.exit(2)

sys.path.insert(0, str(LPO_ROOT))

from lpo.core.comparison import write_comparison_report  # noqa: E402
from lpo.core.cost import CostTracker  # noqa: E402
from lpo.core.multi_engine import run_multi, validate_runtime  # noqa: E402
from lpo.core.task import TaskBundle  # noqa: E402


async def main() -> int:
    print(f"[driver] task bundle: {TASK_PATH}", flush=True)
    task = TaskBundle.load(TASK_PATH)
    validate_runtime(task.config)

    print(f"[driver] metric type: {task.metric.type}", flush=True)
    print(f"[driver] strategy: {task.config.target_strategy}", flush=True)
    print(f"[driver] cost_cap_usd per model: {task.config.stop_conditions.cost_cap_usd}", flush=True)
    for m in task.config.target_models:
        print(f"[driver]   target: {m.slug} ({m.provider}/{m.model_id})", flush=True)

    gold_path = TASK_PATH / "gold_standard.jsonl"
    if not gold_path.exists():
        print("[driver] gold_standard.jsonl missing — rubric can run without it but we expected one from original bundle", flush=True)

    cost = CostTracker()
    result = await run_multi(task, cost=cost, mutator_mode="auto")

    summary_path, report_path = write_comparison_report(
        task.root, task.config.task_name, result
    )
    print(f"[driver] summary: {summary_path}", flush=True)
    print(f"[driver] report:  {report_path}", flush=True)

    per_model = [
        {
            "slug": r.slug,
            "best_score": round(r.best_score, 2),
            "iterations": r.iterations,
            "stop_reason": r.stop_reason.value,
            "cost_usd": round(r.cost_usd, 4),
        }
        for r in sorted(result.per_model, key=lambda x: x.best_score, reverse=True)
    ]
    print("DRIVER_RESULT_JSON=" + json.dumps({
        "strategy": result.strategy,
        "per_model": per_model,
        "total_cost_usd": round(result.total_cost_usd, 4),
        "summary_path": str(summary_path),
        "report_path": str(report_path),
    }), flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except Exception:
        traceback.print_exc()
        sys.exit(1)
