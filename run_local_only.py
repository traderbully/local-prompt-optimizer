"""Run optimization only for gemma-4-26b-local (the LM Studio target).

The other three targets already have completed runs on disk; we only need to
fill in the missing fourth. Uses RatchetEngine directly so we don't re-run
any of the cloud targets.
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
TASK_PATH = LPO_ROOT / "tasks" / "windows_file_ops_reliability"
TARGET_SLUG = "gemma-4-26b-local"

load_dotenv(LPO_ROOT / ".env", override=True)
sys.path.insert(0, str(LPO_ROOT))

from lpo.core.cost import CostTracker  # noqa: E402
from lpo.core.engine import RatchetEngine  # noqa: E402
from lpo.core.target_factory import build_target_context  # noqa: E402
from lpo.core.task import TaskBundle  # noqa: E402


async def main() -> int:
    task = TaskBundle.load(TASK_PATH)
    target = next(m for m in task.config.target_models if m.slug == TARGET_SLUG)
    print(f"[local] target: {target.slug} ({target.provider}/{target.model_id})", flush=True)
    print(f"[local] base_url: {target.base_url}", flush=True)

    # Wipe any prior local-gemma artifacts so this run starts fresh.
    import shutil
    run_dir = task.root / "runs" / TARGET_SLUG
    if run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)
        print(f"[local] cleared prior artifacts at {run_dir}", flush=True)

    cost = CostTracker()
    ctx = build_target_context(task, target, cost, mutator_mode="auto")

    try:
        engine = RatchetEngine(
            task=task,
            target_cfg=target,
            client=ctx.client,
            scorer=ctx.scorer,
            mutator=ctx.mutator,
            cost_tracker=cost,
        )
        result = await engine.run()
    finally:
        await ctx.aclose()

    print(f"[local] DONE", flush=True)
    print(f"[local] best_score: {result.best_score:.2f}", flush=True)
    print(f"[local] iterations: {len(result.iterations)}", flush=True)
    print(f"[local] stop_reason: {result.stop_reason.value}", flush=True)
    print(f"[local] cost_usd: ${result.total_cost_usd:.4f}", flush=True)
    print("LOCAL_RESULT_JSON=" + json.dumps({
        "slug": TARGET_SLUG,
        "best_score": round(result.best_score, 2),
        "iterations": len(result.iterations),
        "stop_reason": result.stop_reason.value,
        "cost_usd": round(result.total_cost_usd, 4),
    }), flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except Exception:
        traceback.print_exc()
        sys.exit(1)
