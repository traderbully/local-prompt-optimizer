"""Task bundle load/save. See `LPO_SDP.md` §4.1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lpo.config.schema import (
    EvalRecord,
    GoldRecord,
    MetricConfig,
    RunConfig,
    load_eval_set,
    load_gold_standard,
    load_metric_config,
    load_run_config,
)


@dataclass
class TaskBundle:
    root: Path
    config: RunConfig
    metric: MetricConfig
    eval_records: list[EvalRecord]
    gold_standard: dict[str, GoldRecord]
    seed_prompt: str
    task_md: str = ""

    @classmethod
    def load(cls, root: Path) -> "TaskBundle":
        root = Path(root).resolve()
        if not root.exists():
            raise FileNotFoundError(f"Task directory not found: {root}")
        config_path = root / "config.yaml"
        metric_path = root / "metric.yaml"
        eval_path = root / "eval_set.jsonl"
        gold_path = root / "gold_standard.jsonl"
        seed_prompt_path = root / "prompt_seed.txt"
        task_md_path = root / "task.md"

        for required in (config_path, metric_path, eval_path):
            if not required.exists():
                raise FileNotFoundError(f"Missing required task file: {required}")

        config = load_run_config(config_path)
        metric = load_metric_config(metric_path)
        eval_records = load_eval_set(eval_path)
        gold = load_gold_standard(gold_path)

        seed_prompt = ""
        if seed_prompt_path.exists():
            seed_prompt = seed_prompt_path.read_text(encoding="utf-8")
        task_md = ""
        if task_md_path.exists():
            task_md = task_md_path.read_text(encoding="utf-8")

        return cls(
            root=root,
            config=config,
            metric=metric,
            eval_records=eval_records,
            gold_standard=gold,
            seed_prompt=seed_prompt,
            task_md=task_md,
        )
