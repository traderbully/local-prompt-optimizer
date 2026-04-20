"""Iteration persistence (filesystem only — no DB). See `LPO_SDP.md` §4.1, §5.5, §6.1."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Atomic IO
# ---------------------------------------------------------------------------


def atomic_write_text(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    """Write text to ``path`` atomically: tmp-in-dir → fsync → os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".{}_".format(path.name), dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="\n") as f:
            f.write(content)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                # fsync may be unavailable on some filesystems; best-effort.
                pass
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def atomic_write_json(path: Path, data: Any, *, indent: int = 2) -> None:
    atomic_write_text(path, json.dumps(data, indent=indent, ensure_ascii=False, default=str) + "\n")


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str))
        f.write("\n")


# ---------------------------------------------------------------------------
# Run layout
# ---------------------------------------------------------------------------


@dataclass
class RunPaths:
    """Paths for a single target-model run. See `LPO_SDP.md` §4.1."""

    task_root: Path
    model_slug: str

    @property
    def run_root(self) -> Path:
        return self.task_root / "runs" / self.model_slug

    @property
    def current_prompt(self) -> Path:
        return self.run_root / "prompt.txt"

    @property
    def best_prompt(self) -> Path:
        return self.run_root / "prompt.txt.best"

    @property
    def history_root(self) -> Path:
        return self.run_root / "history"

    @property
    def winner_root(self) -> Path:
        return self.run_root / "winner"

    @property
    def log_file(self) -> Path:
        return self.task_root / "logs" / f"{self.model_slug}.jsonl"

    def iteration_dir(self, index: int) -> Path:
        return self.history_root / f"iter_{index:04d}"

    def ensure(self) -> None:
        for p in (self.run_root, self.history_root, self.log_file.parent):
            p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Iteration record
# ---------------------------------------------------------------------------


@dataclass
class IterationRecord:
    index: int
    prompt: str
    aggregate_score: float
    per_example: dict[str, float] = field(default_factory=dict)
    per_scenario: dict[str, float] = field(default_factory=dict)
    failed_ids: list[str] = field(default_factory=list)
    outputs: list[dict[str, Any]] = field(default_factory=list)
    decision: str = "pending"  # accepted | rejected | initial
    delta: float = 0.0
    timings: dict[str, float] = field(default_factory=dict)
    cost_usd: float = 0.0
    notes: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def write(self, paths: RunPaths) -> Path:
        d = paths.iteration_dir(self.index)
        d.mkdir(parents=True, exist_ok=True)
        atomic_write_text(d / "prompt.txt", self.prompt)
        # outputs.jsonl — one line per eval example
        out_path = d / "outputs.jsonl"
        if out_path.exists():
            out_path.unlink()
        for row in self.outputs:
            append_jsonl(out_path, row)
        atomic_write_json(
            d / "scores.json",
            {
                "aggregate": self.aggregate_score,
                "per_example": self.per_example,
                "per_scenario": self.per_scenario,
                "failed_ids": self.failed_ids,
            },
        )
        atomic_write_json(
            d / "decision.json",
            {
                "decision": self.decision,
                "delta": self.delta,
                "timings": self.timings,
                "cost_usd": self.cost_usd,
                "timestamp": self.timestamp,
                "notes": self.notes,
            },
        )
        return d
