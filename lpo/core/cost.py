"""Cost tracking for billable model calls. See `LPO_SDP.md` §4.9, §6.2.

Prices are USD per 1M tokens. Rates default to Claude Opus 4.x public pricing
and can be overridden per-model via ``CostTracker.set_rate``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock

# Default prices (USD per 1M tokens). Kept conservative and easy to update.
DEFAULT_RATES: dict[str, tuple[float, float]] = {
    # model_id prefix: (input_per_mtok, output_per_mtok)
    "claude-opus-4": (15.0, 75.0),
    "claude-opus-4-7": (15.0, 75.0),
    "claude-sonnet-4": (3.0, 15.0),
    "claude-haiku-4": (0.8, 4.0),
    "claude-3-5-sonnet": (3.0, 15.0),
    "claude-3-5-haiku": (0.8, 4.0),
}


@dataclass
class CallCost:
    model_id: str
    prompt_tokens: int
    completion_tokens: int
    usd: float


@dataclass
class CostTracker:
    """Thread-safe running total of billable API costs for a single run."""

    total_usd: float = 0.0
    calls: list[CallCost] = field(default_factory=list)
    _lock: Lock = field(default_factory=Lock, repr=False)
    _rates: dict[str, tuple[float, float]] = field(default_factory=lambda: dict(DEFAULT_RATES))

    def set_rate(self, model_id_prefix: str, input_per_mtok: float, output_per_mtok: float) -> None:
        with self._lock:
            self._rates[model_id_prefix] = (input_per_mtok, output_per_mtok)

    def _lookup_rate(self, model_id: str) -> tuple[float, float]:
        # Longest-prefix match so 'claude-3-5-sonnet-20241022' picks up '-3-5-sonnet'.
        best: tuple[float, float] | None = None
        best_len = -1
        for prefix, rate in self._rates.items():
            if model_id.startswith(prefix) and len(prefix) > best_len:
                best = rate
                best_len = len(prefix)
        return best if best is not None else (0.0, 0.0)

    def record(self, model_id: str, prompt_tokens: int, completion_tokens: int) -> CallCost:
        in_rate, out_rate = self._lookup_rate(model_id)
        usd = (prompt_tokens * in_rate + completion_tokens * out_rate) / 1_000_000.0
        call = CallCost(
            model_id=model_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            usd=usd,
        )
        with self._lock:
            self.calls.append(call)
            self.total_usd += usd
        return call

    def over_cap(self, cap_usd: float) -> bool:
        return self.total_usd >= cap_usd
