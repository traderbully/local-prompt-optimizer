"""Cost tracking for billable model calls. See `LPO_SDP.md` §4.9, §6.2.

Prices are USD per 1M tokens (public API base-input / base-output rates).
Rates are matched against ``model_id`` with a longest-prefix search so that
a specific snapshot ('claude-opus-4-5-20251101') picks up the right
generation-specific rate before falling back to the family prefix.
Overridable per-tracker via :meth:`CostTracker.set_rate`.

Source: Anthropic pricing page (checked 2026-04-21).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock

# Default prices (USD per 1M tokens). Longest-prefix match; keep more-specific
# entries *in addition to* family prefixes so future snapshots inherit the
# right rate without silent regressions.
DEFAULT_RATES: dict[str, tuple[float, float]] = {
    # model_id prefix: (input_per_mtok, output_per_mtok)
    # ---- Opus 4.x family -------------------------------------------------
    # Opus 4 and 4.1 remain on the original $15/$75 tier.
    "claude-opus-4": (15.0, 75.0),
    # Opus 4.5 / 4.6 / 4.7 moved to a cheaper $5/$25 tier. The longer prefix
    # wins over "claude-opus-4" for these generations.
    "claude-opus-4-5": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    # ---- Sonnet 4.x family (all tiers priced uniformly) ------------------
    "claude-sonnet-4": (3.0, 15.0),
    # ---- Haiku 4.x family ------------------------------------------------
    # Haiku 4.5 is priced higher than Haiku 3.5 ($1/$5 vs $0.80/$4). Family
    # prefix pins the 4.x rate; the explicit 4-5 prefix is redundant today
    # but documents intent for future snapshots.
    "claude-haiku-4": (1.0, 5.0),
    "claude-haiku-4-5": (1.0, 5.0),
    # ---- Legacy 3.5 family ----------------------------------------------
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
