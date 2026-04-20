"""OpenRouter dynamic pricing.

OpenRouter exposes per-model pricing via ``GET /api/v1/models``. Each model
entry includes a ``pricing`` object whose ``prompt`` and ``completion`` fields
are USD **per token** (as strings). We fetch the full model list once per
process, cache it, and convert to the per-1M-token rate unit the rest of LPO
uses via :class:`lpo.core.cost.CostTracker`.

This is a best-effort integration: if the pricing endpoint is unreachable or
returns a malformed payload, unknown models score at ``(0.0, 0.0)`` and a
warning is logged once. The ratchet run continues — we'd rather under-report
cost than crash a long optimization.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

log = logging.getLogger("lpo.openrouter.pricing")

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

# Rate unit expected by CostTracker: (input_per_mtok, output_per_mtok).
RatePair = tuple[float, float]


def _parse_rate(price_field: Any) -> float:
    """OpenRouter returns prices as strings in USD per token. Convert to
    USD per 1M tokens. Missing / unparseable values collapse to 0."""
    try:
        return float(price_field) * 1_000_000.0
    except (TypeError, ValueError):
        return 0.0


def parse_models_payload(payload: dict) -> dict[str, RatePair]:
    """Extract ``{model_id: (in_per_mtok, out_per_mtok)}`` from the raw
    ``/api/v1/models`` JSON body.

    Defensive: missing fields yield ``(0.0, 0.0)`` rather than raising, so a
    partial payload still produces a usable table.
    """
    rates: dict[str, RatePair] = {}
    data = payload.get("data") or []
    if not isinstance(data, list):
        return rates
    for entry in data:
        if not isinstance(entry, dict):
            continue
        model_id = entry.get("id")
        if not isinstance(model_id, str) or not model_id:
            continue
        pricing = entry.get("pricing") or {}
        in_rate = _parse_rate(pricing.get("prompt"))
        out_rate = _parse_rate(pricing.get("completion"))
        rates[model_id] = (in_rate, out_rate)
    return rates


class OpenRouterPricing:
    """Process-shared pricing cache.

    Use :func:`get_shared_pricing` to obtain the singleton. Tests construct
    their own instance and inject a fake ``fetcher`` callable.
    """

    def __init__(
        self,
        *,
        fetcher: "PricingFetcher | None" = None,
        url: str = OPENROUTER_MODELS_URL,
        timeout_s: float = 15.0,
    ) -> None:
        self._fetcher = fetcher or _default_fetcher(url, timeout_s)
        self._rates: dict[str, RatePair] | None = None
        self._lock = asyncio.Lock()
        self._failed = False

    def is_loaded(self) -> bool:
        return self._rates is not None

    def rates_snapshot(self) -> dict[str, RatePair]:
        """Return the current cache without triggering a fetch. Useful for
        debugging and for the smoke-test prelude."""
        return dict(self._rates or {})

    async def ensure_loaded(self) -> None:
        """Fetch and cache the model list on first call. Safe to call from
        many concurrent tasks — only one fetch actually runs."""
        if self._rates is not None or self._failed:
            return
        async with self._lock:
            if self._rates is not None or self._failed:
                return
            try:
                payload = await self._fetcher()
                self._rates = parse_models_payload(payload)
                log.info(
                    "OpenRouter pricing loaded for %d models.", len(self._rates)
                )
            except Exception as e:  # noqa: BLE001
                self._failed = True
                self._rates = {}
                log.warning(
                    "Failed to fetch OpenRouter pricing (%s). All OpenRouter "
                    "model calls will be recorded at $0 cost until restart.", e,
                )

    async def rate_for(self, model_id: str) -> RatePair:
        """Per-model lookup. Returns ``(0.0, 0.0)`` for unknown ids.

        Unknown ids are expected when the user specifies a model that has
        since been delisted, renamed, or isn't reachable through OpenRouter.
        """
        await self.ensure_loaded()
        return (self._rates or {}).get(model_id, (0.0, 0.0))


# Async callable: () -> payload dict. Pluggable for tests.
from typing import Awaitable, Callable  # noqa: E402 - keeps public API block at top

PricingFetcher = Callable[[], Awaitable[dict]]


def _default_fetcher(url: str, timeout_s: float) -> PricingFetcher:
    async def _fetch() -> dict:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.json()

    return _fetch


# ---------------------------------------------------------------------------
# Process-wide singleton
# ---------------------------------------------------------------------------

_SHARED: OpenRouterPricing | None = None


def get_shared_pricing() -> OpenRouterPricing:
    """Return the process-wide pricing cache, creating it on first call."""
    global _SHARED
    if _SHARED is None:
        _SHARED = OpenRouterPricing()
    return _SHARED


def reset_shared_pricing_for_tests() -> None:
    """Test helper: drop the shared singleton so the next ``get_shared_pricing``
    call returns a fresh instance."""
    global _SHARED
    _SHARED = None
