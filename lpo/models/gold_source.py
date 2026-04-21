"""Pluggable Gold Standard Source.

Per ``LPO_SDP.md`` §3.1, the Gold Standard must be frontier-quality — it
defines the reference output every iteration scores against. Historically
LPO hardcoded this to direct Anthropic billing. That's a barrier for
anyone who already has healthy OpenRouter credits but no standalone
Anthropic billing relationship.

This module exposes :func:`build_gold_standard_source`, a factory that
reads env-var configuration and returns a client implementing the narrow
``_GoldSource`` protocol used by :func:`lpo.core.authoring.generate_gold_standard`:

* ``async def complete(system, messages, temperature, max_tokens) -> result``
* ``async def aclose() -> None``
* ``result.text`` — the generated string

Supported providers (via ``GOLD_STANDARD_PROVIDER``):

* ``anthropic`` (default) — direct Anthropic, identical to the legacy path.
* ``openrouter`` — route through OpenRouter's OpenAI-compatible API so any
  ``anthropic/claude-*`` or comparable frontier model reachable via
  OpenRouter can serve as the ground truth.

Env vars:

    GOLD_STANDARD_PROVIDER   anthropic | openrouter     (default: anthropic)
    GOLD_STANDARD_MODEL      model id override          (default: None — caller supplies)
    GOLD_STANDARD_BASE_URL   OpenRouter base URL        (default: https://openrouter.ai/api/v1)
    GOLD_STANDARD_API_KEY_ENV  name of env var holding  (default: provider-dependent)
                               the API key

This keeps the surface small. The Overseer itself remains Anthropic-only
for now — the SDP fixes that binding in §3.1 — but the Gold Standard is
a one-shot role where cost and billing flexibility actually matter.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol

from lpo.core.cost import CostTracker
from lpo.models.base import ModelError


class GoldSource(Protocol):
    """Narrow interface that :func:`generate_gold_standard` depends on."""

    async def complete(self, *, system: str, messages: list[Any], **kw: Any) -> Any: ...
    async def aclose(self) -> None: ...


@dataclass
class _GoldResult:
    """Minimal result shape matching :class:`AnthropicResult`.text access."""
    text: str


class _OpenRouterGoldAdapter:
    """Adapts :class:`OpenRouterClient` to the :class:`GoldSource` interface.

    OpenRouterClient speaks the ``ModelClient.generate(system_prompt,
    user_input, ...)`` shape. The authoring layer speaks Anthropic-style
    ``.complete(system, messages=[{role, content}], ...)``. This adapter
    translates once per call, extracting the single user message (the only
    shape :func:`generate_gold_standard` ever emits).
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    async def complete(
        self,
        *,
        system: str,
        messages: list[Any],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        **_: Any,
    ) -> _GoldResult:
        # The gold-standard caller always supplies exactly one user message.
        if not messages:
            raise ModelError("GoldSource.complete called with no messages.")
        last = messages[-1]
        # Support both dataclass-style (AnthropicMessage) and dict-style.
        user_text = getattr(last, "content", None)
        if user_text is None and isinstance(last, dict):
            user_text = last.get("content", "")
        if not isinstance(user_text, str):
            user_text = str(user_text)

        gen = await self._inner.generate(
            system,
            user_text,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return _GoldResult(text=gen.text or "")

    async def aclose(self) -> None:
        close = getattr(self._inner, "aclose", None)
        if close is not None:
            await close()


def build_gold_standard_source(
    *,
    model_id: str,
    cost_tracker: CostTracker | None = None,
) -> GoldSource:
    """Return a gold-standard source chosen by env configuration.

    ``model_id`` is the caller's preferred model; if the operator has set
    ``GOLD_STANDARD_MODEL`` in the environment, that override wins. This
    matches the SDP principle that authoring decisions flow from config,
    not source code.
    """
    provider = os.environ.get("GOLD_STANDARD_PROVIDER", "anthropic").strip().lower()
    effective_model = os.environ.get("GOLD_STANDARD_MODEL") or model_id

    if provider == "anthropic":
        # Lazy-import to keep the anthropic SDK off the critical path of
        # OpenRouter-only deployments.
        from lpo.models.anthropic_client import AnthropicClient

        api_key_env = os.environ.get("GOLD_STANDARD_API_KEY_ENV", "ANTHROPIC_API_KEY")
        return AnthropicClient(
            model_id=effective_model,
            api_key_env=api_key_env,
            cost_tracker=cost_tracker,
        )

    if provider == "openrouter":
        from lpo.models.openrouter import DEFAULT_BASE_URL, OpenRouterClient

        base_url = os.environ.get("GOLD_STANDARD_BASE_URL", DEFAULT_BASE_URL)
        api_key_env = os.environ.get("GOLD_STANDARD_API_KEY_ENV", "OPENROUTER_API_KEY")
        inner = OpenRouterClient(
            model_id=effective_model,
            base_url=base_url,
            api_key_env=api_key_env,
            cost_tracker=cost_tracker,
        )
        return _OpenRouterGoldAdapter(inner)

    raise ModelError(
        f"GOLD_STANDARD_PROVIDER={provider!r} not supported. "
        "Valid values: 'anthropic' (default), 'openrouter'."
    )
