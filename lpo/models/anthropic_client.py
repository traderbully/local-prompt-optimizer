"""Anthropic async client used as Overseer + Gold Standard Source.

See `LPO_SDP.md` §3.1, §4.9. Wraps `anthropic.AsyncAnthropic` with a minimal
message-completion surface and surfaces token usage so the shared ``CostTracker``
can meter billable calls.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from anthropic import APIError, AsyncAnthropic

from lpo.core.cost import CostTracker
from lpo.models.base import ModelError


@dataclass
class AnthropicMessage:
    role: str  # "user" | "assistant"
    content: str


@dataclass
class AnthropicResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    model_id: str
    stop_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


class AnthropicClient:
    """Thin async wrapper around Anthropic's messages API."""

    def __init__(
        self,
        *,
        model_id: str,
        api_key_env: str = "ANTHROPIC_API_KEY",
        max_tokens: int = 4096,
        timeout_s: float = 120.0,
        cost_tracker: CostTracker | None = None,
    ) -> None:
        key = os.environ.get(api_key_env)
        if not key:
            raise ModelError(
                f"Anthropic API key not found in env var {api_key_env!r}. "
                "Copy .env.example to .env and set ANTHROPIC_API_KEY."
            )
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.cost = cost_tracker or CostTracker()
        self._client = AsyncAnthropic(api_key=key, timeout=timeout_s)
        # Newer Anthropic models (e.g. claude-opus-4-7) reject ``temperature``.
        # We discover that lazily on the first call and cache the outcome.
        self._send_temperature: bool = True

    async def aclose(self) -> None:
        # AsyncAnthropic exposes .close() on newer versions; guard for older.
        close = getattr(self._client, "close", None)
        if close is not None:
            res = close()
            if hasattr(res, "__await__"):
                await res

    async def complete(
        self,
        *,
        system: str,
        messages: list[AnthropicMessage],
        temperature: float = 0.3,
        max_tokens: int | None = None,
    ) -> AnthropicResult:
        payload_messages = [{"role": m.role, "content": m.content} for m in messages]

        def _kwargs(send_temp: bool) -> dict[str, Any]:
            kw: dict[str, Any] = {
                "model": self.model_id,
                "system": system,
                "messages": payload_messages,
                "max_tokens": max_tokens or self.max_tokens,
            }
            if send_temp:
                kw["temperature"] = temperature
            return kw

        sent_temp = self._send_temperature
        try:
            resp = await self._client.messages.create(**_kwargs(sent_temp))
        except APIError as e:
            # Retry without ``temperature`` if:
            #   (a) this call sent it and the server said it's deprecated, OR
            #   (b) a concurrent call already flipped the flag to False but
            #       *this* call had already snapshotted sent_temp=True before
            #       the flip. Either way, retrying unconditionally is safe
            #       because we never retry the retry.
            if sent_temp and _is_deprecated_temperature_error(e):
                self._send_temperature = False
                try:
                    resp = await self._client.messages.create(**_kwargs(False))
                except APIError as e2:
                    raise ModelError(f"Anthropic API error: {e2}") from e2
            else:
                raise ModelError(f"Anthropic API error: {e}") from e

        # Concatenate text blocks; tool-use blocks are ignored for Stage 2.
        parts: list[str] = []
        for block in resp.content:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        text = "".join(parts)
        usage = getattr(resp, "usage", None)
        prompt_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        self.cost.record(self.model_id, prompt_tokens, completion_tokens)

        return AnthropicResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model_id=self.model_id,
            stop_reason=getattr(resp, "stop_reason", None),
            raw=resp.model_dump() if hasattr(resp, "model_dump") else {},
        )


def _is_deprecated_temperature_error(err: APIError) -> bool:
    """Detect Anthropic's 'temperature is deprecated for this model' 400.

    Newer models reject the parameter outright; the API returns a 400 whose
    body contains the phrase ``temperature`` and ``deprecated``.
    """
    msg = str(err).lower()
    return "temperature" in msg and "deprecated" in msg

