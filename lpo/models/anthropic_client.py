"""Anthropic async client used as Overseer + Gold Standard Source.

See `LPO_SDP.md` §3.1, §4.9. Wraps `anthropic.AsyncAnthropic` with a minimal
message-completion surface and surfaces token usage so the shared ``CostTracker``
can meter billable calls.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from anthropic import APIError, AsyncAnthropic

log = logging.getLogger("lpo.models.anthropic_client")

try:
    # dotenv is a runtime dep (pulled in by lpo.cli). Best-effort import so
    # non-CLI contexts (tests, direct imports) still work even if it's absent.
    from dotenv import dotenv_values, find_dotenv
except Exception:  # noqa: BLE001
    dotenv_values = None  # type: ignore[assignment]
    find_dotenv = None  # type: ignore[assignment]

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
        timeout_s: float = 90.0,
        max_retries: int = 2,
        cost_tracker: CostTracker | None = None,
    ) -> None:
        key = os.environ.get(api_key_env)
        if not key:
            raise ModelError(
                f"Anthropic API key not found in env var {api_key_env!r}. "
                "Copy .env.example to .env and set ANTHROPIC_API_KEY."
            )
        if timeout_s <= 0:
            raise ValueError(f"timeout_s must be positive, got {timeout_s!r}")
        if max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {max_retries!r}")
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.cost = cost_tracker or CostTracker()
        self._api_key_env = api_key_env
        self._key_snapshot = key
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        # Disable the SDK's own retry loop — we enforce wall-clock timeout via
        # asyncio.wait_for at the call site and handle retries ourselves. If
        # we left the SDK default (max_retries=2), a single stuck connection
        # could silently burn ``timeout_s * 3`` before surfacing, making the
        # configured timeout effectively useless. See the 2026-04-21 overseer
        # hang incident (>8 min blocked on a single messages.create call).
        self._client = AsyncAnthropic(api_key=key, timeout=timeout_s, max_retries=0)
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
            resp = await self._call_with_timeout_and_retry(_kwargs(sent_temp))
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
                    resp = await self._call_with_timeout_and_retry(_kwargs(False))
                except APIError as e2:
                    raise self._wrap_api_error(e2) from e2
            else:
                raise self._wrap_api_error(e) from e

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
        # Cost is billed against the requested model id (that's what pricing
        # tables are keyed on). Observability / provenance uses the served
        # model id (see below). These are usually the same string; they can
        # diverge when the requested id is an alias that Anthropic resolves
        # to a concrete version (e.g. "claude-opus-4-5" -> "claude-sonnet-4-
        # 20250514" during a redirect window). Conflating the two meant the
        # Stage-8 postmortem surfaced contradictory analyst_model_id values
        # between the top-level envelope and the diagnosis.json payload.
        self.cost.record(self.model_id, prompt_tokens, completion_tokens)

        served_model_id = getattr(resp, "model", None) or self.model_id

        return AnthropicResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model_id=served_model_id,
            stop_reason=getattr(resp, "stop_reason", None),
            raw=resp.model_dump() if hasattr(resp, "model_dump") else {},
        )

    # ------------------------------------------------------------------ timeout + retry

    async def _call_with_timeout_and_retry(self, kwargs: dict[str, Any]) -> Any:
        """Call ``messages.create`` with wall-clock timeout + bounded retry.

        Wraps each SDK call in ``asyncio.wait_for(..., timeout=self._timeout_s)``
        so that a stuck TCP connection, a broken SDK timer, or any other form
        of in-SDK hang cannot block the caller for longer than
        ``self._timeout_s`` per attempt.

        On :class:`asyncio.TimeoutError` we retry up to ``self._max_retries``
        additional times with exponential backoff (1s, 2s, 4s, capped at 8s).
        All timeout attempts exhausted -> raise :class:`ModelError` with a
        diagnostic message. :class:`APIError` is passed through unchanged so
        the caller's existing auth/deprecated-temperature handling still works.
        """
        last_timeout: asyncio.TimeoutError | None = None
        total_attempts = self._max_retries + 1
        for attempt in range(total_attempts):
            try:
                return await asyncio.wait_for(
                    self._client.messages.create(**kwargs),
                    timeout=self._timeout_s,
                )
            except asyncio.TimeoutError as e:
                last_timeout = e
                if attempt + 1 < total_attempts:
                    backoff = min(2.0 ** attempt, 8.0)
                    log.warning(
                        "Anthropic call to %s timed out after %.0fs "
                        "(attempt %d/%d); retrying in %.1fs",
                        self.model_id, self._timeout_s,
                        attempt + 1, total_attempts, backoff,
                    )
                    await asyncio.sleep(backoff)
        # All attempts timed out.
        raise ModelError(
            f"Anthropic call to {self.model_id!r} timed out after "
            f"{self._timeout_s:.0f}s across {total_attempts} attempt(s). "
            "The SDK call did not return within the wall-clock budget. "
            "Increase timeout_s or investigate network/Anthropic availability."
        ) from last_timeout

    # ------------------------------------------------------------------ errors

    def _wrap_api_error(self, err: APIError) -> ModelError:
        """Produce a :class:`ModelError` with extra diagnostic context for
        auth failures. Non-auth errors fall through to the legacy message."""
        if not _is_auth_error(err):
            return ModelError(f"Anthropic API error: {err}")

        # Build a diagnostic block. Never include the raw key — only
        # obfuscated fingerprints.
        lines = [
            "Anthropic authentication failed (401 invalid x-api-key).",
            f"  api_key_env       : {self._api_key_env}",
            f"  key fingerprint   : {_fp(self._key_snapshot)}",
        ]
        current_env = os.environ.get(self._api_key_env)
        if current_env != self._key_snapshot:
            lines.append(
                f"  key drift         : os.environ currently has a DIFFERENT "
                f"value ({_fp(current_env)}) than this client was built with."
            )

        dotenv_val = _dotenv_value(self._api_key_env)
        if dotenv_val is not None:
            if dotenv_val == self._key_snapshot:
                lines.append(f"  .env file         : matches client key ({_fp(dotenv_val)})")
            else:
                lines.append(
                    f"  .env file         : DIFFERS from client key — "
                    f"shell/env: {_fp(self._key_snapshot)}, .env: {_fp(dotenv_val)}"
                )
        request_id = getattr(err, "request_id", None) or getattr(
            getattr(err, "response", None), "request_id", None
        )
        if request_id:
            lines.append(f"  request_id        : {request_id}")
        lines.extend([
            "",
            "Troubleshooting:",
            "  - Verify the key at https://console.anthropic.com/settings/keys",
            "  - Check billing    at https://console.anthropic.com/settings/billing",
            "  - If .env and os.environ disagree, call lpo_reload_env or restart the IDE.",
        ])
        return ModelError("\n".join(lines))


def _is_auth_error(err: APIError) -> bool:
    """Detect an Anthropic 401 authentication failure without relying on
    anthropic-specific exception subclasses (the SDK's class hierarchy
    shifts between versions)."""
    status = getattr(err, "status_code", None) or getattr(
        getattr(err, "response", None), "status_code", None
    )
    if status == 401:
        return True
    msg = str(err).lower()
    return "authentication_error" in msg or "invalid x-api-key" in msg


def _fp(value: str | None) -> str:
    """Fingerprint a secret. Never returns enough material to reconstruct
    the value. Mirrors the helper in lpo.server.mcp_server."""
    if value is None:
        return "(not set)"
    s = str(value)
    if not s:
        return "(empty)"
    if len(s) <= 8:
        return f"(len={len(s)}, short)"
    return f"{s[:4]}...{s[-4:]} (len={len(s)})"


def _dotenv_value(key: str) -> str | None:
    """Read a single key from the nearest .env without mutating os.environ.
    Returns None if dotenv isn't available or no .env is found."""
    if dotenv_values is None or find_dotenv is None:
        return None
    try:
        path = find_dotenv(usecwd=True)
        if not path:
            return None
        return dotenv_values(path).get(key)
    except Exception:  # noqa: BLE001
        return None


def _is_deprecated_temperature_error(err: APIError) -> bool:
    """Detect Anthropic's 'temperature is deprecated for this model' 400.

    Newer models reject the parameter outright; the API returns a 400 whose
    body contains the phrase ``temperature`` and ``deprecated``.
    """
    msg = str(err).lower()
    return "temperature" in msg and "deprecated" in msg

