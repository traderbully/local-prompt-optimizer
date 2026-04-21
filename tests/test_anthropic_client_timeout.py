"""AnthropicClient wall-clock timeout + bounded retry tests.

Regression coverage for the 2026-04-21 overseer-hang incident where a stuck
``messages.create`` call blocked the entire optimization loop for >8 minutes
because the SDK's internal httpx timeout did not fire. These tests verify:

1.  A hung SDK call is broken by the outer ``asyncio.wait_for`` wrapper within
    a bounded wall-clock budget, regardless of whether the SDK's own timer
    fires.
2.  On :class:`asyncio.TimeoutError` the client retries up to ``max_retries``
    additional times with exponential backoff, then surfaces a clean
    :class:`~lpo.models.base.ModelError` (never blocks forever, never leaks
    the underlying ``TimeoutError``).
3.  A transient first-attempt timeout followed by a successful second attempt
    returns the real response (retry works, not just error-path).
4.  ``max_retries=0`` fails fast (one attempt only).
5.  Invalid constructor args (``timeout_s<=0``, ``max_retries<0``) are
    rejected at construction time rather than causing silent hangs later.

No real Anthropic API calls are made — the underlying SDK client is replaced
with an asyncio-level mock that we control directly.
"""
from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from lpo.models.anthropic_client import AnthropicClient, AnthropicMessage
from lpo.models.base import ModelError


def _install_create_fn(client: AnthropicClient, fn) -> None:
    """Swap the SDK's ``messages.create`` coroutine for a test-controlled one."""
    client._client = MagicMock()
    client._client.messages = MagicMock()
    client._client.messages.create = fn


def _fake_response(text: str = "ok") -> MagicMock:
    """Minimal stand-in for an anthropic SDK response that ``complete()`` can
    successfully post-process (text extraction + usage + cost tracking)."""
    resp = MagicMock()
    block = MagicMock()
    block.text = text
    resp.content = [block]
    resp.usage = MagicMock(input_tokens=10, output_tokens=5)
    resp.model = "claude-haiku-4-5"
    resp.stop_reason = "end_turn"
    resp.model_dump = lambda: {"served_model_id": "claude-haiku-4-5"}
    return resp


@pytest.mark.asyncio
async def test_hung_call_times_out_within_bounded_wall_time(monkeypatch):
    """Regression: a hung ``messages.create`` must NOT block forever.

    The outer ``asyncio.wait_for`` must cancel the stuck coroutine at
    ``timeout_s`` and, after ``max_retries`` additional attempts, surface a
    :class:`ModelError`. Total wall time must be bounded."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-timeout")
    client = AnthropicClient(
        model_id="claude-haiku-4-5",
        timeout_s=0.3,
        max_retries=2,
    )

    async def _hang(**_kwargs):
        await asyncio.sleep(60)  # would hang the test runner if not cancelled

    _install_create_fn(client, _hang)

    start = time.monotonic()
    with pytest.raises(ModelError) as excinfo:
        await client.complete(
            system="s",
            messages=[AnthropicMessage(role="user", content="hi")],
        )
    elapsed = time.monotonic() - start

    # Budget: 3 attempts x 0.3s timeout = 0.9s + backoffs (1s + 2s = 3s) = 3.9s.
    # Allow 6s slack for slow CI.
    assert elapsed < 6.0, f"client hung for {elapsed:.2f}s (expected <6s)"
    # The surfaced error must be actionable, not a bare TimeoutError.
    msg = str(excinfo.value).lower()
    assert "timed out" in msg
    assert "claude-haiku-4-5" in msg
    assert "3 attempt" in msg  # total_attempts = max_retries + 1 = 3


@pytest.mark.asyncio
async def test_transient_timeout_recovers_via_retry(monkeypatch):
    """First attempt times out, second attempt succeeds -> real response
    is returned to the caller. Retry path is functional, not just error-path."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-transient")
    client = AnthropicClient(
        model_id="claude-haiku-4-5",
        timeout_s=0.3,
        max_retries=2,
    )

    call_count = {"n": 0}
    good_resp = _fake_response(text="recovered")

    async def _flaky(**_kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            await asyncio.sleep(60)  # first call times out
        return good_resp

    _install_create_fn(client, _flaky)

    result = await client.complete(
        system="s",
        messages=[AnthropicMessage(role="user", content="hi")],
    )
    assert result.text == "recovered"
    assert call_count["n"] == 2, "expected exactly one retry after timeout"


@pytest.mark.asyncio
async def test_zero_retries_fails_fast(monkeypatch):
    """``max_retries=0`` -> single attempt, no backoff, immediate fail."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-noretry")
    client = AnthropicClient(
        model_id="claude-haiku-4-5",
        timeout_s=0.2,
        max_retries=0,
    )

    async def _hang(**_kwargs):
        await asyncio.sleep(60)

    _install_create_fn(client, _hang)

    start = time.monotonic()
    with pytest.raises(ModelError):
        await client.complete(
            system="s",
            messages=[AnthropicMessage(role="user", content="hi")],
        )
    elapsed = time.monotonic() - start
    # Single 0.2s timeout, no backoff. 1.5s is generous headroom.
    assert elapsed < 1.5, f"expected fast failure, took {elapsed:.2f}s"


@pytest.mark.asyncio
async def test_successful_call_does_not_retry(monkeypatch):
    """Sanity: a fast, successful call is returned on attempt #1 with no
    spurious retry overhead."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-happy")
    client = AnthropicClient(
        model_id="claude-haiku-4-5",
        timeout_s=5.0,
        max_retries=2,
    )

    call_count = {"n": 0}
    good_resp = _fake_response(text="fast")

    async def _ok(**_kwargs):
        call_count["n"] += 1
        return good_resp

    _install_create_fn(client, _ok)

    result = await client.complete(
        system="s",
        messages=[AnthropicMessage(role="user", content="hi")],
    )
    assert result.text == "fast"
    assert call_count["n"] == 1


def test_invalid_timeout_rejected_at_construction(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    with pytest.raises(ValueError, match="timeout_s"):
        AnthropicClient(model_id="claude-haiku-4-5", timeout_s=0)
    with pytest.raises(ValueError, match="timeout_s"):
        AnthropicClient(model_id="claude-haiku-4-5", timeout_s=-1.0)


def test_invalid_max_retries_rejected_at_construction(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    with pytest.raises(ValueError, match="max_retries"):
        AnthropicClient(model_id="claude-haiku-4-5", max_retries=-1)
