"""AnthropicClient 401 enrichment tests.

The production client wraps Anthropic's authentication errors with a
diagnostic block containing: env var name, key fingerprint, drift between
client-snapshot and live ``os.environ``, and a .env-vs-shell comparison.
These are verified here without hitting the real Anthropic API by
injecting a canned ``APIError`` through the underlying AsyncAnthropic.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from anthropic import APIError

from lpo.models.anthropic_client import AnthropicClient, AnthropicMessage, _fp
from lpo.models.base import ModelError


class _FakeAuthError(APIError):
    """APIError subclass that mimics the status/request shape Anthropic
    returns on a 401. We can't use the SDK's AuthenticationError because
    its __init__ signature shifts between versions."""

    def __init__(self, message: str = "authentication_error: invalid x-api-key"):
        # Don't call super().__init__ — APIError's constructor varies across
        # SDK versions and we only need the subclass-of-APIError identity
        # plus a readable string.
        self._message = message
        self.status_code = 401
        self.request_id = "req_test_12345"

    def __str__(self) -> str:  # noqa: D401
        return self._message


def _build_client(monkeypatch: pytest.MonkeyPatch, api_key: str = "sk-ant-test-abcd1234efgh5678") -> AnthropicClient:
    monkeypatch.setenv("ANTHROPIC_API_KEY", api_key)
    c = AnthropicClient(model_id="claude-haiku-4-5")
    # Replace the underlying SDK client with a mock that raises auth error.
    c._client = MagicMock()
    c._client.messages = MagicMock()
    c._client.messages.create = AsyncMock(side_effect=_FakeAuthError())
    return c


@pytest.mark.asyncio
async def test_auth_error_includes_key_fingerprint_and_env_name(monkeypatch):
    client = _build_client(monkeypatch, api_key="sk-ant-test-abcd1234efgh5678")
    with pytest.raises(ModelError) as excinfo:
        await client.complete(system="s", messages=[AnthropicMessage("user", "hi")])
    msg = str(excinfo.value)
    assert "Anthropic authentication failed" in msg
    assert "ANTHROPIC_API_KEY" in msg
    # Fingerprint must be obfuscated but identifiable by the operator.
    assert "sk-a" in msg and "5678" in msg
    # Middle of the secret MUST NOT appear.
    assert "test-abcd1234" not in msg
    assert "req_test_12345" in msg


@pytest.mark.asyncio
async def test_auth_error_flags_env_drift(monkeypatch):
    client = _build_client(monkeypatch, api_key="sk-ant-original-aaaa1111bbbb2222")
    # Simulate a later rotation: .env / shell env now contain a new value,
    # but this client instance is still holding the original. The enriched
    # error should flag the drift so the operator knows to restart / reload.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-rotated-cccc3333dddd4444")
    with pytest.raises(ModelError) as excinfo:
        await client.complete(system="s", messages=[AnthropicMessage("user", "hi")])
    msg = str(excinfo.value)
    assert "key drift" in msg
    assert "DIFFERENT" in msg


@pytest.mark.asyncio
async def test_non_auth_errors_use_legacy_message_format(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    c = AnthropicClient(model_id="claude-haiku-4-5")
    c._client = MagicMock()
    c._client.messages = MagicMock()

    class _ServerError(APIError):
        def __init__(self):
            self.status_code = 500
        def __str__(self): return "upstream is unhappy"

    c._client.messages.create = AsyncMock(side_effect=_ServerError())
    with pytest.raises(ModelError) as excinfo:
        await c.complete(system="s", messages=[AnthropicMessage("user", "hi")])
    # Legacy phrasing preserved — no auth-diagnostic block when it wasn't
    # a 401.
    assert "Anthropic API error" in str(excinfo.value)
    assert "authentication failed" not in str(excinfo.value).lower()


def test_fingerprint_helper_never_leaks_middle_of_secret():
    # Defence in depth: the module-level helper used by auth enrichment
    # (and mirrored in the MCP server) must never emit middle chars.
    secret = "sk-ant-api03-verysecretmiddlexyz-tail1234"
    fp = _fp(secret)
    assert "verysecretmiddle" not in fp
    assert fp.startswith(secret[:4])
    assert fp.endswith(f"{secret[-4:]} (len={len(secret)})")


def test_fingerprint_short_values_are_length_only():
    assert _fp(None) == "(not set)"
    assert _fp("") == "(empty)"
    assert "short" in _fp("tiny")
