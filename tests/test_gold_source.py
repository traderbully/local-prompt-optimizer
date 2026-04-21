"""Gold Standard source factory tests.

Covers the env-var-driven provider selection in :mod:`lpo.models.gold_source`.
The Anthropic path is verified without hitting the network by checking that
the returned object is an :class:`AnthropicClient` with the expected model id.
The OpenRouter path is verified by confirming the adapter wraps an
:class:`OpenRouterClient` and that ``.complete`` correctly translates
Anthropic-style ``messages=[{role, content}]`` into a generate() call.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from lpo.models.base import ModelError
from lpo.models.gold_source import (
    _OpenRouterGoldAdapter,
    build_gold_standard_source,
)


@pytest.fixture(autouse=True)
def _clear_gs_env(monkeypatch):
    for k in (
        "GOLD_STANDARD_PROVIDER",
        "GOLD_STANDARD_MODEL",
        "GOLD_STANDARD_BASE_URL",
        "GOLD_STANDARD_API_KEY_ENV",
    ):
        monkeypatch.delenv(k, raising=False)
    yield


def test_default_provider_is_anthropic(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-xyz")
    source = build_gold_standard_source(model_id="claude-haiku-4-5")
    from lpo.models.anthropic_client import AnthropicClient
    assert isinstance(source, AnthropicClient)
    assert source.model_id == "claude-haiku-4-5"


def test_gold_standard_model_env_overrides_caller(monkeypatch):
    # Operator preference (env) wins over the caller's default; this matches
    # the SDP's "config over code" principle.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-xyz")
    monkeypatch.setenv("GOLD_STANDARD_MODEL", "claude-opus-override")
    source = build_gold_standard_source(model_id="claude-haiku-4-5")
    assert source.model_id == "claude-opus-override"


def test_openrouter_provider_returns_adapter(monkeypatch):
    monkeypatch.setenv("GOLD_STANDARD_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    source = build_gold_standard_source(model_id="anthropic/claude-opus-4.6")
    assert isinstance(source, _OpenRouterGoldAdapter)
    # Inner client is OpenRouterClient with the right model.
    from lpo.models.openrouter import OpenRouterClient
    assert isinstance(source._inner, OpenRouterClient)
    assert source._inner.model_id == "anthropic/claude-opus-4.6"


def test_openrouter_provider_respects_custom_base_url_and_key_env(monkeypatch):
    monkeypatch.setenv("GOLD_STANDARD_PROVIDER", "openrouter")
    monkeypatch.setenv("GOLD_STANDARD_BASE_URL", "https://proxy.example/api/v1")
    monkeypatch.setenv("GOLD_STANDARD_API_KEY_ENV", "CUSTOM_KEY")
    monkeypatch.setenv("CUSTOM_KEY", "sk-custom-1234")
    source = build_gold_standard_source(model_id="x/y")
    assert source._inner.base_url == "https://proxy.example/api/v1"


def test_unknown_provider_raises(monkeypatch):
    monkeypatch.setenv("GOLD_STANDARD_PROVIDER", "vertex")
    with pytest.raises(ModelError) as excinfo:
        build_gold_standard_source(model_id="claude-haiku-4-5")
    assert "vertex" in str(excinfo.value)
    # Error must list valid options so the operator can fix .env.
    assert "anthropic" in str(excinfo.value)
    assert "openrouter" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Adapter translation layer: Anthropic-shape messages -> OpenRouter generate()
# ---------------------------------------------------------------------------


class _FakeInner:
    """Records exactly what the adapter forwards into generate()."""

    def __init__(self) -> None:
        self.call: dict[str, Any] | None = None
        self.closed = False

    async def generate(
        self,
        system: str,
        user_input: str,
        *,
        temperature: float,
        max_tokens: int,
    ) -> Any:
        self.call = {
            "system": system,
            "user_input": user_input,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        return SimpleNamespace(text="GOLD_STANDARD_OUTPUT")

    async def aclose(self) -> None:
        self.closed = True


@dataclass
class _AntMsgLike:
    role: str
    content: str


@pytest.mark.asyncio
async def test_adapter_translates_anthropic_message_to_generate_call():
    inner = _FakeInner()
    adapter = _OpenRouterGoldAdapter(inner)
    result = await adapter.complete(
        system="You are gold source.",
        messages=[_AntMsgLike(role="user", content="Produce ideal output for X.")],
        temperature=0.0,
        max_tokens=512,
    )
    assert result.text == "GOLD_STANDARD_OUTPUT"
    assert inner.call == {
        "system": "You are gold source.",
        "user_input": "Produce ideal output for X.",
        "temperature": 0.0,
        "max_tokens": 512,
    }


@pytest.mark.asyncio
async def test_adapter_accepts_dict_messages_too():
    # Minor robustness check: the authoring layer uses AnthropicMessage today
    # but any {role, content} dict should work so we never surprise a future
    # caller that doesn't want to depend on the anthropic SDK.
    inner = _FakeInner()
    adapter = _OpenRouterGoldAdapter(inner)
    await adapter.complete(
        system="s", messages=[{"role": "user", "content": "hi"}],
    )
    assert inner.call is not None
    assert inner.call["user_input"] == "hi"


@pytest.mark.asyncio
async def test_adapter_forwards_close():
    inner = _FakeInner()
    adapter = _OpenRouterGoldAdapter(inner)
    await adapter.aclose()
    assert inner.closed is True


@pytest.mark.asyncio
async def test_adapter_raises_on_empty_messages():
    inner = _FakeInner()
    adapter = _OpenRouterGoldAdapter(inner)
    with pytest.raises(ModelError):
        await adapter.complete(system="s", messages=[])
