"""Stage 4.5 — OpenRouter client + dynamic pricing tests.

All network I/O is faked. The goals:

1. The pricing parser extracts ``{model_id: (in_per_mtok, out_per_mtok)}``
   from OpenRouter's ``/api/v1/models`` payload with the correct unit
   conversion (USD/token → USD per 1M tokens).
2. The shared pricing cache fetches exactly once no matter how many clients
   touch it, and tolerates fetch failures by collapsing to $0 rates.
3. The client registers its model's rates with the CostTracker on first call
   and records usage at the correct rate.
4. Transport concerns — Bearer auth header, reasoning-model empty-content
   warning, 429 retry path — are exercised against a monkeypatched AsyncClient.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from lpo.config.schema import TargetModelConfig
from lpo.core.cost import CostTracker
from lpo.models.base import ModelError
from lpo.models.openrouter import OpenRouterClient
from lpo.models.openrouter_pricing import (
    OpenRouterPricing,
    parse_models_payload,
    reset_shared_pricing_for_tests,
)
from lpo.models.registry import build_client


# ---------------------------------------------------------------------------
# Pricing parser
# ---------------------------------------------------------------------------


def test_parse_models_payload_converts_per_token_to_per_mtok():
    payload = {
        "data": [
            {
                "id": "google/gemma-4-31b-it",
                "pricing": {"prompt": "0.0000002", "completion": "0.0000006"},
            }
        ]
    }
    rates = parse_models_payload(payload)
    assert rates["google/gemma-4-31b-it"] == (pytest.approx(0.2), pytest.approx(0.6))


def test_parse_models_payload_is_defensive():
    # Missing or malformed fields collapse to 0 without raising.
    payload = {
        "data": [
            {"id": "a/b", "pricing": {"prompt": "bogus", "completion": None}},
            {"id": "c/d"},  # no pricing at all
            {"pricing": {"prompt": "0.0000001"}},  # no id — skipped
            "not a dict",  # skipped
        ]
    }
    rates = parse_models_payload(payload)
    assert rates == {
        "a/b": (0.0, 0.0),
        "c/d": (0.0, 0.0),
    }


def test_parse_models_payload_empty():
    assert parse_models_payload({}) == {}
    assert parse_models_payload({"data": "not-a-list"}) == {}


# ---------------------------------------------------------------------------
# OpenRouterPricing cache
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pricing_fetches_once_then_caches():
    calls = {"n": 0}

    async def fake_fetcher() -> dict:
        calls["n"] += 1
        return {
            "data": [
                {"id": "x/y", "pricing": {"prompt": "0.0000001", "completion": "0.0000002"}}
            ]
        }

    cache = OpenRouterPricing(fetcher=fake_fetcher)
    assert not cache.is_loaded()
    assert await cache.rate_for("x/y") == (pytest.approx(0.1), pytest.approx(0.2))
    assert await cache.rate_for("x/y") == (pytest.approx(0.1), pytest.approx(0.2))
    assert await cache.rate_for("unknown/model") == (0.0, 0.0)
    assert calls["n"] == 1, "expected a single fetch across three lookups"


@pytest.mark.asyncio
async def test_pricing_fetch_failure_is_tolerated():
    async def failing_fetcher() -> dict:
        raise httpx.ConnectError("no network")

    cache = OpenRouterPricing(fetcher=failing_fetcher)
    # Should NOT raise — fall through to (0, 0) and set failed flag.
    assert await cache.rate_for("anything/goes") == (0.0, 0.0)
    # Retrying after a failure should not re-fetch (we don't hammer a dead
    # endpoint mid-run).
    calls_after = 0
    async def failing_fetcher_2() -> dict:
        nonlocal calls_after
        calls_after += 1
        return {"data": []}

    cache._fetcher = failing_fetcher_2  # swap to a working fetcher
    await cache.rate_for("anything/else")
    assert calls_after == 0


@pytest.mark.asyncio
async def test_pricing_concurrent_access_fetches_once():
    """Many coroutines hitting rate_for simultaneously must produce one fetch."""
    import asyncio

    calls = {"n": 0}

    async def slow_fetcher() -> dict:
        calls["n"] += 1
        await asyncio.sleep(0.02)
        return {"data": [{"id": "a/b", "pricing": {"prompt": "0.0000005", "completion": "0.0000005"}}]}

    cache = OpenRouterPricing(fetcher=slow_fetcher)
    results = await asyncio.gather(*[cache.rate_for("a/b") for _ in range(8)])
    assert all(r == (pytest.approx(0.5), pytest.approx(0.5)) for r in results)
    assert calls["n"] == 1


# ---------------------------------------------------------------------------
# OpenRouterClient — construction & auth
# ---------------------------------------------------------------------------


def test_client_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ModelError, match="OpenRouter API key"):
        OpenRouterClient(model_id="google/gemma-4-31b-it")


def test_client_sets_bearer_header(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-123")
    client = OpenRouterClient(model_id="google/gemma-4-31b-it")
    assert client._client.headers["Authorization"] == "Bearer sk-or-test-123"
    assert client._client.headers["Content-Type"] == "application/json"


def test_client_adds_attribution_headers(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-456")
    client = OpenRouterClient(
        model_id="google/gemma-4-31b-it",
        http_referer="https://example.com",
        x_title="LPO",
    )
    assert client._client.headers["HTTP-Referer"] == "https://example.com"
    assert client._client.headers["X-Title"] == "LPO"


# ---------------------------------------------------------------------------
# OpenRouterClient — generate path
# ---------------------------------------------------------------------------


async def _fake_pricing(rates: dict[str, tuple[float, float]]) -> OpenRouterPricing:
    async def fetcher() -> dict:
        return {
            "data": [
                {"id": mid, "pricing": {"prompt": str(r[0] / 1e6), "completion": str(r[1] / 1e6)}}
                for mid, r in rates.items()
            ]
        }

    return OpenRouterPricing(fetcher=fetcher)


def _ok_response(text: str, prompt_tokens: int, completion_tokens: int) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json = MagicMock(
        return_value={
            "choices": [
                {
                    "message": {"content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        }
    )
    resp.text = "ok"
    return resp


@pytest.mark.asyncio
async def test_generate_registers_rate_and_records_cost(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    pricing = await _fake_pricing({"google/gemma-4-31b-it": (0.4, 0.6)})
    cost = CostTracker()

    client = OpenRouterClient(
        model_id="google/gemma-4-31b-it",
        cost_tracker=cost,
        pricing=pricing,
    )
    # Patch the underlying AsyncClient.post to return a canned response.
    client._client.post = AsyncMock(return_value=_ok_response("hello", 1000, 2000))

    gen = await client.generate("sys", "user-text")
    assert gen.text == "hello"
    assert gen.prompt_tokens == 1000
    assert gen.completion_tokens == 2000
    assert gen.provider == "openrouter"

    # Cost = 1000 * 0.4 / 1M + 2000 * 0.6 / 1M = 0.0004 + 0.0012 = 0.0016
    assert cost.total_usd == pytest.approx(0.0016)
    assert len(cost.calls) == 1

    # Subsequent call must reuse the rate without re-registering.
    client._client.post = AsyncMock(return_value=_ok_response("again", 500, 500))
    await client.generate("sys", "more")
    expected_add = (500 * 0.4 + 500 * 0.6) / 1e6  # 0.0005
    assert cost.total_usd == pytest.approx(0.0016 + expected_add)


@pytest.mark.asyncio
async def test_generate_unknown_model_records_zero_cost(monkeypatch, caplog):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    pricing = await _fake_pricing({})  # empty catalogue
    cost = CostTracker()

    client = OpenRouterClient(
        model_id="nobody/ever-heard-of-this",
        cost_tracker=cost,
        pricing=pricing,
    )
    client._client.post = AsyncMock(return_value=_ok_response("hi", 100, 100))
    with caplog.at_level("WARNING", logger="lpo.models.openrouter"):
        await client.generate("sys", "x")
    assert cost.total_usd == 0.0
    assert any("no known pricing" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_generate_4xx_raises(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    pricing = await _fake_pricing({})
    cost = CostTracker()

    bad = MagicMock()
    bad.status_code = 401
    bad.text = '{"error": {"message": "Invalid key"}}'

    client = OpenRouterClient(
        model_id="google/gemma-4-31b-it",
        cost_tracker=cost,
        pricing=pricing,
    )
    client._client.post = AsyncMock(return_value=bad)
    with pytest.raises(ModelError, match="HTTP 401"):
        await client.generate("sys", "x")


@pytest.mark.asyncio
async def test_generate_429_retries(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    pricing = await _fake_pricing({"google/gemma-4-31b-it": (0.4, 0.6)})
    cost = CostTracker()

    throttled = MagicMock()
    throttled.status_code = 429
    throttled.text = "rate limited"

    client = OpenRouterClient(
        model_id="google/gemma-4-31b-it",
        cost_tracker=cost,
        pricing=pricing,
        max_retries=3,
    )
    # First two 429s, then success.
    client._client.post = AsyncMock(
        side_effect=[throttled, throttled, _ok_response("done", 10, 10)]
    )
    # Don't actually sleep in tests.
    import lpo.models.openrouter as mod

    monkeypatch.setattr(mod, "_backoff", lambda _a: 0.0)
    gen = await client.generate("sys", "x")
    assert gen.text == "done"
    assert client._client.post.call_count == 3


@pytest.mark.asyncio
async def test_reasoning_empty_content_warning(monkeypatch, caplog):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    pricing = await _fake_pricing({"deepseek/deepseek-v3.2": (0.1, 0.2)})
    cost = CostTracker()
    client = OpenRouterClient(
        model_id="deepseek/deepseek-v3.2",
        cost_tracker=cost,
        pricing=pricing,
    )
    resp = MagicMock()
    resp.status_code = 200
    resp.json = MagicMock(
        return_value={
            "choices": [
                {
                    "message": {"content": "", "reasoning_content": "I thought about it..." * 20},
                    "finish_reason": "length",
                }
            ],
            "usage": {"prompt_tokens": 50, "completion_tokens": 500},
        }
    )
    resp.text = "ok"
    client._client.post = AsyncMock(return_value=resp)
    with caplog.at_level("INFO", logger="lpo.models.openrouter"):
        gen = await client.generate("sys", "x")
    assert gen.text == ""
    # Auto-retry path: one INFO log announcing the retry, one ERROR after
    # the retry still fails. The exact message wording is less important
    # than the fact that both records mention reasoning_content.
    assert any("reasoning_content" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_openrouter_reasoning_auto_retry_recovers(monkeypatch):
    # Parity with LMStudioClient: empty content + populated reasoning +
    # finish_reason='length' must trigger a retry with doubled max_tokens,
    # and content returned on the retry should surface as the final result.
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    pricing = await _fake_pricing({"z/thinker": (0.1, 0.2)})
    client = OpenRouterClient(model_id="z/thinker", pricing=pricing)

    first = MagicMock()
    first.status_code = 200
    first.text = "ok"
    first.json = MagicMock(return_value={
        "choices": [{
            "message": {"content": "", "reasoning_content": "think " * 100},
            "finish_reason": "length",
        }],
        "usage": {"prompt_tokens": 20, "completion_tokens": 500},
    })
    second = MagicMock()
    second.status_code = 200
    second.text = "ok"
    second.json = MagicMock(return_value={
        "choices": [{
            "message": {"content": '{"ok": true}'},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 20, "completion_tokens": 200},
    })
    client._client.post = AsyncMock(side_effect=[first, second])

    gen = await client.generate("sys", "x", max_tokens=2048)

    assert gen.text == '{"ok": true}'
    assert client._client.post.call_count == 2
    second_payload = client._client.post.call_args_list[1].kwargs["json"]
    assert second_payload["max_tokens"] == 4096


@pytest.mark.asyncio
async def test_openrouter_empty_content_with_finish_stop_no_retry(monkeypatch, caplog):
    # finish_reason=='stop' with empty content is a tokenizer/server bug,
    # not a reasoning-budget issue. Doubling max_tokens wouldn't help and
    # we don't want to burn latency on a doomed retry. Only finish='length'
    # triggers the reasoning-budget auto-retry.
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    pricing = await _fake_pricing({"x/broken": (0.1, 0.2)})
    client = OpenRouterClient(model_id="x/broken", pricing=pricing)
    resp = MagicMock()
    resp.status_code = 200
    resp.text = "ok"
    resp.json = MagicMock(return_value={
        "choices": [{
            "message": {"content": "", "reasoning_content": ""},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 0},
    })
    client._client.post = AsyncMock(return_value=resp)
    with caplog.at_level("ERROR", logger="lpo.models.openrouter"):
        gen = await client.generate("sys", "x", max_tokens=1024)
    assert gen.text == ""
    assert client._client.post.call_count == 1
    # Must still escalate to ERROR (part of Stage-7 logging hygiene).
    assert any(
        r.levelname == "ERROR" and "no reasoning" in r.message
        for r in caplog.records
    ), [(r.levelname, r.message) for r in caplog.records]


@pytest.mark.asyncio
async def test_openrouter_retry_fires_when_reasoning_absent_but_finish_is_length(monkeypatch):
    """Parity with the LM Studio fix: an OpenRouter-routed reasoning model
    whose provider strips hidden CoT from the response used to silently
    skip the auto-retry because the old precondition required
    ``reasoning.strip()`` to be non-empty. The new precondition drops that
    and retries on empty-content + finish='length' alone. Verified against
    the same failure pattern that hit gemma-4-26b-local on LM Studio.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    pricing = await _fake_pricing({"x/reasoner-no-cot": (0.1, 0.2)})
    client = OpenRouterClient(model_id="x/reasoner-no-cot", pricing=pricing)

    first = MagicMock()
    first.status_code = 200
    first.text = "ok"
    first.json = MagicMock(return_value={
        "choices": [{
            "message": {"content": "", "reasoning_content": ""},  # CoT ABSENT
            "finish_reason": "length",
        }],
        "usage": {"prompt_tokens": 100, "completion_tokens": 2048},
    })

    second = MagicMock()
    second.status_code = 200
    second.text = "ok"
    second.json = MagicMock(return_value={
        "choices": [{
            "message": {"content": '{"ok": true}'},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50},
    })

    client._client.post = AsyncMock(side_effect=[first, second])

    gen = await client.generate("sys", "x", max_tokens=2048)

    assert gen.text == '{"ok": true}'
    assert client._client.post.call_count == 2
    # Second call used doubled max_tokens per the retry contract.
    second_payload = client._client.post.call_args_list[1].kwargs["json"]
    assert second_payload["max_tokens"] == 4096


# ---------------------------------------------------------------------------
# Schema / registry integration
# ---------------------------------------------------------------------------


def test_schema_rejects_invalid_openrouter_model_id():
    with pytest.raises(Exception):  # pydantic wraps the ValueError
        TargetModelConfig(
            slug="or1",
            provider="openrouter",
            model_id="not-an-openrouter-style-id",
        )


def test_schema_accepts_valid_openrouter_model_id():
    cfg = TargetModelConfig(
        slug="or1", provider="openrouter", model_id="google/gemma-4-31b-it"
    )
    assert cfg.provider == "openrouter"


def test_registry_builds_openrouter_client(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    cfg = TargetModelConfig(
        slug="or1", provider="openrouter", model_id="google/gemma-4-31b-it"
    )
    cost = CostTracker()
    client = build_client(cfg, cost_tracker=cost)
    assert isinstance(client, OpenRouterClient)
    # Base URL must be the OpenRouter endpoint even though the schema default
    # is the LM Studio one.
    assert client.base_url.startswith("https://openrouter.ai")


# ---------------------------------------------------------------------------
# Cleanup — don't leak the global pricing singleton across test modules.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_pricing_singleton():
    reset_shared_pricing_for_tests()
    yield
    reset_shared_pricing_for_tests()
