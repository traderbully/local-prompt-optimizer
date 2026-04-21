"""LM Studio client tests — reasoning-model auto-retry + empty-content logging."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from lpo.models.lmstudio import LMStudioClient


def _resp(body: dict, status: int = 200) -> MagicMock:
    r = MagicMock()
    r.status_code = status
    r.json = MagicMock(return_value=body)
    r.text = "ok"
    return r


def _make(msg: dict, *, finish: str = "stop", usage: dict | None = None) -> dict:
    return {
        "choices": [{"message": msg, "finish_reason": finish}],
        "usage": usage or {"prompt_tokens": 10, "completion_tokens": 10},
    }


@pytest.mark.asyncio
async def test_reasoning_budget_auto_retry_recovers_with_doubled_max_tokens():
    # First call: reasoning model ate the budget, no content produced.
    # Second call (auto-retry with 2x max_tokens): content is produced.
    client = LMStudioClient(model_id="thinker")
    first = _resp(_make(
        {"content": "", "reasoning_content": "thinking... " * 100},
        finish="length",
        usage={
            "prompt_tokens": 50,
            "completion_tokens": 500,
            "completion_tokens_details": {"reasoning_tokens": 500},
        },
    ))
    second = _resp(_make(
        {"content": '{"answer": 42}', "reasoning_content": "shorter cot"},
        finish="stop",
        usage={"prompt_tokens": 50, "completion_tokens": 120},
    ))
    client._client.post = AsyncMock(side_effect=[first, second])

    gen = await client.generate("sys", "hi", max_tokens=2048)

    assert gen.text == '{"answer": 42}'
    assert client._client.post.call_count == 2
    # Second call must have used a doubled max_tokens.
    second_payload = client._client.post.call_args_list[1].kwargs["json"]
    assert second_payload["max_tokens"] == 4096


@pytest.mark.asyncio
async def test_reasoning_budget_retry_caps_at_limit():
    # If caller already passes a max_tokens at-or-above the cap, no retry
    # should happen — we're as generous as we can be; failure is in the model.
    client = LMStudioClient(model_id="thinker")
    truncated = _resp(_make(
        {"content": "", "reasoning_content": "still thinking"},
        finish="length",
    ))
    client._client.post = AsyncMock(return_value=truncated)

    gen = await client.generate(
        "sys", "hi", max_tokens=LMStudioClient.REASONING_RETRY_CAP
    )

    assert gen.text == ""
    # Exactly one post — no retry because we were already at the cap.
    assert client._client.post.call_count == 1


@pytest.mark.asyncio
async def test_no_retry_when_reasoning_content_is_empty():
    # Empty content AND empty reasoning → a tokenizer/server bug, not a CoT
    # budget issue. Retrying with more tokens would not help; don't waste
    # latency on a doomed retry.
    client = LMStudioClient(model_id="broken")
    client._client.post = AsyncMock(return_value=_resp(
        _make({"content": "", "reasoning_content": ""}, finish="stop")
    ))

    gen = await client.generate("sys", "hi", max_tokens=2048)

    assert gen.text == ""
    assert client._client.post.call_count == 1


@pytest.mark.asyncio
async def test_empty_content_after_retry_logs_at_error_level(caplog):
    # Regression: pre-Stage-7 the empty-content message was a WARNING.
    # That's too quiet — callers scored 0 silently. It's now an ERROR.
    client = LMStudioClient(model_id="thinker")
    client._client.post = AsyncMock(return_value=_resp(_make(
        {"content": "", "reasoning_content": "thinking " * 50},
        finish="length",
    )))

    with caplog.at_level("ERROR", logger="lpo.models.lmstudio"):
        await client.generate("sys", "hi", max_tokens=LMStudioClient.REASONING_RETRY_CAP)

    assert any(
        "empty content" in rec.message.lower() and rec.levelname == "ERROR"
        for rec in caplog.records
    ), [(r.levelname, r.message) for r in caplog.records]


@pytest.mark.asyncio
async def test_normal_content_path_still_works():
    # Sanity: the refactor must not regress the happy path.
    client = LMStudioClient(model_id="plain")
    client._client.post = AsyncMock(return_value=_resp(_make(
        {"content": "hello"}, usage={"prompt_tokens": 3, "completion_tokens": 1},
    )))

    gen = await client.generate("sys", "hi")

    assert gen.text == "hello"
    assert gen.prompt_tokens == 3
    assert gen.completion_tokens == 1
    assert client._client.post.call_count == 1
