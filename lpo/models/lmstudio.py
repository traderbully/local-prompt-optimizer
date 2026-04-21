"""LM Studio (OpenAI-compatible) async client.

See `LPO_SDP.md` §5.2, §6.1. Implements exponential backoff with retry on transient
network and 5xx errors. Rate limits (429) and hard 4xx errors are not retried.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from typing import Any

import httpx

from lpo.models.base import ContentBlock, GenerationResult, ModelClient, ModelError

log = logging.getLogger("lpo.models.lmstudio")

RETRYABLE_STATUS = {500, 502, 503, 504}


class LMStudioClient(ModelClient):
    provider = "lmstudio"

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:1234/v1",
        model_id: str,
        api_key_env: str | None = None,
        timeout_s: float = 300.0,
        max_retries: int = 3,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.max_retries = max_retries
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key_env:
            key = os.environ.get(api_key_env)
            if key:
                headers["Authorization"] = f"Bearer {key}"
        self._client = httpx.AsyncClient(timeout=timeout_s, headers=headers)

    async def aclose(self) -> None:
        await self._client.aclose()

    @staticmethod
    def _to_message_content(user_input: str | list[ContentBlock]) -> Any:
        if isinstance(user_input, str):
            return user_input
        parts: list[dict[str, Any]] = []
        for block in user_input:
            if block.kind == "text":
                parts.append({"type": "text", "text": block.data})
            elif block.kind == "image_base64":
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{block.data}"},
                    }
                )
            elif block.kind == "image_url":
                parts.append({"type": "image_url", "image_url": {"url": block.data}})
            else:
                raise ModelError(f"Unsupported ContentBlock.kind: {block.kind!r}")
        return parts

    # Cap on automatic max_tokens escalation when a reasoning model eats the
    # entire budget with its Chain-of-Thought. Prevents runaway retries from
    # a mis-configured model that simply never produces `content`.
    REASONING_RETRY_CAP = 16384

    async def generate(
        self,
        system_prompt: str,
        user_input: str | list[ContentBlock],
        *,
        seed: int | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> GenerationResult:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": self._to_message_content(user_input)},
        ]
        start = time.perf_counter()

        # First attempt with the caller-provided budget.
        data, text, reasoning, finish, usage = await self._post_chat(
            messages=messages, seed=seed, temperature=temperature, max_tokens=max_tokens
        )

        # Reasoning-budget auto-retry: if the model produced *no* final
        # content and hit the length ceiling, double max_tokens (up to the
        # cap) and try once more. One retry only to bound cost.
        #
        # Precondition history: originally required `reasoning.strip()` to
        # be non-empty on the theory that visible hidden-CoT was the only
        # reliable signal of a reasoning model eating its budget. The
        # Apr 21 forensic investigation on gemma-4-26b-local found the
        # opposite: LM Studio's response for some models (Gemma 4 MoE
        # among them) doesn't surface `reasoning_content` even when the
        # model IS internally reasoning and consuming the ceiling. The old
        # precondition therefore silently skipped the retry on exactly
        # the cases where it was most needed. Empty content + length
        # truncation is, on its own, sufficient evidence: doubling the
        # budget is the right move regardless of whether we can see the
        # hidden CoT. Pathological no-CoT cases (tokenizer bug producing
        # empty output at length) escalate to the ERROR log below on the
        # second attempt — same end state as before, minus the silent
        # retry-skip that used to cause 3+ iterations of 0-scored outputs.
        if (
            not text.strip()
            and finish == "length"
            and max_tokens < self.REASONING_RETRY_CAP
        ):
            bumped = min(max_tokens * 2, self.REASONING_RETRY_CAP)
            log.info(
                "LM Studio reasoning-budget auto-retry: model=%s produced empty "
                "content at max_tokens=%d (finish=length, reasoning_content=%d "
                "chars). Retrying with max_tokens=%d (cap=%d).",
                self.model_id, max_tokens, len(reasoning), bumped, self.REASONING_RETRY_CAP,
            )
            data, text, reasoning, finish, usage = await self._post_chat(
                messages=messages, seed=seed, temperature=temperature, max_tokens=bumped,
            )
            max_tokens = bumped  # surfaced through the final result's raw payload

        # Escalated logging: empty content after all retries is a real error,
        # not a cosmetic warning. The caller's iteration will score 0 and
        # operators need to see this prominently in the log.
        if not text.strip():
            reasoning_tokens = usage.get("completion_tokens_details", {}).get("reasoning_tokens")
            hint = f"finish_reason={finish!r}, reasoning_tokens={reasoning_tokens}"
            if reasoning.strip():
                log.error(
                    "LM Studio returned empty content after reasoning-budget retry: "
                    "model=%s, reasoning_content=%d chars, %s. Cap=%d reached; increase "
                    "target_models.max_tokens in config.yaml or load a model with a "
                    "larger context window.",
                    self.model_id, len(reasoning), hint, self.REASONING_RETRY_CAP,
                )
            else:
                log.error(
                    "LM Studio returned empty content (no reasoning either): model=%s, "
                    "%s. Likely a tokenizer/server misconfiguration. Raw: %s",
                    self.model_id, hint, str(data)[:500],
                )

        latency = time.perf_counter() - start
        return GenerationResult(
            text=text,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            latency_s=latency,
            seed=seed,
            model_id=self.model_id,
            provider=self.provider,
            raw=data,
            finish_reason=_as_str_or_none(finish),
            reasoning_tokens=_reasoning_tokens(usage),
        )

    async def _post_chat(
        self,
        *,
        messages: list[dict[str, Any]],
        seed: int | None,
        temperature: float,
        max_tokens: int,
    ) -> tuple[dict[str, Any], str, str, Any, dict[str, Any]]:
        """Single chat completion with transient-failure retries. Returns
        ``(raw_data, text, reasoning_text, finish_reason, usage)``."""
        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            payload["seed"] = seed

        url = f"{self.base_url}/chat/completions"
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = await self._client.post(url, json=payload)
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as e:
                last_err = e
                await asyncio.sleep(_backoff(attempt))
                continue
            if resp.status_code in RETRYABLE_STATUS:
                last_err = ModelError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                await asyncio.sleep(_backoff(attempt))
                continue
            if resp.status_code >= 400:
                raise ModelError(f"HTTP {resp.status_code}: {resp.text[:500]}")
            data = resp.json()
            break
        else:
            raise ModelError(
                f"LM Studio request failed after {self.max_retries} attempts: {last_err}"
            )

        try:
            choice = data["choices"][0]
            msg = choice["message"]
            text = msg.get("content") or ""
        except (KeyError, IndexError, TypeError) as e:
            raise ModelError(f"Malformed LM Studio response: {e}; raw={data}") from e
        usage = data.get("usage") or {}
        finish = choice.get("finish_reason")
        reasoning = msg.get("reasoning_content") or ""
        return data, text, reasoning, finish, usage


def _backoff(attempt: int) -> float:
    base = 0.5 * (2**attempt)
    return base + random.uniform(0, 0.25)


def _as_str_or_none(v: Any) -> str | None:
    """Normalize a provider-reported finish_reason to a plain string or None.
    LM Studio occasionally returns non-string sentinels; we keep the type
    boring so downstream serializers don't have to worry."""
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _reasoning_tokens(usage: dict[str, Any]) -> int | None:
    """Pull the hidden-CoT token count from the usage block, if present.

    OpenAI-compatible servers (including LM Studio) nest this under
    ``completion_tokens_details.reasoning_tokens``. Returns None when
    the server didn't report one; 0 when it reported a zero. The
    distinction matters for diagnosis: None means "server didn't
    surface this" (our best-case guess is unknown); 0 means "model
    claims to have done zero reasoning" (which on a reasoning model
    is itself suspicious)."""
    details = usage.get("completion_tokens_details")
    if not isinstance(details, dict):
        return None
    value = details.get("reasoning_tokens")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
