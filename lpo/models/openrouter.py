"""OpenRouter target client.

OpenRouter exposes an OpenAI-compatible ``/api/v1/chat/completions`` endpoint
for a wide catalog of hosted models (``google/gemma-4-31b-it``,
``deepseek/deepseek-v3.2``, ``qwen/qwen3-235b``, etc.). This client is almost a
direct mirror of :mod:`lpo.models.lmstudio` with two key differences:

1. ``Authorization: Bearer <OPENROUTER_API_KEY>`` is required.
2. Per-model pricing is pulled dynamically from
   :mod:`lpo.models.openrouter_pricing` on first use and registered with the
   shared :class:`~lpo.core.cost.CostTracker`, so :meth:`generate` calls feed
   the running cost total without any hardcoded rate table.

Reasoning-model output handling mirrors the LM Studio client: if the response
body contains a non-empty ``reasoning_content`` but empty ``content``, we log
a warning so the user knows to raise ``max_tokens`` (the same gotcha described
in the README).
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from typing import Any

import httpx

from lpo.core.cost import CostTracker
from lpo.models.base import ContentBlock, GenerationResult, ModelClient, ModelError
from lpo.models.openrouter_pricing import OpenRouterPricing, get_shared_pricing

log = logging.getLogger("lpo.models.openrouter")

RETRYABLE_STATUS = {500, 502, 503, 504}
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterClient(ModelClient):
    provider = "openrouter"

    def __init__(
        self,
        *,
        model_id: str,
        base_url: str = DEFAULT_BASE_URL,
        api_key_env: str = "OPENROUTER_API_KEY",
        http_referer: str | None = None,
        x_title: str | None = None,
        timeout_s: float = 300.0,
        max_retries: int = 3,
        cost_tracker: CostTracker | None = None,
        pricing: OpenRouterPricing | None = None,
    ) -> None:
        key = os.environ.get(api_key_env)
        if not key:
            raise ModelError(
                f"OpenRouter API key not found in env var {api_key_env!r}. "
                "Set OPENROUTER_API_KEY in your .env file."
            )
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.max_retries = max_retries
        self.cost = cost_tracker
        self._pricing = pricing or get_shared_pricing()
        self._rate_registered = False
        self._rate_lock = asyncio.Lock()

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        }
        # OpenRouter recommends these for attribution/analytics. Harmless when
        # unset but nice to include for anyone running LPO at scale.
        if http_referer:
            headers["HTTP-Referer"] = http_referer
        if x_title:
            headers["X-Title"] = x_title
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

    async def _register_rate_once(self) -> None:
        """On first call, pull this model's pricing from OpenRouter and
        register it with the CostTracker. Subsequent calls short-circuit.

        Guarded by an async lock so the rate is looked up & logged exactly
        once even when :meth:`generate` is fanned out across the eval set.
        """
        if self._rate_registered or self.cost is None:
            return
        async with self._rate_lock:
            if self._rate_registered:
                return
            in_rate, out_rate = await self._pricing.rate_for(self.model_id)
            # set_rate uses longest-prefix match; registering under the exact
            # model_id is effectively an exact-match rule.
            self.cost.set_rate(self.model_id, in_rate, out_rate)
            if in_rate == 0.0 and out_rate == 0.0:
                log.warning(
                    "OpenRouter model %r has no known pricing; cost will be recorded as $0.",
                    self.model_id,
                )
            else:
                log.info(
                    "OpenRouter pricing for %s: in=$%.4f/Mtok out=$%.4f/Mtok",
                    self.model_id, in_rate, out_rate,
                )
            self._rate_registered = True

    async def generate(
        self,
        system_prompt: str,
        user_input: str | list[ContentBlock],
        *,
        seed: int | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> GenerationResult:
        await self._register_rate_once()

        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": self._to_message_content(user_input)},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            payload["seed"] = seed

        url = f"{self.base_url}/chat/completions"
        last_err: Exception | None = None
        start = time.perf_counter()

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
            if resp.status_code == 429:
                # OpenRouter's rate-limit response; back off then retry.
                last_err = ModelError(f"HTTP 429 rate-limit: {resp.text[:200]}")
                await asyncio.sleep(_backoff(attempt) + 0.5)
                continue
            if resp.status_code >= 400:
                raise ModelError(f"HTTP {resp.status_code}: {resp.text[:500]}")
            data = resp.json()
            break
        else:
            raise ModelError(
                f"OpenRouter request failed after {self.max_retries} attempts: {last_err}"
            )

        latency = time.perf_counter() - start
        try:
            choice = data["choices"][0]
            msg = choice["message"]
            text = msg.get("content") or ""
        except (KeyError, IndexError, TypeError) as e:
            raise ModelError(f"Malformed OpenRouter response: {e}; raw={str(data)[:500]}") from e
        usage = data.get("usage") or {}
        finish = choice.get("finish_reason")
        reasoning = msg.get("reasoning_content") or ""
        if not text.strip() and reasoning.strip():
            log.warning(
                "OpenRouter %s returned empty content but reasoning_content has %d chars "
                "(finish_reason=%r). Likely a reasoning model that ran out of tokens. "
                "Raise target_models.max_tokens.",
                self.model_id, len(reasoning), finish,
            )

        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        if self.cost is not None:
            self.cost.record(self.model_id, prompt_tokens, completion_tokens)

        return GenerationResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_s=latency,
            seed=seed,
            model_id=self.model_id,
            provider=self.provider,
            raw=data,
        )


def _backoff(attempt: int) -> float:
    base = 0.5 * (2**attempt)
    return base + random.uniform(0, 0.25)
