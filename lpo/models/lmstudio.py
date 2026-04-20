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

    async def generate(
        self,
        system_prompt: str,
        user_input: str | list[ContentBlock],
        *,
        seed: int | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> GenerationResult:
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
            if resp.status_code >= 400:
                raise ModelError(f"HTTP {resp.status_code}: {resp.text[:500]}")
            data = resp.json()
            break
        else:
            raise ModelError(f"LM Studio request failed after {self.max_retries} attempts: {last_err}")

        latency = time.perf_counter() - start
        try:
            choice = data["choices"][0]
            msg = choice["message"]
            text = msg.get("content") or ""
        except (KeyError, IndexError, TypeError) as e:
            raise ModelError(f"Malformed LM Studio response: {e}; raw={data}") from e
        usage = data.get("usage") or {}
        finish = choice.get("finish_reason")
        reasoning = msg.get("reasoning_content") or ""
        if not text.strip():
            # The model returned empty content. Almost always one of two causes:
            #   (a) Reasoning model that spent all tokens thinking — `reasoning_content`
            #       is populated and finish_reason="length". Raise max_tokens.
            #   (b) Server/tokenizer misconfiguration.
            hint = (
                f"finish_reason={finish!r}, reasoning_tokens={usage.get('completion_tokens_details', {}).get('reasoning_tokens')}"
            )
            if reasoning.strip():
                log.warning(
                    "LM Studio returned empty content but reasoning_content has %d chars "
                    "(%s). Likely a reasoning model that ran out of tokens before the final answer. "
                    "Increase target_models.max_tokens.",
                    len(reasoning),
                    hint,
                )
            else:
                log.warning("LM Studio returned empty content (%s). Raw: %s", hint, str(data)[:500])
        return GenerationResult(
            text=text,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            latency_s=latency,
            seed=seed,
            model_id=self.model_id,
            provider=self.provider,
            raw=data,
        )


def _backoff(attempt: int) -> float:
    base = 0.5 * (2**attempt)
    return base + random.uniform(0, 0.25)
