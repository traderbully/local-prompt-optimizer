"""Stub target client for smoke tests and offline validation.

Useful when only one local model is available but Stage 4 multi-target
orchestration must be exercised end-to-end. The stub produces deterministic,
lower-quality outputs so the comparison report has something non-trivial
to show.

Four modes, selected via ``TargetModelConfig.stub_mode``:

- ``fixed`` — always returns ``stub_fixed_text`` (default: empty string).
- ``echo`` — returns the user_input verbatim (tends to score low on most
  rubrics because it isn't a transformation).
- ``truncate`` — returns the first ``stub_truncate_chars`` characters of the
  input. Good simulation of "small model with weak instruction following."
- ``prefix`` — returns ``stub_prefix`` concatenated with the input. Simulates
  a model that adds unwanted preamble.
"""

from __future__ import annotations

import time
from typing import Any

from lpo.config.schema import TargetModelConfig
from lpo.models.base import ContentBlock, GenerationResult, ModelClient, ModelError


class StubClient(ModelClient):
    provider = "stub"

    def __init__(self, cfg: TargetModelConfig) -> None:
        self.cfg = cfg
        self.model_id = cfg.model_id
        self.mode = (cfg.stub_mode or "fixed").lower()
        self.fixed_text = cfg.stub_fixed_text or ""
        self.prefix = cfg.stub_prefix or ""
        self.truncate_chars = cfg.stub_truncate_chars or 40
        if self.mode not in {"fixed", "echo", "truncate", "prefix"}:
            raise ModelError(f"Unknown stub_mode: {self.mode!r}")

    async def aclose(self) -> None:
        return None

    @staticmethod
    def _flatten(user_input: str | list[ContentBlock]) -> str:
        if isinstance(user_input, str):
            return user_input
        parts: list[str] = []
        for block in user_input:
            if block.kind == "text":
                parts.append(block.data)
        return "\n".join(parts)

    async def generate(
        self,
        system_prompt: str,
        user_input: str | list[ContentBlock],
        *,
        seed: int | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> GenerationResult:
        t0 = time.perf_counter()
        user_text = self._flatten(user_input)
        if self.mode == "fixed":
            text = self.fixed_text
        elif self.mode == "echo":
            text = user_text
        elif self.mode == "truncate":
            text = user_text[: self.truncate_chars]
        else:  # prefix
            text = f"{self.prefix}{user_text}"
        latency = time.perf_counter() - t0
        return GenerationResult(
            text=text,
            prompt_tokens=len(system_prompt) // 4 + len(user_text) // 4,
            completion_tokens=len(text) // 4,
            latency_s=latency,
            seed=seed,
            model_id=self.model_id,
            provider=self.provider,
            raw={"stub_mode": self.mode},
        )


def _extra(cfg: TargetModelConfig, name: str) -> Any:
    """Defensive accessor: stub-specific fields live as extra fields."""
    return getattr(cfg, name, None)
