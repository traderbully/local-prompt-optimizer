from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from lpo.models.base import ContentBlock, GenerationResult, ModelClient


@dataclass
class StubClient(ModelClient):
    """Test double that returns scripted outputs.

    ``responder`` receives ``(system_prompt, user_input, seed)`` and returns the
    response text. Default responder echoes the user input.
    """

    provider: str = "stub"
    model_id: str = "stub-model"
    responder: Callable[[str, str, int | None], str] | None = None
    calls: list[dict] = field(default_factory=list)

    async def generate(
        self,
        system_prompt: str,
        user_input: str | list[ContentBlock],
        *,
        seed: int | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> GenerationResult:
        if isinstance(user_input, list):
            user_text = " ".join(b.data for b in user_input if b.kind == "text")
        else:
            user_text = user_input
        fn = self.responder or (lambda s, u, sd: u)
        text = fn(system_prompt, user_text, seed)
        self.calls.append(
            {"system": system_prompt, "user": user_text, "seed": seed, "text": text}
        )
        return GenerationResult(
            text=text,
            prompt_tokens=len(system_prompt) // 4,
            completion_tokens=len(text) // 4,
            latency_s=0.0,
            seed=seed,
            model_id=self.model_id,
            provider=self.provider,
        )
