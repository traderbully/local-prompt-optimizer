"""Model client abstraction. See `LPO_SDP.md` §5.2."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


class ModelError(RuntimeError):
    """Raised for non-retryable model failures."""


@dataclass
class ContentBlock:
    """Multimodal input block. For Stage 1 only text is exercised."""

    kind: str  # "text" | "image_base64" | "image_url"
    data: str


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    seed: int | None = None
    model_id: str = ""
    provider: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


class ModelClient(ABC):
    """Abstract async model client."""

    provider: str = "abstract"
    model_id: str = ""

    @abstractmethod
    async def generate(
        self,
        system_prompt: str,
        user_input: str | list[ContentBlock],
        *,
        seed: int | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> GenerationResult: ...

    async def aclose(self) -> None:  # pragma: no cover - default no-op
        return None
