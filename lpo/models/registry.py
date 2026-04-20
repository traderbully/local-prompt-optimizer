"""Provider registry. Maps ``TargetModelConfig.provider`` to a client.

OpenRouter and stub clients need a ``CostTracker`` handle so their calls
contribute to the shared run total; :mod:`lpo.core.target_factory` supplies
it and the registry plumbs it through via ``**extras``.
"""

from __future__ import annotations

from typing import Any

from lpo.config.schema import TargetModelConfig
from lpo.models.base import ModelClient
from lpo.models.lmstudio import LMStudioClient


def build_client(cfg: TargetModelConfig, **extras: Any) -> ModelClient:
    if cfg.provider in ("lmstudio", "openai_compatible"):
        return LMStudioClient(
            base_url=cfg.base_url,
            model_id=cfg.model_id,
            api_key_env=cfg.api_key_env,
        )
    if cfg.provider == "openrouter":
        from lpo.models.openrouter import DEFAULT_BASE_URL, OpenRouterClient

        base_url = cfg.base_url or DEFAULT_BASE_URL
        # Default base_url from the schema is the LM Studio one; swap it out
        # when the user didn't override it for OpenRouter.
        if base_url.startswith("http://localhost"):
            base_url = DEFAULT_BASE_URL
        return OpenRouterClient(
            model_id=cfg.model_id,
            base_url=base_url,
            api_key_env=cfg.api_key_env or "OPENROUTER_API_KEY",
            http_referer=cfg.http_referer,
            x_title=cfg.x_title,
            cost_tracker=extras.get("cost_tracker"),
        )
    if cfg.provider == "stub":
        from lpo.models.stub import StubClient

        return StubClient(cfg)
    raise ValueError(f"Unknown target provider: {cfg.provider!r}")
