"""Parse the Overseer's structured response. See `LPO_SDP.md` §5.3, §6.1.

The Overseer is asked to respond with three XML-style tagged blocks:

    <analysis>free text failure analysis</analysis>
    <hypothesis>one-line expected improvement</hypothesis>
    <prompt>the full new system prompt to try next</prompt>

Parsing is defensive — if ``<prompt>`` is missing or empty, ``parse_overseer_response``
returns ``None`` so the caller can issue a clarifying retry per §6.1.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_TAG_RE = {
    "analysis": re.compile(r"<analysis>(.*?)</analysis>", re.DOTALL | re.IGNORECASE),
    "hypothesis": re.compile(r"<hypothesis>(.*?)</hypothesis>", re.DOTALL | re.IGNORECASE),
    "prompt": re.compile(r"<prompt>(.*?)</prompt>", re.DOTALL | re.IGNORECASE),
}


@dataclass
class OverseerResponse:
    analysis: str
    hypothesis: str
    new_prompt: str


def _extract(tag: str, text: str) -> str:
    m = _TAG_RE[tag].search(text)
    return m.group(1).strip() if m else ""


def parse_overseer_response(text: str) -> OverseerResponse | None:
    """Return a parsed response, or ``None`` if the prompt tag is missing/empty."""
    new_prompt = _extract("prompt", text)
    if not new_prompt:
        return None
    return OverseerResponse(
        analysis=_extract("analysis", text),
        hypothesis=_extract("hypothesis", text),
        new_prompt=new_prompt,
    )
