"""Intervention application.

Turns an :class:`Intervention` (or a subset of a :class:`PostmortemPlan`)
into a concrete patched prompt we can send to the focused-retry runner.
Only ``prompt_patch`` and ``seed_reset`` are applicable here — per the
Apr 21 review, ``metric_patch`` and ``eval_addition`` interventions are
surfaced to human review via the decision gate rather than applied.

Confidence-floor enforcement also lives here because it's part of the
"is this intervention safe to try?" question, not the "did it work?"
question that the gate answers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from lpo.postmortem.schemas import Intervention, PostmortemConfig, PostmortemPlan

log = logging.getLogger("lpo.postmortem.patches")


# ---------------------------------------------------------------------------
# Single-intervention application
# ---------------------------------------------------------------------------


class PatchApplicationError(ValueError):
    """Raised when an intervention's patch payload is self-inconsistent
    in a way the schema validator didn't catch (rare — the Pydantic
    schema covers most shapes)."""


def apply_prompt_patch(current_prompt: str, patch: dict[str, Any]) -> str:
    """Apply a ``prompt_patch`` payload to ``current_prompt``.

    Supported modes:

    * ``append``  — add ``patch.content`` after the existing prompt (with
      a newline separator). If ``patch.after_section`` is set and the
      section header is present, insert just after that section's block.
    * ``prepend`` — add ``patch.content`` before the existing prompt.
    * ``replace`` — overwrite the whole prompt with ``patch.content``.

    Whitespace normalization is intentionally minimal: we trim trailing
    whitespace on joined boundaries but don't reformat the operator's
    existing content.
    """
    mode = patch.get("mode")
    content = str(patch.get("content", ""))
    if not content.strip():
        raise PatchApplicationError("prompt_patch.content is empty after stripping.")

    if mode == "replace":
        return content.strip() + "\n"

    if mode == "prepend":
        return content.rstrip() + "\n\n" + current_prompt.lstrip()

    if mode == "append":
        after = patch.get("after_section")
        if after:
            inserted = _insert_after_section(current_prompt, str(after), content)
            if inserted is not None:
                return inserted
            # Fall through to naive append when the target section isn't
            # present — never silently drop the operator's rule.
            log.info(
                "prompt_patch.after_section=%r not found; falling back to "
                "end-of-prompt append.", after,
            )
        return current_prompt.rstrip() + "\n\n" + content.strip() + "\n"

    raise PatchApplicationError(
        f"prompt_patch.mode={mode!r} is not one of append|prepend|replace"
    )


def apply_seed_reset(patch: dict[str, Any]) -> str:
    """Apply a ``seed_reset`` payload — returns the new seed verbatim."""
    new_seed = str(patch.get("new_seed", ""))
    if not new_seed.strip():
        raise PatchApplicationError("seed_reset.new_seed is empty.")
    return new_seed.strip() + "\n"


def _insert_after_section(prompt: str, section_header: str, content: str) -> str | None:
    """Insert ``content`` just after the named section's block.

    A "section" is a block starting at a Markdown-style header line that
    equals ``section_header`` (case-sensitive; matches both ``# X`` and
    ``## X`` variants as long as the header text matches exactly). The
    block ends at the next header of equal or shallower depth, or at end
    of file. Returns None if no matching section is found.
    """
    lines = prompt.splitlines()
    header_idx = _find_section_header(lines, section_header)
    if header_idx is None:
        return None
    depth = _header_depth(lines[header_idx])
    end_idx = len(lines)
    for j in range(header_idx + 1, len(lines)):
        if _header_depth(lines[j]) is not None and _header_depth(lines[j]) <= depth:
            end_idx = j
            break
    before = "\n".join(lines[:end_idx]).rstrip()
    after = "\n".join(lines[end_idx:])
    joined = before + "\n\n" + content.strip() + "\n"
    if after.strip():
        joined += "\n" + after.lstrip()
    return joined.rstrip() + "\n"


def _find_section_header(lines: list[str], section_header: str) -> int | None:
    """Return index of the line whose markdown header matches
    ``section_header`` (just the text, without the leading #s). Returns
    None if not found."""
    normalized_target = section_header.lstrip("#").strip()
    for idx, line in enumerate(lines):
        if _header_depth(line) is None:
            continue
        text = line.lstrip("#").strip()
        if text == normalized_target or line.strip() == section_header.strip():
            return idx
    return None


def _header_depth(line: str) -> int | None:
    """Return the number of leading '#' for a Markdown header, or None
    if ``line`` isn't a header."""
    stripped = line.lstrip()
    if not stripped.startswith("#"):
        return None
    depth = 0
    for ch in stripped:
        if ch == "#":
            depth += 1
        else:
            break
    if depth == 0 or depth > 6:
        return None
    # Real headers have a space after the #s (per CommonMark). Without
    # that, treat as a non-header line.
    if len(stripped) > depth and stripped[depth] != " ":
        return None
    return depth


# ---------------------------------------------------------------------------
# Plan-level selection + application
# ---------------------------------------------------------------------------


@dataclass
class PatchSelection:
    """Result of :func:`select_and_apply`.

    ``patched_prompt`` is None when no intervention cleared the confidence
    floor — the caller should record that as an abstain with
    ``skipped_low_confidence_ids`` naming the culprits.
    """

    patched_prompt: str | None
    applied_intervention_ids: list[str]
    skipped_low_confidence_ids: list[str]
    report_only_intervention_ids: list[str]


def select_and_apply(
    plan: PostmortemPlan,
    *,
    current_best_prompt: str,
    cfg: PostmortemConfig,
) -> PatchSelection:
    """Filter auto-applicable interventions by confidence, apply in order.

    The confidence floor is per-type:
    * ``prompt_patch`` must have confidence >= ``cfg.min_confidence_prompt_patch``
    * ``seed_reset``  must have confidence >= ``cfg.min_confidence_seed_reset``

    Auto-applicable interventions that fail the floor are dropped with
    their IDs returned in ``skipped_low_confidence_ids`` so the decision
    gate's rationale can name them. A ``seed_reset`` replaces the prompt
    entirely, so if both a ``seed_reset`` and one or more ``prompt_patch``
    interventions clear the floor, the ``seed_reset`` is applied first
    and subsequent ``prompt_patch`` applications build on it. The
    ordering inside ``plan.proposal.interventions`` is preserved otherwise.
    """
    report_only = [i.id for i in plan.report_only_interventions()]

    # Stable-sort: seed_reset first (so a downstream prompt_patch can
    # stack on it), then everything else in original order.
    auto = plan.auto_applicable_interventions()
    ordered = sorted(auto, key=lambda i: (0 if i.type == "seed_reset" else 1))

    patched = current_best_prompt
    applied: list[str] = []
    skipped: list[str] = []
    for intervention in ordered:
        floor = _floor_for(intervention, cfg)
        if intervention.confidence < floor:
            skipped.append(intervention.id)
            log.info(
                "Intervention %s skipped: confidence=%.2f < floor=%.2f (type=%s).",
                intervention.id, intervention.confidence, floor, intervention.type,
            )
            continue
        patched = _apply_one(intervention, patched)
        applied.append(intervention.id)

    if not applied:
        return PatchSelection(
            patched_prompt=None,
            applied_intervention_ids=[],
            skipped_low_confidence_ids=skipped,
            report_only_intervention_ids=report_only,
        )

    return PatchSelection(
        patched_prompt=patched,
        applied_intervention_ids=applied,
        skipped_low_confidence_ids=skipped,
        report_only_intervention_ids=report_only,
    )


def _floor_for(intervention: Intervention, cfg: PostmortemConfig) -> float:
    if intervention.type == "seed_reset":
        return cfg.min_confidence_seed_reset
    if intervention.type == "prompt_patch":
        return cfg.min_confidence_prompt_patch
    # Non-auto-applicable types aren't supposed to reach this path;
    # guard defensively by treating them as 'never clears the floor'.
    return float("inf")


def _apply_one(intervention: Intervention, current_prompt: str) -> str:
    if intervention.type == "prompt_patch":
        return apply_prompt_patch(current_prompt, intervention.patch)
    if intervention.type == "seed_reset":
        return apply_seed_reset(intervention.patch)
    raise PatchApplicationError(
        f"_apply_one called with non-auto-applicable type {intervention.type!r}"
    )
