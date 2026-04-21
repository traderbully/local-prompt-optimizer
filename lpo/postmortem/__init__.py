"""Stage 8 — Postmortem analysis & remediation.

See ``STAGE_8_DESIGN.md`` at the repo root. This package adds a post-ratchet
phase that diagnoses structural issues in a completed run, proposes targeted
interventions, runs a focused validation retry, and (per Apr 21 review) may
auto-commit only prompt/seed changes — never metric or eval changes.
"""

from __future__ import annotations

from lpo.postmortem.schemas import (
    Decision,
    DecisionOutcome,
    Diagnosis,
    Evidence,
    Finding,
    FindingType,
    Intervention,
    InterventionType,
    PostmortemConfig,
    PostmortemPlan,
    Proposal,
)

__all__ = [
    "Decision",
    "DecisionOutcome",
    "Diagnosis",
    "Evidence",
    "Finding",
    "FindingType",
    "Intervention",
    "InterventionType",
    "PostmortemConfig",
    "PostmortemPlan",
    "Proposal",
]
