"""Stage 8 — Postmortem analysis & remediation.

See ``STAGE_8_DESIGN.md`` at the repo root. This package adds a post-ratchet
phase that diagnoses structural issues in a completed run, proposes targeted
interventions, runs a focused validation retry, and (per Apr 21 review) may
auto-commit only prompt/seed changes — never metric or eval changes.
"""

from __future__ import annotations

from lpo.postmortem.analyst import (
    AnalystClient,
    AnalystError,
    AnalystResult,
    build_analyst_context,
    run_analyst,
)
from lpo.postmortem.artifacts import (
    IterationArtifact,
    IterationDecision,
    IterationScores,
    RunHistoryBundle,
    load_run_history,
)
from lpo.postmortem.gate import (
    compute_deltas,
    decide_abstain,
    decide_on_retry,
    thresholds_snapshot,
)
from lpo.postmortem.patches import (
    PatchApplicationError,
    PatchSelection,
    apply_prompt_patch,
    apply_seed_reset,
    select_and_apply,
)
from lpo.postmortem.retry import FocusedRetryResult, run_focused_retry
from lpo.postmortem.runner import PostmortemMode, PostmortemResult, run_postmortem
from lpo.postmortem.schemas import (
    Decision,
    DecisionDeltas,
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
    "AnalystClient",
    "AnalystError",
    "AnalystResult",
    "Decision",
    "DecisionDeltas",
    "DecisionOutcome",
    "Diagnosis",
    "Evidence",
    "Finding",
    "FindingType",
    "FocusedRetryResult",
    "Intervention",
    "InterventionType",
    "IterationArtifact",
    "IterationDecision",
    "IterationScores",
    "PatchApplicationError",
    "PatchSelection",
    "PostmortemConfig",
    "PostmortemMode",
    "PostmortemPlan",
    "PostmortemResult",
    "Proposal",
    "RunHistoryBundle",
    "apply_prompt_patch",
    "apply_seed_reset",
    "build_analyst_context",
    "compute_deltas",
    "decide_abstain",
    "decide_on_retry",
    "load_run_history",
    "run_analyst",
    "run_focused_retry",
    "run_postmortem",
    "select_and_apply",
    "thresholds_snapshot",
]
