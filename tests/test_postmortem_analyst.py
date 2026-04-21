"""Stage 8 — Postmortem Analyst tests.

Validates:
1. Context builder embeds every artifact the Analyst needs (task
   description, eval set, gold standard, metric, per-iteration prompts +
   outputs + scores + overseer analysis).
2. Happy-path single-call flow returns a validated PostmortemPlan.
3. Markdown-fence tolerance (model wrapping JSON in ```).
4. Retry flow: first attempt invalid JSON -> second attempt valid ->
   returns with retries=1 and the retry user-message contains the error.
5. Retry flow: first attempt schema violation -> second attempt valid.
6. Both attempts failing raises AnalystError with both error messages.
7. analyst_model_id auto-stamping when the Analyst forgets the field.

All tests use a stub client that records calls and returns canned text —
no real Anthropic API traffic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from lpo.postmortem.analyst import (
    AnalystError,
    AnalystResult,
    build_analyst_context,
    run_analyst,
)
from lpo.postmortem.artifacts import load_run_history
from lpo.postmortem.schemas import PostmortemConfig, PostmortemPlan


# Reuse the same fixture helpers the artifact tests set up.
from tests.test_postmortem_artifacts import _write_iteration, _write_task_files


# ---------------------------------------------------------------------------
# Stub client — records everything, returns queued responses in order.
# ---------------------------------------------------------------------------


@dataclass
class _StubResponse:
    text: str
    model_id: str = "claude-opus-4-5"


class _StubAnalystClient:
    def __init__(self, responses: list[_StubResponse] | list[str]):
        self._queue: list[_StubResponse] = [
            r if isinstance(r, _StubResponse) else _StubResponse(text=r)
            for r in responses
        ]
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self,
        *,
        system: str,
        messages: list[Any],
        temperature: float = 0.0,
        max_tokens: int = 8192,
    ) -> _StubResponse:
        self.calls.append({
            "system": system,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        })
        if not self._queue:
            raise AssertionError("StubAnalystClient queue exhausted")
        return self._queue.pop(0)


# ---------------------------------------------------------------------------
# Canned plans — the minimum each happy-path test needs.
# ---------------------------------------------------------------------------


def _valid_plan_json(
    *,
    analyst_model_id: str = "claude-opus-4-5",
    task_id: str = "test",
    slug: str = "stub",
) -> str:
    return json.dumps({
        "diagnosis": {
            "findings": [
                {
                    "id": "F1",
                    "type": "scenario_blindspot",
                    "severity": "high",
                    "confidence": 0.9,
                    "summary": "ex002 never improved.",
                    "evidence": {
                        "iterations": [1, 2, 3],
                        "example_ids": ["ex002"],
                        "scenarios": ["hard"],
                        "score_breakdown": {
                            "ex002": {"iter_1": 43.3, "iter_2": 50.0, "iter_3": 46.0},
                        },
                    },
                    "root_cause_hypothesis": "Prompt has no 'hard' rule.",
                }
            ],
            "metric_observations": [],
            "overseer_drift_observations": [],
            "analyst_model_id": analyst_model_id,
            "task_id": task_id,
            "slug": slug,
        },
        "proposal": {
            "interventions": [
                {
                    "id": "I1",
                    "type": "prompt_patch",
                    "fixes": ["F1"],
                    "confidence": 0.8,
                    "summary": "Add 'hard' rule.",
                    "patch": {"mode": "append", "content": "- For hard inputs, do X."},
                }
            ],
            "rationale": "One patch addresses the only finding.",
            "human_review_summary": "",
        },
    })


def _plan_missing_evidence_json() -> str:
    # A plan that violates the evidence invariant — score_breakdown empty.
    return json.dumps({
        "diagnosis": {
            "findings": [
                {
                    "id": "F1",
                    "type": "scenario_blindspot",
                    "severity": "high",
                    "confidence": 0.9,
                    "summary": "s",
                    "evidence": {
                        "iterations": [1],
                        "example_ids": ["ex002"],
                        "scenarios": [],
                        "score_breakdown": {},
                    },
                    "root_cause_hypothesis": "h",
                }
            ],
            "metric_observations": [],
            "overseer_drift_observations": [],
            "analyst_model_id": "x",
            "task_id": "test",
            "slug": "stub",
        },
        "proposal": {
            "interventions": [
                {
                    "id": "I1",
                    "type": "prompt_patch",
                    "fixes": ["F1"],
                    "confidence": 0.8,
                    "summary": "s",
                    "patch": {"mode": "append", "content": "x"},
                }
            ],
            "rationale": "r",
        },
    })


# ---------------------------------------------------------------------------
# Fixtures — minimal built-run the Analyst can reason over.
# ---------------------------------------------------------------------------


@pytest.fixture
def bundle(tmp_path: Path):
    root = tmp_path / "task1"
    root.mkdir()
    _write_task_files(root, slug="stub")
    history = root / "runs" / "stub" / "history"
    _write_iteration(
        history / "iter_0001",
        index=1,
        prompt="seed v1",
        aggregate=43.3,
        per_example={"ex001": 43.3, "ex002": 43.3},
        per_scenario={"easy": 43.3, "hard": 43.3},
        decision="initial",
        delta=43.3,
    )
    _write_iteration(
        history / "iter_0002",
        index=2,
        prompt="v2",
        aggregate=55.0,
        per_example={"ex001": 60.0, "ex002": 50.0},
        per_scenario={"easy": 60.0, "hard": 50.0},
        decision="accepted",
        delta=11.7,
        overseer_analysis="tried adding rules",
    )
    return load_run_history(root, "stub")


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------


class TestBuildAnalystContext:
    def test_includes_task_description_and_metric(self, bundle):
        ctx = build_analyst_context(bundle)
        data = json.loads(ctx)
        assert data["task_description"] == "A task."
        assert data["metric"]["type"] == "deterministic"
        assert data["slug"] == "stub"

    def test_includes_eval_set_and_gold_standard(self, bundle):
        data = json.loads(build_analyst_context(bundle))
        ids = [r["id"] for r in data["eval_set"]]
        assert set(ids) == {"ex001", "ex002"}
        assert data["gold_standard"] == {"ex001": "a1", "ex002": "a2"}

    def test_includes_every_iteration_with_full_scores(self, bundle):
        data = json.loads(build_analyst_context(bundle))
        assert len(data["iterations"]) == 2
        it2 = data["iterations"][1]
        assert it2["index"] == 2
        assert it2["scores"]["aggregate"] == pytest.approx(55.0)
        assert it2["scores"]["per_example"] == {"ex001": 60.0, "ex002": 50.0}
        assert it2["overseer_analysis"] == "tried adding rules"
        assert it2["decision"]["decision"] == "accepted"

    def test_is_valid_json_for_all_artifacts(self, bundle):
        # The context is handed to the model as a JSON string — it must
        # be parseable. Regression guard against a future edit that
        # introduces non-serializable objects.
        ctx = build_analyst_context(bundle)
        json.loads(ctx)  # must not raise


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestAnalystHappyPath:
    @pytest.mark.asyncio
    async def test_single_call_returns_validated_plan(self, bundle):
        client = _StubAnalystClient([_valid_plan_json()])
        cfg = PostmortemConfig()
        result = await run_analyst(bundle, cfg=cfg, client=client)
        assert isinstance(result, AnalystResult)
        assert isinstance(result.plan, PostmortemPlan)
        assert result.retries == 0
        assert len(client.calls) == 1
        assert result.plan.diagnosis.findings[0].id == "F1"
        assert result.plan.proposal.interventions[0].fixes == ["F1"]

    @pytest.mark.asyncio
    async def test_system_prompt_contains_invariant_language(self, bundle):
        # The prompt itself is part of the contract — if someone
        # accidentally deletes the invariant section we want to notice.
        client = _StubAnalystClient([_valid_plan_json()])
        await run_analyst(bundle, cfg=PostmortemConfig(), client=client)
        system = client.calls[0]["system"]
        assert "Evidence" in system
        assert "Intervention provenance" in system
        assert "metric_patch" in system and "report-only" in system.lower()
        # Finding type closed-set must be present.
        for t in ("scenario_blindspot", "prompt_gap", "metric_mismatch"):
            assert t in system

    @pytest.mark.asyncio
    async def test_user_message_embeds_run_context(self, bundle):
        client = _StubAnalystClient([_valid_plan_json()])
        await run_analyst(bundle, cfg=PostmortemConfig(), client=client)
        msgs = client.calls[0]["messages"]
        assert len(msgs) == 1
        content = getattr(msgs[0], "content", None) or msgs[0]["content"]
        # The context was built by build_analyst_context; just spot-check.
        assert "ex002" in content
        assert "task_description" in content

    @pytest.mark.asyncio
    async def test_markdown_fences_are_stripped(self, bundle):
        raw = "```json\n" + _valid_plan_json() + "\n```"
        client = _StubAnalystClient([raw])
        result = await run_analyst(bundle, cfg=PostmortemConfig(), client=client)
        assert result.retries == 0
        assert result.plan.diagnosis.findings[0].id == "F1"

    @pytest.mark.asyncio
    async def test_bare_fence_without_language_marker_also_stripped(self, bundle):
        raw = "```\n" + _valid_plan_json() + "\n```"
        client = _StubAnalystClient([raw])
        result = await run_analyst(bundle, cfg=PostmortemConfig(), client=client)
        assert result.retries == 0


# ---------------------------------------------------------------------------
# Retry flow
# ---------------------------------------------------------------------------


class TestAnalystRetry:
    @pytest.mark.asyncio
    async def test_invalid_json_then_valid(self, bundle):
        client = _StubAnalystClient(["this is not json", _valid_plan_json()])
        result = await run_analyst(bundle, cfg=PostmortemConfig(), client=client)
        assert result.retries == 1
        assert len(client.calls) == 2

    @pytest.mark.asyncio
    async def test_retry_message_contains_error_feedback(self, bundle):
        # The second call's user message must include the error so the
        # model can self-correct. This is the whole point of the retry.
        client = _StubAnalystClient(["not json at all", _valid_plan_json()])
        await run_analyst(bundle, cfg=PostmortemConfig(), client=client)
        second_call_messages = client.calls[1]["messages"]
        # First message = original context, second = assistant's bad reply,
        # third = feedback pointing out the error.
        assert len(second_call_messages) == 3
        feedback_content = getattr(second_call_messages[2], "content", None) or second_call_messages[2]["content"]
        assert "did not validate" in feedback_content or "schema" in feedback_content.lower()

    @pytest.mark.asyncio
    async def test_schema_violation_then_valid(self, bundle):
        # First response parses as JSON but violates the evidence invariant;
        # retry succeeds. Validates that schema-level failures also trip
        # the retry path (not just json.JSONDecodeError).
        client = _StubAnalystClient([_plan_missing_evidence_json(), _valid_plan_json()])
        result = await run_analyst(bundle, cfg=PostmortemConfig(), client=client)
        assert result.retries == 1
        # Retry feedback should mention score_breakdown — the specific
        # field that failed.
        feedback_content = (
            getattr(client.calls[1]["messages"][2], "content", None)
            or client.calls[1]["messages"][2]["content"]
        )
        assert "score_breakdown" in feedback_content

    @pytest.mark.asyncio
    async def test_both_attempts_fail_raises_analyst_error(self, bundle):
        client = _StubAnalystClient(["bad 1", "bad 2"])
        with pytest.raises(AnalystError) as exc:
            await run_analyst(bundle, cfg=PostmortemConfig(), client=client)
        # Error message must mention both attempts so debugging a failed
        # postmortem is possible without re-running.
        assert "First error" in str(exc.value)
        assert "Second error" in str(exc.value)

    @pytest.mark.asyncio
    async def test_both_attempts_fail_calls_client_exactly_twice(self, bundle):
        client = _StubAnalystClient(["bad", "still bad"])
        with pytest.raises(AnalystError):
            await run_analyst(bundle, cfg=PostmortemConfig(), client=client)
        assert len(client.calls) == 2  # one attempt, one retry, stop.


# ---------------------------------------------------------------------------
# analyst_model_id auto-stamping
# ---------------------------------------------------------------------------


class TestAnalystModelIdStamping:
    @pytest.mark.asyncio
    async def test_prefers_client_result_model_id(self, bundle):
        # When the client reports its own model_id on the result, that
        # wins — the config's configured id is only a fallback.
        client = _StubAnalystClient([_StubResponse(
            text=_valid_plan_json(analyst_model_id="placeholder"),
            model_id="claude-opus-1.2-real",
        )])
        result = await run_analyst(
            bundle,
            cfg=PostmortemConfig(analyst_model_id="config-fallback"),
            client=client,
        )
        assert result.model_id == "claude-opus-1.2-real"

    @pytest.mark.asyncio
    async def test_falls_back_to_config_model_id(self, bundle):
        # Stub that doesn't expose model_id on the result object.
        class _ModellessResult:
            text = _valid_plan_json(analyst_model_id="should-not-see-this")

        class _ModellessClient:
            async def complete(self, *, system, messages, temperature=0.0, max_tokens=8192):
                return _ModellessResult()

        result = await run_analyst(
            bundle,
            cfg=PostmortemConfig(analyst_model_id="config-fallback"),
            client=_ModellessClient(),
        )
        # The stamp on the plan comes from the model's own output first;
        # analyst_model_id on AnalystResult falls back to config when
        # the client didn't surface one.
        assert result.model_id == "config-fallback"


# ---------------------------------------------------------------------------
# Output-shape plumbing — real-Claude-output forgiveness (Stage 8 bug fix)
# ---------------------------------------------------------------------------


def _plan_with_extra_fields_json() -> str:
    """A plan whose models carry fields NOT in our schema. Mirrors what
    Claude Opus actually emits in practice — it reliably adds `priority`,
    `affected_scenarios`, `confidence_rationale`, and similar narration
    fields on findings, `version` or `notes` at the plan level, etc.
    """
    return json.dumps({
        "version": 1,  # top-level extra
        "notes": "Generated by Postmortem Analyst v2.",  # top-level extra
        "diagnosis": {
            "findings": [
                {
                    "id": "F1",
                    "type": "scenario_blindspot",
                    "severity": "high",
                    "confidence": 0.9,
                    "confidence_rationale": "Three scenarios, same signature.",  # extra
                    "affected_scenarios": ["gui_launch_with_content"],  # extra
                    "priority": 1,  # extra
                    "summary": "Three content scenarios consistently scored 0.",
                    "evidence": {
                        "iterations": [1, 2],
                        "example_ids": ["ex001"],
                        "scenarios": ["gui_launch_with_content"],
                        "score_breakdown": {
                            "ex001": {"iter_1": 0.0, "iter_2": 0.0}
                        },
                        "notes": "Budget truncation.",  # extra on Evidence
                    },
                    "root_cause_hypothesis": "Prompt lacks content-escaping rules.",
                }
            ],
            "metric_observations": [],
            "overseer_drift_observations": [],
            "analyst_model_id": "claude-opus-4-5",
            "task_id": "t",
            "slug": "s",
            "extra_metadata": {"generator": "v2"},  # extra on Diagnosis
        },
        "proposal": {
            "interventions": [
                {
                    "id": "I1",
                    "type": "prompt_patch",
                    "fixes": ["F1"],
                    "confidence": 0.85,
                    "confidence_rationale": "Direct fix for the failure class.",
                    "reversible": True,  # extra
                    "risk_level": "low",  # extra
                    "summary": "Add content-escaping rules.",
                    "expected_impact": {
                        "global": [5, 12],
                        "remediation": [15, 35],
                        "notes": "Assumes no regression on passing scenarios.",  # extra
                    },
                    "patch": {
                        "mode": "append",
                        "content": "- Content with single quotes: escape them.",
                    },
                }
            ],
            "rationale": "Directly addresses the only failing class.",
            "human_review_summary": "",
            "priority_order": ["I1"],  # extra on Proposal
        },
    })


class TestAnalystAcceptsExtraFields:
    """The Apr 21 hard invariants are about positive content (evidence
    must cite specific iterations + examples + scores; interventions must
    reference finding IDs). They are NOT about the absence of extra
    fields. Real Claude output reliably carries descriptive extras; our
    schema should drop them silently rather than burn the single retry
    budget on a cosmetic re-emit.
    """

    @pytest.mark.asyncio
    async def test_extra_fields_on_every_layer_accepted_without_retry(self, bundle):
        client = _StubAnalystClient([_plan_with_extra_fields_json()])
        result = await run_analyst(bundle, cfg=PostmortemConfig(), client=client)
        # No retry needed — extras are ignored, positive invariants still hold.
        assert result.retries == 0
        plan = result.plan
        # The positive invariants passed: one finding with full evidence,
        # one intervention with a `fixes` reference.
        assert len(plan.diagnosis.findings) == 1
        assert plan.diagnosis.findings[0].id == "F1"
        assert plan.diagnosis.findings[0].evidence.example_ids == ["ex001"]
        assert plan.proposal.interventions[0].fixes == ["F1"]

    def test_schema_still_rejects_missing_positive_fields(self):
        """Sanity check: extra='ignore' is NOT the same as 'anything goes'.
        A plan missing a required field (e.g. no evidence block) still
        fails validation. We didn't accidentally disable the real
        contract."""
        from pydantic import ValidationError
        from lpo.postmortem.schemas import Finding

        with pytest.raises(ValidationError):
            Finding.model_validate({
                "id": "F1",
                "type": "scenario_blindspot",
                "severity": "high",
                "confidence": 0.9,
                "summary": "s",
                # evidence missing — required
                "root_cause_hypothesis": "h",
            })


class TestAnalystToleratesProseWrappedJson:
    """Claude Opus frequently prefaces structured output with narration
    despite the 'respond with exactly one JSON object' instruction. We
    recover instead of burning the retry budget.
    """

    @pytest.mark.asyncio
    async def test_prose_prefix_stripped(self, bundle):
        raw = "Here is the postmortem analysis:\n\n" + _valid_plan_json()
        client = _StubAnalystClient([raw])
        result = await run_analyst(bundle, cfg=PostmortemConfig(), client=client)
        assert result.retries == 0

    @pytest.mark.asyncio
    async def test_prose_suffix_stripped(self, bundle):
        raw = _valid_plan_json() + "\n\nLet me know if you want more detail."
        client = _StubAnalystClient([raw])
        result = await run_analyst(bundle, cfg=PostmortemConfig(), client=client)
        assert result.retries == 0

    @pytest.mark.asyncio
    async def test_prose_prefix_and_suffix_and_fence_combined(self, bundle):
        raw = (
            "I'll start the postmortem now.\n\n"
            "```json\n"
            + _valid_plan_json()
            + "\n```\n\n"
            "This addresses the three failing scenarios."
        )
        client = _StubAnalystClient([raw])
        result = await run_analyst(bundle, cfg=PostmortemConfig(), client=client)
        assert result.retries == 0


class TestExtractFirstJsonObject:
    """Direct unit tests for the balanced-brace scanner. These cover
    edge cases the integration tests above don't exercise — braces
    inside JSON string values, escaped quotes, no-match returns None.
    """

    def test_returns_none_when_no_object_present(self):
        from lpo.postmortem.analyst import _extract_first_json_object
        assert _extract_first_json_object("no braces here at all") is None

    def test_braces_inside_strings_do_not_confuse_depth_counter(self):
        from lpo.postmortem.analyst import _extract_first_json_object
        # The `}` inside the string value must NOT close the outer object.
        src = 'prefix {"k": "value with } and { in it"} suffix'
        extracted = _extract_first_json_object(src)
        assert extracted == '{"k": "value with } and { in it"}'

    def test_escaped_quotes_inside_strings(self):
        from lpo.postmortem.analyst import _extract_first_json_object
        # Escaped quote in the middle of a string shouldn't toggle in_string.
        src = r'prefix {"k": "has \"quotes\" and { brace"} suffix'
        extracted = _extract_first_json_object(src)
        assert extracted == r'{"k": "has \"quotes\" and { brace"}'

    def test_nested_objects_balance_correctly(self):
        from lpo.postmortem.analyst import _extract_first_json_object
        src = 'x {"a": {"b": {"c": 1}}, "d": 2} y'
        extracted = _extract_first_json_object(src)
        assert extracted == '{"a": {"b": {"c": 1}}, "d": 2}'
        # And it parses as valid JSON, proving the extraction is correct.
        assert json.loads(extracted) == {"a": {"b": {"c": 1}}, "d": 2}
