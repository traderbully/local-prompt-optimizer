"""Conversational scoring (Type C). See `LPO_SDP.md` §4.4.

The judge (an Anthropic model) is shown the **stated goal** up-front in its
system prompt, then for each iteration receives the full batch of outputs in
a single user message. Unlike rubric scoring, the judge maintains conversation
state across iterations, so it can say things like "output X improved from
last time" — this is what makes Type C appropriate for goals that are hard
to decompose into fixed criteria.

The judge returns one aggregate JSON:
    {"scores": {"<eval_id>": {"score": 0..100, "rationale": "..."}, ...}}

Budget management reuses :class:`ConversationContext` so long runs don't blow
past the judge's context window.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from lpo.config.schema import ConversationalMetric, EvalRecord, GoldRecord
from lpo.models.anthropic_client import AnthropicClient, AnthropicMessage
from lpo.overseer.context import ConversationContext
from lpo.scoring.base import ScoreResult, Scorer, ScoringContext

log = logging.getLogger("lpo.scoring.conversational")


SYSTEM_TEMPLATE = """\
You are the scoring judge for an autonomous prompt-optimization loop. A model
is being asked to accomplish a user-defined goal; for each optimization
iteration you receive all of the model's outputs and must score each one.

## Stated goal

{stated_goal}

## History summary

{history_summary}

## Scoring rules

- Each output is scored 0..100.
- Anchor 0 = wrong/harmful. Anchor 50 = partially useful. Anchor 100 = fully
  meets the stated goal.
- Be strict; use the full range. Don't cluster scores around 50.
- When an iteration's output clearly improves on a previous iteration's
  output for the same input, say so in the rationale.

## Response format — required

Respond with one JSON object. No prose, no Markdown fences.

{{"scores": {{"<eval_id>": {{"score": <int 0..100>, "rationale": "<one line>"}}, ...}}}}
"""


SUMMARY_SYSTEM = """\
You are compressing a conversational scoring history. Produce a single dense
paragraph recording: which outputs have been scored, how their scores have
moved across iterations, and any persistent failure patterns. No preamble,
no markdown headings. Under 300 words.
"""


_JSON_SPAN_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse(text: str) -> dict[str, dict[str, Any]]:
    s = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", s, re.DOTALL | re.IGNORECASE)
    if fence:
        s = fence.group(1).strip()
    try:
        data = json.loads(s)
    except json.JSONDecodeError:
        m = _JSON_SPAN_RE.search(s)
        if not m:
            raise ValueError(f"no JSON object in judge response: {text[:200]!r}")
        data = json.loads(m.group(0))
    scores = data.get("scores", data)
    if not isinstance(scores, dict):
        raise ValueError("'scores' is not an object")
    return scores


def _format_user(iteration: int, eval_records: list[EvalRecord], outputs: dict[str, str]) -> str:
    parts: list[str] = [f"ITERATION {iteration}", ""]
    for rec in eval_records:
        out = outputs.get(rec.id, "")
        parts.append(f"### id={rec.id}" + (f"  (scenario={rec.scenario})" if rec.scenario else ""))
        parts.append("Input:")
        parts.append(f"  {_oneline(rec.input)}")
        parts.append("Output:")
        parts.append(f"  {_oneline(out)}")
        parts.append("")
    parts.append("Score every output. Return the JSON object now.")
    return "\n".join(parts)


def _oneline(v: Any, max_len: int = 500) -> str:
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    s = s.replace("\n", " ").replace("\r", " ")
    if len(s) > max_len:
        s = s[: max_len - 1] + "…"
    return s


class ConversationalScorer(Scorer):
    def __init__(
        self,
        metric: ConversationalMetric,
        judge: AnthropicClient,
        *,
        context: ConversationContext | None = None,
        max_retries: int = 1,
    ) -> None:
        self.metric = metric
        self.judge = judge
        self.context = context or ConversationContext()
        self.max_retries = max_retries

    async def aclose(self) -> None:
        await self.judge.aclose()

    # Per-example scoring isn't meaningful for Type C — the SDP explicitly
    # describes it as iteration-level. We raise to surface mis-wiring early.
    async def score(
        self,
        output: str | bytes,
        gold: GoldRecord | None,
        input_record: EvalRecord,
        context: ScoringContext,
    ) -> ScoreResult:  # pragma: no cover - guarded by score_iteration override
        raise RuntimeError(
            "ConversationalScorer does not support per-example scoring. "
            "Invoke score_iteration() instead."
        )

    async def score_iteration(
        self,
        *,
        outputs: dict[str, str],
        eval_records: list[EvalRecord],
        gold_standard: dict[str, GoldRecord],
        context: ScoringContext,
    ) -> dict[str, ScoreResult]:
        user_msg = _format_user(context.iteration_index, eval_records, outputs)
        await self._maybe_summarize(pending_user=user_msg)

        raw_text: str = ""
        parsed: dict[str, dict[str, Any]] | None = None
        last_err: Exception | None = None
        for attempt in range(1 + self.max_retries):
            raw_text = await self._call(user_msg)
            try:
                parsed = _parse(raw_text)
                break
            except ValueError as e:
                last_err = e
                log.warning("Conversational judge unparseable (attempt %d): %s", attempt + 1, e)
                user_msg = (
                    "Your previous response was not valid JSON. Re-send the scores "
                    'as a single JSON object: {"scores": {"<id>": {"score": <int>, "rationale": "<str>"}, ...}}. '
                    "No prose, no Markdown fences."
                )

        if parsed is None:
            log.error("Conversational judge failed after retries: %s", last_err)
            # Record the failed exchange and give every example a 0.
            self.context.add_user(user_msg)
            self.context.add_assistant(raw_text)
            return {
                rec.id: ScoreResult(
                    aggregate=0.0,
                    rationale=f"judge unparseable: {last_err}",
                )
                for rec in eval_records
            }

        # Commit the successful exchange to context.
        self.context.add_user(_format_user(context.iteration_index, eval_records, outputs))
        self.context.add_assistant(raw_text)

        results: dict[str, ScoreResult] = {}
        for rec in eval_records:
            entry = parsed.get(rec.id)
            if isinstance(entry, dict):
                try:
                    s = float(entry.get("score", 0.0))
                except (TypeError, ValueError):
                    s = 0.0
                r = str(entry.get("rationale", ""))
            elif isinstance(entry, (int, float)):
                s = float(entry)
                r = ""
            else:
                s = 0.0
                r = "judge did not score this example"
            results[rec.id] = ScoreResult(aggregate=s, rationale=r).clamp()
        return results

    # ------------------------------------------------------------------

    def _system_prompt(self) -> str:
        return SYSTEM_TEMPLATE.format(
            stated_goal=self.metric.stated_goal.strip(),
            history_summary=self.context.summary.strip() or "(none yet — this is the start of the run)",
        )

    async def _call(self, user_text: str) -> str:
        messages = list(self.context.turns) + [AnthropicMessage(role="user", content=user_text)]
        last_exc: Exception | None = None
        for i in range(1 + self.max_retries):
            try:
                r = await self.judge.complete(
                    system=self._system_prompt(),
                    messages=messages,
                    temperature=0.0,
                    max_tokens=2048,
                )
                return r.text
            except Exception as e:  # noqa: BLE001
                last_exc = e
                if i + 1 <= self.max_retries:
                    await asyncio.sleep(0.5 * (i + 1))
        assert last_exc is not None
        raise last_exc

    async def _maybe_summarize(self, *, pending_user: str) -> None:
        if not self.context.needs_summarization(pending_user):
            return
        old, _recent = self.context._fold_candidates()
        if not old:
            return
        log.info("Conversational scorer: summarizing %d old turns.", len(old))
        walk_parts: list[str] = []
        if self.context.summary:
            walk_parts.append("PRIOR SUMMARY:\n" + self.context.summary)
        for t in old:
            walk_parts.append(f"{t.role.upper()}:\n{t.content}")
        walk = "\n\n---\n\n".join(walk_parts)
        r = await self.judge.complete(
            system=SUMMARY_SYSTEM,
            messages=[AnthropicMessage(role="user", content=walk)],
            temperature=0.0,
            max_tokens=1024,
        )
        self.context.apply_summary(r.text)
