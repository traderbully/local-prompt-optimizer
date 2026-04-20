"""LLM-as-judge rubric scoring (Type B). See `LPO_SDP.md` §4.4.

The judge is called once per eval example with a fixed rubric. It returns a
JSON object mapping each criterion name to ``{score, rationale}``. Scores are
0..100 per criterion, combined by weight to produce the aggregate.

Judge calls are issued concurrently (bounded by ``concurrency``) and funnel
cost into the shared :class:`CostTracker`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from lpo.config.schema import EvalRecord, GoldRecord, RubricCriterion, RubricMetric
from lpo.models.anthropic_client import AnthropicClient, AnthropicMessage
from lpo.scoring.base import ScoreResult, Scorer, ScoringContext

log = logging.getLogger("lpo.scoring.rubric")


JUDGE_SYSTEM = """\
You are an impartial evaluation judge. For each example you will receive an
input, (optionally) a gold-standard reference, and the model's actual output.
Score the output against a fixed rubric.

Scoring rules:
- Every criterion is scored on the integer range 0..100.
- Anchors are guidance, not hard cutoffs — interpolate when appropriate.
- Be strict but fair. Do not anchor your scores around 50; use the full range.
- If gold is provided, treat it as ONE valid reference, not the only one.

Respond with a single JSON object:
{"scores": {"<criterion_name>": {"score": <int 0..100>, "rationale": "<one line>"}, ...}}

No prose before or after the JSON. No Markdown fences.
"""


def _format_rubric(criteria: list[RubricCriterion]) -> str:
    lines: list[str] = []
    for c in criteria:
        lines.append(f"### {c.name} (weight={c.weight})")
        lines.append(c.description)
        if c.anchors:
            lines.append("Anchors:")
            for k in sorted(c.anchors):
                lines.append(f"  {k}: {c.anchors[k]}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _format_user(
    *,
    input_value: Any,
    output: str,
    gold: GoldRecord | None,
    criteria: list[RubricCriterion],
) -> str:
    parts = ["## Criteria", "", _format_rubric(criteria), "", "## Input", str(input_value)]
    if gold is not None:
        parts.extend(["", "## Gold standard reference", json.dumps(gold.output, ensure_ascii=False)])
    parts.extend(["", "## Actual output", output, "", "Respond now with the JSON scores object."])
    return "\n".join(parts)


_JSON_SPAN_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_judge_response(text: str, criterion_names: list[str]) -> dict[str, tuple[float, str]]:
    """Return ``{criterion_name: (score, rationale)}``. Tolerant of prose wrap."""
    stripped = text.strip()
    # Allow ```json fences.
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, re.DOTALL | re.IGNORECASE)
    if fence:
        stripped = fence.group(1).strip()
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        m = _JSON_SPAN_RE.search(stripped)
        if not m:
            raise ValueError(f"no JSON object in judge response: {text[:200]!r}")
        data = json.loads(m.group(0))
    scores = data.get("scores", data)
    if not isinstance(scores, dict):
        raise ValueError("judge response 'scores' is not an object")
    out: dict[str, tuple[float, str]] = {}
    for name in criterion_names:
        entry = scores.get(name)
        if entry is None:
            out[name] = (0.0, "judge did not return this criterion")
            continue
        if isinstance(entry, (int, float)):
            out[name] = (float(entry), "")
            continue
        if isinstance(entry, dict):
            s = entry.get("score")
            if s is None:
                out[name] = (0.0, "judge response missing 'score'")
                continue
            out[name] = (float(s), str(entry.get("rationale", "")))
            continue
        out[name] = (0.0, f"unexpected shape {type(entry).__name__}")
    return out


class RubricScorer(Scorer):
    def __init__(
        self,
        metric: RubricMetric,
        judge: AnthropicClient,
        *,
        concurrency: int = 4,
        max_retries: int = 1,
    ) -> None:
        if not metric.criteria:
            raise ValueError("RubricScorer requires at least one criterion")
        self.metric = metric
        self.judge = judge
        self.criteria = metric.criteria
        self.criterion_names = [c.name for c in self.criteria]
        self._weights = {c.name: c.weight for c in self.criteria}
        self._total_weight = sum(c.weight for c in self.criteria) or 1.0
        self.semaphore = asyncio.Semaphore(max(1, concurrency))
        self.max_retries = max_retries

    async def aclose(self) -> None:
        await self.judge.aclose()

    async def score(
        self,
        output: str | bytes,
        gold: GoldRecord | None,
        input_record: EvalRecord,
        context: ScoringContext,
    ) -> ScoreResult:
        text = output.decode("utf-8", errors="replace") if isinstance(output, bytes) else output
        user_msg = _format_user(
            input_value=input_record.input,
            output=text,
            gold=gold,
            criteria=self.criteria,
        )
        async with self.semaphore:
            result_text = await self._call_with_retry(user_msg)

        try:
            parsed = _parse_judge_response(result_text, self.criterion_names)
        except ValueError as e:
            log.warning("Judge response unparseable for %s: %s", input_record.id, e)
            return ScoreResult(
                aggregate=0.0,
                per_criterion={n: 0.0 for n in self.criterion_names},
                rationale=f"judge response unparseable: {e}",
            ).clamp()

        per = {n: parsed[n][0] for n in self.criterion_names}
        weighted = sum(per[n] * self._weights[n] for n in self.criterion_names) / self._total_weight
        bits: list[str] = []
        for n in self.criterion_names:
            sc, rat = parsed[n]
            bit = f"{n}={sc:.0f}"
            if rat:
                bit += f" [{rat}]"
            bits.append(bit)
        return ScoreResult(aggregate=weighted, per_criterion=per, rationale="; ".join(bits)).clamp()

    async def _call_with_retry(self, user_msg: str) -> str:
        last_exc: Exception | None = None
        attempts = 1 + max(0, self.max_retries)
        for i in range(attempts):
            try:
                r = await self.judge.complete(
                    system=JUDGE_SYSTEM,
                    messages=[AnthropicMessage(role="user", content=user_msg)],
                    temperature=0.0,
                    max_tokens=1024,
                )
                return r.text
            except Exception as e:  # noqa: BLE001
                last_exc = e
                if i + 1 < attempts:
                    log.warning("Judge call failed (%s); retrying...", e)
                    await asyncio.sleep(0.5 * (i + 1))
        assert last_exc is not None
        raise last_exc
