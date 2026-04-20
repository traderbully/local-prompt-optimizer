# Example task — JSON extraction across multiple targets

Same underlying task as `example_json_extract` (extract event name, date, and
location from an announcement, return JSON) but configured to exercise the
Stage 4 multi-target orchestrator: one **real** local target (Gemma-4 via LM
Studio) and one **stub** target that returns a deliberately broken "sorry I
can't parse that" response. The stub simulates a weak small model.

Running:

```
lpo run tasks\example_multi --fresh
```

The stub always scores very low (0 on most rubric criteria) while the real
local target should quickly reach 90+. The comparison report at
`tasks/example_multi/comparison/report.md` demonstrates the cross-model
orchestration end-to-end.

Config is pre-set to `target_strategy: parallel_independent`. Switch to
`unified_portable` to exercise Strategy C on the same task bundle.
