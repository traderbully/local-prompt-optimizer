# Example task — JSON extraction across local and OpenRouter targets

Same JSON extraction task as `example_json_extract`, configured for **Strategy B
(parallel_independent)** with two real targets:

1. `gemma-4-26b-local` — Gemma 4 26B served by LM Studio at `localhost:1234`.
2. `gemma-4-31b-openrouter` — the larger Gemma 4 31B hosted on OpenRouter.

This is the Stage 4.5 smoke test — it proves the OpenRouter provider works
end-to-end inside the same ratchet loop as a local model, and lets us eyeball
whether the 31B hosted variant actually outperforms the 26B local one on a
structured-extraction task.

Requires `OPENROUTER_API_KEY` in `.env`.

Run:

```powershell
lpo run tasks\example_or --fresh
```
