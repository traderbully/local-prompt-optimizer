"""End-to-end simulation: feed the winning LPO prompt to Qwen 3.6 Plus (OpenRouter)
with the original ``greeting.txt`` request, print the JSON response.

This does NOT execute the resulting command — that's a separate phase so the
operator can review before touching the filesystem.

Run:
    python scripts/simulate_greeting_txt.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

REPO = Path(__file__).resolve().parent.parent
PROMPT_PATH = (
    REPO
    / "tasks"
    / "windows_file_ops_reliability_rubric"
    / "runs"
    / "qwen3-235b-2507"
    / "prompt.txt.best"
)
MODEL = "qwen/qwen3.6-plus"
USER_REQUEST = (
    "Create a text file on the Windows PC desktop with the content "
    "'Hello, Matt and Scott' and open it in notepad so I can see it."
)
OUT_JSON = Path(os.environ.get("TEMP", "/tmp")) / "lpo_sim_response.json"


def main() -> int:
    load_dotenv(REPO / ".env")
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set", file=sys.stderr)
        return 2

    system_prompt = PROMPT_PATH.read_text(encoding="utf-8")
    print(f"=== winning prompt loaded: {len(system_prompt)} chars from {PROMPT_PATH.name} ===\n")
    print(f"=== USER REQUEST ===\n{USER_REQUEST}\n")
    print(f"=== calling {MODEL} via OpenRouter ===")

    start = time.monotonic()
    try:
        resp = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/traderbully/local-prompt-optimizer",
                "X-Title": "LPO end-to-end simulation",
            },
            json={
                "model": MODEL,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": USER_REQUEST},
                ],
            },
            timeout=60.0,
        )
    except Exception as e:
        print(f"  network error: {e}", file=sys.stderr)
        return 3

    elapsed = time.monotonic() - start
    if resp.status_code != 200:
        print(f"  HTTP {resp.status_code}: {resp.text}")
        return 4

    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    in_tok = usage.get("prompt_tokens", 0)
    out_tok = usage.get("completion_tokens", 0)
    # Published pricing: $0.325 / 1M input, $1.95 / 1M output
    cost = (in_tok * 0.325 + out_tok * 1.95) / 1_000_000
    print(
        f"  elapsed: {elapsed:.1f}s | in_tok: {in_tok} | out_tok: {out_tok} | "
        f"cost: ${cost:.6f}"
    )
    print(f"\n=== RAW MODEL OUTPUT ===\n{content}\n")

    # Strip any markdown fences the model might have added despite instructions
    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        # remove first fence line and any closing fence
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    print("=== PARSED JSON ===")
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as e:
        print(f"  PARSE FAILED: {e}")
        print(f"  first 200 chars of stripped: {stripped[:200]!r}")
        return 5

    print(f"command        : {parsed.get('command', '(missing)')}")
    print(f"verify_command : {parsed.get('verify_command', '(missing)')}")
    print(f"user_message   : {parsed.get('user_message', '(missing)')}")

    OUT_JSON.write_text(json.dumps(parsed, indent=2), encoding="utf-8")
    print(f"\n  saved to {OUT_JSON} for execution phase")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
