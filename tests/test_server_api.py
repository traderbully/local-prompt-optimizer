"""Stage 5 — FastAPI backend + WebSocket tests.

These exercise the server against the filesystem fixture produced by the
existing ``tasks/example_json_extract`` bundle (re-used so we don't have to
regenerate gold standards). The live-run path uses the stub target so no
network or LM Studio dependency is required.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from lpo.server.api import create_app


FIXTURE_SRC = Path(__file__).resolve().parent.parent / "tasks" / "example_json_extract"


def _stub_task_bundle(tmp_path: Path) -> Path:
    """Copy the real example into a tmp tasks/ dir and swap the target to the
    stub provider so the live-run test is offline and fast."""
    tasks_root = tmp_path / "tasks"
    dst = tasks_root / "stub_task"
    shutil.copytree(FIXTURE_SRC, dst)

    # Clear any artifacts the source bundle may carry.
    for d in ("runs", "logs", "comparison"):
        if (dst / d).exists():
            shutil.rmtree(dst / d)

    # Rewrite config.yaml to a single stub target with tight stop conditions.
    (dst / "config.yaml").write_text(
        """\
task_name: stub_task
mode: autonomous
target_strategy: single
target_models:
  - slug: stub-target
    provider: stub
    model_id: stub-fixed
    stub_mode: fixed
    stub_fixed_text: "{\\\"name\\\": \\\"X\\\", \\\"date\\\": \\\"2025-01-01\\\", \\\"location\\\": \\\"online\\\"}"
stop_conditions:
  target_score: 0
  max_iterations: 1
  plateau_patience: 1
  cost_cap_usd: 1.0
metric_evolution:
  enabled: false
output_type: json
eval_concurrency: 2
""",
        encoding="utf-8",
    )
    return tasks_root


@pytest.fixture
def client(tmp_path) -> TestClient:
    tasks_root = _stub_task_bundle(tmp_path)
    app = create_app(tasks_root)
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Read-only endpoints
# ---------------------------------------------------------------------------


def test_health(client: TestClient):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_list_tasks(client: TestClient):
    r = client.get("/api/tasks")
    assert r.status_code == 200
    tasks = r.json()
    assert len(tasks) == 1
    assert tasks[0]["name"] == "stub_task"
    assert tasks[0]["strategy"] == "single"
    assert tasks[0]["targets"][0]["slug"] == "stub-target"


def test_task_detail(client: TestClient):
    r = client.get("/api/tasks/stub_task")
    assert r.status_code == 200
    d = r.json()
    assert d["metric_type"] == "deterministic"
    assert "config.yaml" not in d  # we embed as text instead
    assert "task_name: stub_task" in d["config_yaml"]
    assert len(d["eval_records"]) >= 1


def test_task_detail_404(client: TestClient):
    r = client.get("/api/tasks/nope")
    assert r.status_code == 404


def test_update_metric_roundtrip(client: TestClient):
    r = client.get("/api/tasks/stub_task")
    original = r.json()["metric_yaml"]
    # Append a harmless comment — parses cleanly, schema validates.
    edited = original + "\n# edited by test\n"
    r2 = client.put("/api/tasks/stub_task/metric", json={"yaml": edited})
    assert r2.status_code == 200, r2.text

    r3 = client.get("/api/tasks/stub_task")
    assert r3.json()["metric_yaml"].endswith("# edited by test\n")


def test_update_metric_invalid_rolls_back(client: TestClient):
    r = client.get("/api/tasks/stub_task")
    original = r.json()["metric_yaml"]
    bad = "type: not_a_real_metric_type\n"
    r2 = client.put("/api/tasks/stub_task/metric", json={"yaml": bad})
    assert r2.status_code == 400
    # Rollback: disk should still have the original content.
    r3 = client.get("/api/tasks/stub_task")
    assert r3.json()["metric_yaml"] == original


# ---------------------------------------------------------------------------
# Live run lifecycle
# ---------------------------------------------------------------------------


def _wait_for_done(client: TestClient, run_id: str, timeout: float = 20.0) -> dict:
    """Poll ``/api/runs/{id}`` until status is terminal. The stub run finishes
    in well under a second; we allow 20 s to absorb CI variance."""
    import time
    start = time.time()
    while time.time() - start < timeout:
        r = client.get(f"/api/runs/{run_id}")
        assert r.status_code == 200
        info = r.json()
        if info["status"] in ("done", "stopped", "error"):
            return info
        time.sleep(0.05)
    raise AssertionError(f"run {run_id} did not finish within {timeout}s")


def test_start_run_completes_and_writes_artifacts(client: TestClient, tmp_path):
    r = client.post(
        "/api/runs",
        json={"task_name": "stub_task", "mutator": "null", "fresh": True},
    )
    assert r.status_code == 200, r.text
    start = r.json()
    run_id = start["run_id"]
    assert start["slugs"] == ["stub-target"]

    info = _wait_for_done(client, run_id)
    assert info["status"] in ("done", "stopped"), info

    # On-disk artifacts for the selected slug should exist now.
    state = client.get("/api/tasks/stub_task/state").json()
    assert "stub-target" in state
    st = state["stub-target"]
    assert st["exists"] is True
    assert st["iteration_count"] >= 1
    assert st["winner_ready"] is True

    # Iteration detail endpoint returns something reasonable.
    idx = st["latest_iteration"]
    det = client.get(f"/api/tasks/stub_task/state/stub-target/iter/{idx}").json()
    assert det["summary"]["index"] == idx
    assert isinstance(det["outputs"], list)


def test_start_run_unknown_task_404(client: TestClient):
    r = client.post("/api/runs", json={"task_name": "nope", "mutator": "null"})
    assert r.status_code == 404


def test_stop_nonexistent_run(client: TestClient):
    r = client.post("/api/runs/deadbeef/stop")
    assert r.status_code == 404


def test_signal_mode_update(client: TestClient):
    # Start a run, immediately hit it with a mode signal (the run may finish
    # before the signal is processed; we're just asserting the endpoint
    # doesn't blow up and reports ok).
    r = client.post("/api/runs", json={"task_name": "stub_task", "mutator": "null", "fresh": True})
    run_id = r.json()["run_id"]
    r2 = client.post(
        f"/api/runs/{run_id}/signal",
        json={"mode": "supervised", "feedback": ""},
    )
    assert r2.status_code == 200
    _wait_for_done(client, run_id)


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------


def test_ws_streams_hello_and_done(client: TestClient):
    r = client.post("/api/runs", json={"task_name": "stub_task", "mutator": "null", "fresh": True})
    run_id = r.json()["run_id"]
    # Deterministic: wait for the run to finish, then connect. The server
    # synthesizes hello + done for late subscribers so the client still
    # sees the full terminal snapshot.
    _wait_for_done(client, run_id)

    got_types: list[str] = []
    with client.websocket_connect(f"/api/runs/{run_id}/ws") as ws:
        while len(got_types) < 10:
            try:
                msg = ws.receive_json()
            except Exception:
                break
            got_types.append(msg.get("type"))
            if msg.get("type") == "done":
                break

    assert got_types[:2] == ["hello", "done"], got_types


# ---------------------------------------------------------------------------
# Static SPA
# ---------------------------------------------------------------------------


def test_index_served(client: TestClient):
    # The frontend bundle was built in the main pipeline; this test passes
    # iff ``lpo/ui/static/index.html`` exists on disk.
    from lpo.server.api import STATIC_DIR
    if not (STATIC_DIR / "index.html").exists():
        pytest.skip("frontend bundle not built")
    r = client.get("/")
    assert r.status_code == 200
    assert "<html" in r.text.lower()
