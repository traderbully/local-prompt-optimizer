"""FastAPI app exposing the REST + WebSocket surface for the LPO web UI.

Route map (all under ``/api``):

  GET  /tasks                                          → list TaskSummary
  GET  /tasks/{task_name}                              → TaskDetail
  GET  /tasks/{task_name}/state                        → dict[slug -> RunState]
  GET  /tasks/{task_name}/state/{slug}/iter/{index}    → IterationDetail
  GET  /tasks/{task_name}/comparison                   → ComparisonView
  GET  /tasks/{task_name}/winner/{slug}                → winner prompt bundle
  PUT  /tasks/{task_name}/metric                       → replace metric.yaml
  PUT  /tasks/{task_name}/prompt_seed                  → replace prompt_seed.txt

  GET  /runs                                           → list LiveRunInfo
  POST /runs                                           → StartRunResponse
  POST /runs/{run_id}/stop
  POST /runs/{run_id}/signal
  GET  /runs/{run_id}                                  → LiveRunInfo
  WS   /runs/{run_id}/ws                               → event stream

At startup the static SPA assets under ``lpo/ui/static/`` (if present, built
by Vite) are mounted at the root, so ``lpo ui`` serves a single-origin app.
When the bundle is absent (dev), a tiny JSON placeholder is served instead.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from lpo.server.runs import RunManager
from lpo.server.schemas import (
    ComparisonView,
    IterationDetail,
    LiveRunInfo,
    RunState,
    SignalRequest,
    StartRunRequest,
    StartRunResponse,
    TaskDetail,
    TaskSummary,
)
from lpo.server.tasks import (
    read_all_tasks,
    read_comparison,
    read_iteration_detail,
    read_run_state,
    read_task_detail,
    read_winner,
    resolve_task_path,
)

log = logging.getLogger("lpo.server.api")

STATIC_DIR = Path(__file__).resolve().parent.parent / "ui" / "static"


def _factory_from_env() -> FastAPI:  # pragma: no cover — used by uvicorn --reload
    """Factory used when ``lpo ui --reload`` spawns a child process. The
    child doesn't have access to the original ``tasks_root`` argument, so
    we ferry it through an env var."""
    import os
    tasks_root = Path(os.environ.get("LPO_TASKS_ROOT", "tasks")).resolve()
    return create_app(tasks_root)


def create_app(tasks_root: Path, *, enable_cors: bool = True) -> FastAPI:
    """Build the FastAPI application.

    ``tasks_root`` is the directory scanned for task bundles (default
    ``./tasks``). ``enable_cors`` is helpful for the Vite dev server (port
    5173 by default); production builds served from the same origin don't
    need CORS.
    """
    app = FastAPI(title="Local Prompt Optimizer", version="0.5.0")
    runs = RunManager(tasks_root=tasks_root)

    if enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # local-only host; see SDP §6.3
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.state.tasks_root = tasks_root
    app.state.runs = runs

    # ------------------------------------------------------------------
    # Task readers
    # ------------------------------------------------------------------

    @app.get("/api/tasks", response_model=list[TaskSummary])
    async def _list_tasks() -> list[TaskSummary]:
        return read_all_tasks(tasks_root)

    @app.get("/api/tasks/{task_name}", response_model=TaskDetail)
    async def _task_detail(task_name: str) -> TaskDetail:
        path = _resolve_or_404(task_name)
        return read_task_detail(path)

    @app.get("/api/tasks/{task_name}/state")
    async def _task_state(task_name: str) -> dict[str, RunState]:
        path = _resolve_or_404(task_name)
        # Probe every target-model slug (plus _unified for Strategy C).
        detail = read_task_detail(path)
        slugs = [t.slug for t in detail.summary.targets]
        if detail.summary.strategy == "unified_portable":
            slugs = ["_unified"]
        return {slug: read_run_state(path, slug) for slug in slugs}

    @app.get(
        "/api/tasks/{task_name}/state/{slug}/iter/{index}",
        response_model=IterationDetail,
    )
    async def _iteration_detail(task_name: str, slug: str, index: int) -> IterationDetail:
        path = _resolve_or_404(task_name)
        try:
            return read_iteration_detail(path, slug, index)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="No such iteration")

    @app.get("/api/tasks/{task_name}/comparison", response_model=ComparisonView)
    async def _comparison(task_name: str) -> ComparisonView:
        path = _resolve_or_404(task_name)
        return read_comparison(path)

    @app.get("/api/tasks/{task_name}/winner/{slug}")
    async def _winner(task_name: str, slug: str) -> dict:
        path = _resolve_or_404(task_name)
        return read_winner(path, slug)

    # ------------------------------------------------------------------
    # Task writers (metric editor + prompt seed edits)
    # ------------------------------------------------------------------

    @app.put("/api/tasks/{task_name}/metric")
    async def _update_metric(task_name: str, payload: dict) -> dict:
        path = _resolve_or_404(task_name)
        new_yaml = payload.get("yaml")
        if not isinstance(new_yaml, str) or not new_yaml.strip():
            raise HTTPException(status_code=400, detail="Request body must include non-empty 'yaml'")
        # Validate by round-tripping through the loader.
        from lpo.config.schema import load_metric_config  # local import
        import yaml
        try:
            yaml.safe_load(new_yaml)
        except yaml.YAMLError as e:
            raise HTTPException(status_code=400, detail=f"Invalid YAML: {e}")
        # Write to a temp file first so we can validate with the real loader.
        metric_path = path / "metric.yaml"
        backup = metric_path.read_text(encoding="utf-8") if metric_path.exists() else ""
        metric_path.write_text(new_yaml, encoding="utf-8")
        try:
            load_metric_config(metric_path)
        except Exception as e:  # noqa: BLE001
            metric_path.write_text(backup, encoding="utf-8")  # rollback
            raise HTTPException(status_code=400, detail=f"Metric config invalid: {e}")
        return {"status": "ok"}

    @app.put("/api/tasks/{task_name}/prompt_seed")
    async def _update_prompt_seed(task_name: str, payload: dict) -> dict:
        path = _resolve_or_404(task_name)
        new_text = payload.get("text", "")
        if not isinstance(new_text, str):
            raise HTTPException(status_code=400, detail="Request body must include string 'text'")
        (path / "prompt_seed.txt").write_text(new_text, encoding="utf-8")
        return {"status": "ok"}

    # ------------------------------------------------------------------
    # Run control
    # ------------------------------------------------------------------

    @app.get("/api/runs", response_model=list[LiveRunInfo])
    async def _list_runs() -> list[LiveRunInfo]:
        return runs.list_runs()

    @app.post("/api/runs", response_model=StartRunResponse)
    async def _start_run(req: StartRunRequest) -> StartRunResponse:
        try:
            run = await runs.start(req)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return StartRunResponse(
            run_id=run.run_id,
            task_name=run.task_name,
            strategy=run.strategy,
            slugs=list(run.slugs),
        )

    @app.get("/api/runs/{run_id}", response_model=LiveRunInfo)
    async def _run_info(run_id: str) -> LiveRunInfo:
        try:
            return runs.get_run(run_id).info()
        except KeyError:
            raise HTTPException(status_code=404, detail="No such run")

    @app.post("/api/runs/{run_id}/stop")
    async def _stop_run(run_id: str) -> dict:
        try:
            await runs.stop(run_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="No such run")
        return {"status": "stopping"}

    @app.post("/api/runs/{run_id}/signal")
    async def _signal_run(run_id: str, req: SignalRequest) -> dict:
        try:
            await runs.submit_signal(
                run_id,
                slug=req.slug,
                mode=req.mode,
                feedback=req.feedback,
                stop=req.stop,
            )
        except KeyError:
            raise HTTPException(status_code=404, detail="No such run")
        return {"status": "ok"}

    # ------------------------------------------------------------------
    # WebSocket
    # ------------------------------------------------------------------

    @app.websocket("/api/runs/{run_id}/ws")
    async def _run_ws(ws: WebSocket, run_id: str) -> None:
        await ws.accept()
        try:
            run = runs.get_run(run_id)
        except KeyError:
            await ws.send_json({"type": "error", "data": {"error": "no such run"}})
            await ws.close()
            return

        # Hello snapshot for late-joiners.
        await ws.send_json(runs.hello_snapshot(run).model_dump())

        # If the run already finished before the client connected, the
        # terminal "done" event was dispatched to an empty subscriber set.
        # Re-emit a synthetic one here so the UI can render the final state.
        if run.status in ("done", "stopped", "error"):
            await ws.send_json({"type": "done", "data": {"status": run.status}})
            await ws.close()
            return

        q = runs.subscribe(run_id)
        try:
            while True:
                try:
                    event = await asyncio.wait_for(q.get(), timeout=30.0)
                    await ws.send_json(event)
                    if event.get("type") == "done":
                        # Give the UI a moment to process, then close.
                        await asyncio.sleep(0.1)
                        break
                except asyncio.TimeoutError:
                    # Keepalive ping — some proxies drop idle sockets.
                    await ws.send_json({"type": "status", "data": {"heartbeat": True}})
        except WebSocketDisconnect:
            pass
        finally:
            runs.unsubscribe(run_id, q)
            try:
                await ws.close()
            except Exception:  # pragma: no cover
                pass

    # ------------------------------------------------------------------
    # Health + static SPA
    # ------------------------------------------------------------------

    @app.get("/api/health")
    async def _health() -> dict:
        return {"status": "ok", "tasks_root": str(tasks_root)}

    if STATIC_DIR.exists() and (STATIC_DIR / "index.html").exists():
        # Mount SPA assets. The trailing ``index.html`` catch-all enables
        # client-side routing (/tasks/:name paths).
        app.mount(
            "/assets",
            StaticFiles(directory=STATIC_DIR / "assets"),
            name="spa-assets",
        )

        @app.get("/")
        async def _index() -> FileResponse:
            return FileResponse(STATIC_DIR / "index.html")

        @app.get("/{full_path:path}")
        async def _spa_catchall(full_path: str) -> FileResponse:
            # Let /api/* 404 naturally by not reaching here (FastAPI matches
            # /api/* routes first). Everything else falls back to index.html.
            candidate = STATIC_DIR / full_path
            if candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(STATIC_DIR / "index.html")
    else:
        @app.get("/")
        async def _dev_placeholder() -> JSONResponse:
            return JSONResponse(
                {
                    "message": "LPO backend is running. Frontend bundle not built yet.",
                    "next": "Run `npm install && npm run build` in lpo/ui/frontend/.",
                    "tasks_root": str(tasks_root),
                }
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_or_404(task_name: str) -> Path:
        try:
            return resolve_task_path(tasks_root, task_name)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"No such task: {task_name}")

    return app
