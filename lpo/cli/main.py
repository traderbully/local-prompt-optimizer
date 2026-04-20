"""`lpo` CLI entrypoint. See `LPO_SDP.md` §4.8, §7.2."""

from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from lpo.core.comparison import write_comparison_report
from lpo.core.cost import CostTracker
from lpo.core.engine import RatchetEngine
from lpo.core.multi_engine import run_multi, validate_runtime
from lpo.core.target_factory import build_target_context
from lpo.core.task import TaskBundle

app = typer.Typer(add_completion=False, help="Local Prompt Optimizer.")
task_app = typer.Typer(help="Task bundle commands.")
app.add_typer(task_app, name="task")

console = Console()


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-5s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )
    # Quiet noisy third parties unless --verbose.
    if not verbose:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("anthropic").setLevel(logging.WARNING)


@app.callback()
def _root(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    load_dotenv(override=True)
    _configure_logging(verbose)


@app.command()
def run(
    task_dir: Path = typer.Argument(..., help="Path to task bundle directory"),
    model_slug: str | None = typer.Option(
        None, "--model", "-m", help="Target model slug (single-target runs only; ignored for B/C).",
    ),
    mutator_name: str = typer.Option(
        "auto",
        "--mutator",
        help="Prompt mutator: auto | overseer | null. 'auto' picks overseer when autonomous mode + ANTHROPIC_API_KEY are present.",
    ),
    fresh: bool = typer.Option(
        False, "--fresh", help="Delete existing runs/, logs/ and comparison/ for this task before starting.",
    ),
) -> None:
    """Execute the ratchet loop on a task bundle.

    Dispatches by ``target_strategy``:
      * ``single`` → single-target RatchetEngine.
      * ``parallel_independent`` → one full optimization per target (Strategy B).
      * ``unified_portable`` → one shared prompt across all targets (Strategy C).
    """
    task = TaskBundle.load(task_dir)
    validate_runtime(task.config)
    strategy = task.config.target_strategy

    if strategy == "single":
        _run_single(task, model_slug, mutator_name, fresh)
        return

    if fresh:
        _wipe_all_run_artifacts(task.root)

    console.print(
        f"[bold]Strategy:[/] {strategy}   "
        f"[bold]Targets:[/] {', '.join(m.slug for m in task.config.target_models)}"
    )
    if strategy == "parallel_independent":
        console.print(f"  execution: {task.config.parallel_execution}")
    if strategy == "unified_portable":
        console.print(f"  aggregation: {task.config.unified_aggregation}")

    cost = CostTracker()

    async def _go() -> None:
        result = await run_multi(task, cost=cost, mutator_mode=mutator_name)

        summary_path, report_path = write_comparison_report(
            task.root, task.config.task_name, result
        )

        console.rule("Run complete")
        _print_multi_summary(result, report_path)

    asyncio.run(_go())


def _run_single(
    task: TaskBundle, model_slug: str | None, mutator_name: str, fresh: bool
) -> None:
    """Classic single-target path. Preserved verbatim for Strategy A."""
    target = _pick_target(task, model_slug)

    if fresh:
        _wipe_run_artifacts(task.root, target.slug)

    cost = CostTracker()
    ctx = build_target_context(task, target, cost, mutator_mode=mutator_name)

    async def _go() -> None:
        try:
            engine = RatchetEngine(
                task=task,
                target_cfg=target,
                client=ctx.client,
                scorer=ctx.scorer,
                mutator=ctx.mutator,
                cost_tracker=cost,
            )
            result = await engine.run()
        finally:
            await ctx.aclose()
        console.rule("Run complete")
        console.print(f"[bold]Best score:[/] {result.best_score:.2f}")
        console.print(f"[bold]Iterations:[/] {len(result.iterations)}")
        console.print(f"[bold]Stop reason:[/] {result.stop_reason.value}")
        console.print(f"[bold]Total cost:[/] ${result.total_cost_usd:.4f}")
        console.print(f"Winner: {engine.paths.winner_root / 'prompt.txt'}")

    asyncio.run(_go())


def _print_multi_summary(result, report_path: Path) -> None:
    table = Table("Slug", "Best", "Iters", "Stop", "Cost")
    for r in sorted(result.per_model, key=lambda x: x.best_score, reverse=True):
        table.add_row(
            r.slug,
            f"{r.best_score:.2f}",
            str(r.iterations),
            r.stop_reason.value,
            f"${r.cost_usd:.4f}",
        )
    console.print(table)

    if result.strategy == "unified_portable" and result.shared_best_score is not None:
        console.print(
            f"[bold]Combined:[/] {result.shared_best_score:.2f}  "
            f"([dim]{result.shared_stop_reason.value if result.shared_stop_reason else ''}[/])"
        )
    console.print(f"[bold]Total cost:[/] ${result.total_cost_usd:.4f}")
    console.print(f"Report: {report_path}")


def _wipe_run_artifacts(task_root: Path, slug: str) -> None:
    for p in (task_root / "runs" / slug, task_root / "logs"):
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)
    console.print(f"[dim]--fresh: cleared prior artifacts for {slug}.[/]")


def _wipe_all_run_artifacts(task_root: Path) -> None:
    for p in (task_root / "runs", task_root / "logs", task_root / "comparison"):
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)
    console.print("[dim]--fresh: cleared runs/, logs/, comparison/.[/]")


def _pick_target(task: TaskBundle, slug: str | None):
    if slug is None:
        return task.config.target_models[0]
    for m in task.config.target_models:
        if m.slug == slug:
            return m
    raise typer.BadParameter(f"No target model with slug {slug!r} in {task.root / 'config.yaml'}")


@task_app.command("list")
def task_list(root: Path = typer.Argument(Path("tasks"))) -> None:
    """List task bundles under a directory."""
    if not root.exists():
        console.print(f"[red]No such directory:[/] {root}")
        raise typer.Exit(1)
    table = Table("Task", "Strategy", "Targets", "Mode")
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        cfg = child / "config.yaml"
        if not cfg.exists():
            continue
        try:
            task = TaskBundle.load(child)
        except Exception as e:  # noqa: BLE001
            table.add_row(child.name, "[red]error[/]", str(e)[:40], "-")
            continue
        table.add_row(
            task.config.task_name,
            task.config.target_strategy,
            ", ".join(m.slug for m in task.config.target_models),
            task.config.mode,
        )
    console.print(table)


@app.command()
def ui(
    tasks_root: Path = typer.Option(
        Path("tasks"), "--tasks-root", help="Directory holding task bundles.",
    ),
    host: str = typer.Option("127.0.0.1", "--host", help="Bind host. Local-only by default."),
    port: int = typer.Option(8787, "--port", help="Bind port."),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code change (dev)."),
    open_browser: bool = typer.Option(
        True, "--open/--no-open", help="Open the UI in the default browser.",
    ),
) -> None:
    """Launch the FastAPI + React web UI at http://HOST:PORT.

    The React bundle is expected at ``lpo/ui/static/`` (built via
    ``npm run build`` in ``lpo/ui/frontend/``). When it's missing the
    backend still runs and ``/`` returns a helpful placeholder.
    """
    import uvicorn
    import webbrowser

    from lpo.server.api import STATIC_DIR, create_app

    resolved = tasks_root.resolve()
    if not resolved.exists():
        console.print(f"[yellow]Creating tasks root:[/] {resolved}")
        resolved.mkdir(parents=True, exist_ok=True)

    bundle_ready = STATIC_DIR.exists() and (STATIC_DIR / "index.html").exists()
    console.print(f"[bold]LPO UI[/] → http://{host}:{port}   tasks_root={resolved}")
    if not bundle_ready:
        console.print(
            "[yellow]Frontend bundle not found.[/] "
            "Build it with [bold]npm install && npm run build[/] "
            "inside [cyan]lpo/ui/frontend/[/]."
        )

    if reload:
        # uvicorn's reload requires an import string; use the factory.
        import os
        os.environ["LPO_TASKS_ROOT"] = str(resolved)
        uvicorn.run(
            "lpo.server.api:_factory_from_env",
            host=host, port=port, reload=True,
        )
        return

    app_instance = create_app(resolved)
    if open_browser:
        try:
            webbrowser.open(f"http://{host}:{port}")
        except Exception:  # pragma: no cover
            pass
    uvicorn.run(app_instance, host=host, port=port, log_level="info")


@app.command()
def mcp(
    tasks_root: Path = typer.Option(
        Path("tasks"), "--tasks-root", help="Directory holding task bundles.",
    ),
) -> None:
    """Run LPO as an MCP stdio server.

    Register this command in your MCP client (e.g. Windsurf) so the agent can
    invoke LPO tools (create_task, run_optimization, get_winner, …) on any
    project's task bundles. Operates headlessly in Autonomous mode; Manual
    and Visual modes are UI-only and rejected over MCP.

    See README for the exact JSON registration snippet.
    """
    import sys
    from lpo.server.mcp_server import run_stdio

    resolved = tasks_root.resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    # IMPORTANT: MCP stdio owns stdin/stdout for JSON-RPC. Log startup to
    # stderr only — any stdout write here corrupts the handshake.
    print(f"LPO MCP server starting. tasks_root={resolved}", file=sys.stderr, flush=True)
    asyncio.run(run_stdio(resolved))


@task_app.command("show")
def task_show(task_dir: Path) -> None:
    """Summarize a task bundle and its latest run, if any."""
    task = TaskBundle.load(task_dir)
    console.print(f"[bold]{task.config.task_name}[/]  ({task.config.target_strategy})")
    console.print(f"  eval examples: {len(task.eval_records)}")
    console.print(f"  gold records:  {len(task.gold_standard)}")
    for m in task.config.target_models:
        run_dir = task.root / "runs" / m.slug
        history = run_dir / "history"
        n_iters = len(list(history.glob("iter_*"))) if history.exists() else 0
        best = run_dir / "prompt.txt.best"
        status = "never run" if n_iters == 0 else f"{n_iters} iter(s)"
        best_marker = "  (best prompt on disk)" if best.exists() else ""
        console.print(f"  - {m.slug}: {status}{best_marker}")


if __name__ == "__main__":
    app()
