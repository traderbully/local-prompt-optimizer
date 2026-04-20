import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import {
  api,
  type IterationDetail,
  type IterationSummary,
  type LiveRunInfo,
  type TaskDetail as TaskDetailT,
} from "../api";
import { Badge, Btn, Card, Code, Scroll, Stat, Tabs } from "../components/ui";
import OutputGrid from "../components/OutputGrid";
import { useRunWebSocket, type WsEvent } from "../ws";

// The live run dashboard: per-slug tab for Strategy B (one tab per target)
// and a "Unified" tab for Strategy C. Each tab shows a conversation/history
// column plus the current iteration's output grid + prompt + overseer notes.

type PerSlugState = {
  iterations: IterationSummary[]; // ascending by index
  awaiting: { mode: string; iteration: IterationSummary } | null;
};

const MODES = ["autonomous", "supervised", "manual", "visual"] as const;

export default function RunViewer() {
  const { name = "", runId = "" } = useParams();
  const [run, setRun] = useState<LiveRunInfo | null>(null);
  const [task, setTask] = useState<TaskDetailT | null>(null);
  const [perSlug, setPerSlug] = useState<Record<string, PerSlugState>>({});
  const [selectedSlug, setSelectedSlug] = useState<string | null>(null);
  const [selectedIter, setSelectedIter] = useState<number | null>(null);
  const [iterDetail, setIterDetail] = useState<IterationDetail | null>(null);
  const [iterLoadId, setIterLoadId] = useState(0);

  const [manualFeedback, setManualFeedback] = useState("");

  const { events, connected } = useRunWebSocket(runId);

  // Hydrate run + task metadata once.
  useEffect(() => {
    api.getRun(runId).then(setRun).catch(() => void 0);
    api.getTask(name).then(setTask).catch(() => void 0);
  }, [runId, name]);

  // Reduce WebSocket events into per-slug state.
  useEffect(() => {
    if (!events.length) return;
    setPerSlug((prev) => {
      const next = { ...prev };
      for (const ev of events) {
        applyEvent(ev, next);
      }
      return next;
    });
    // Last event may update the run summary (status/mode changes).
    const last = events[events.length - 1];
    if (last.type === "mode_changed" && run) {
      setRun({ ...run, current_mode: last.data.mode });
    } else if (last.type === "status" && run && typeof last.data.status === "string") {
      setRun({ ...run, status: last.data.status as any });
    } else if (last.type === "done" && run) {
      setRun({ ...run, status: (last.data.status ?? "done") as any });
    }
    // Auto-select first slug once we see something.
    if (!selectedSlug) {
      for (const ev of events) {
        if (ev.type === "iteration" && typeof ev.data.slug === "string") {
          setSelectedSlug(ev.data.slug);
          break;
        }
        if (ev.type === "hello" && ev.data.run?.slugs?.length) {
          setSelectedSlug(ev.data.run.slugs[0]);
          break;
        }
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [events]);

  // Whenever selected slug changes, if no iteration is pinned, pick the newest.
  useEffect(() => {
    if (!selectedSlug) return;
    const s = perSlug[selectedSlug];
    if (s && s.iterations.length) {
      setSelectedIter(s.iterations[s.iterations.length - 1].index);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedSlug, perSlug[selectedSlug ?? ""]?.iterations.length]);

  // Fetch the detail for the selected (slug, iter) pair.
  useEffect(() => {
    if (!selectedSlug || selectedIter === null) {
      setIterDetail(null);
      return;
    }
    const id = iterLoadId + 1;
    setIterLoadId(id);
    api
      .getIteration(name, selectedSlug, selectedIter)
      .then((d) => {
        // Guard against out-of-order responses.
        if (id === iterLoadIdRef.current) setIterDetail(d);
      })
      .catch(() => setIterDetail(null));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedSlug, selectedIter, name]);
  const iterLoadIdRef = useRef(iterLoadId);
  iterLoadIdRef.current = iterLoadId;

  // --- actions --------------------------------------------------------

  const stopRun = async () => {
    try {
      await api.stopRun(runId);
    } catch {}
  };

  const setMode = async (mode: (typeof MODES)[number]) => {
    try {
      await api.signal(runId, { mode });
      if (run) setRun({ ...run, current_mode: mode });
    } catch {}
  };

  const sendSignal = async (opts: { feedback?: string; stop?: boolean }) => {
    try {
      await api.signal(runId, {
        slug: selectedSlug,
        feedback: opts.feedback ?? "",
        stop: opts.stop ?? false,
      });
      setManualFeedback("");
    } catch {}
  };

  // --- render ---------------------------------------------------------

  const slugTabs = useMemo(() => {
    const slugs = run?.slugs ?? Object.keys(perSlug);
    return slugs.map((slug) => {
      const st = perSlug[slug];
      const iters = st?.iterations ?? [];
      const best = iters.length
        ? Math.max(...iters.map((i) => i.aggregate_score)).toFixed(1)
        : "–";
      return {
        key: slug,
        label: <span className="mono">{slug}</span>,
        badge: (
          <span className="flex items-center gap-1">
            {st?.awaiting && (
              <span className="w-2 h-2 rounded-full bg-warn-500 animate-pulse" />
            )}
            <Badge tone={iters.length ? "info" : "default"}>
              {iters.length ? `${iters.length} iter · best ${best}` : "—"}
            </Badge>
          </span>
        ),
      };
    });
  }, [run, perSlug]);

  const currentSt = selectedSlug ? perSlug[selectedSlug] : undefined;
  const awaiting = currentSt?.awaiting;
  const outputType = task?.summary.output_type ?? "text";

  return (
    <div className="h-full flex flex-col">
      {/* Top bar ---------------------------------------------------- */}
      <div className="px-4 py-2 border-b border-ink-700 bg-ink-800/70 flex flex-wrap items-center gap-3">
        <Link to={`/tasks/${encodeURIComponent(name)}`} className="text-sm text-ink-300 hover:text-ink-50">
          ← {name}
        </Link>
        <Badge tone="info">{run?.strategy ?? "…"}</Badge>
        <Badge
          tone={
            run?.status === "running" || run?.status === "awaiting_signal"
              ? "good"
              : run?.status === "error"
                ? "bad"
                : "info"
          }
        >
          {run?.status ?? "…"}
        </Badge>
        <span className="text-xs text-ink-400 mono">{runId}</span>
        <span
          className={
            "ml-2 text-xs " +
            (connected ? "text-accent-500" : "text-ink-500 italic")
          }
        >
          {connected ? "● live" : "○ disconnected"}
        </span>

        {/* Mode switcher — active always, engine only honors it for the
            next iteration gate in supervised/manual. Visual is a UI-only
            rendering hint. */}
        <div className="flex items-center gap-1 ml-auto">
          {MODES.map((m) => {
            const active = run?.current_mode === m;
            return (
              <button
                key={m}
                onClick={() => setMode(m)}
                className={
                  "px-2 py-1 rounded text-xs border " +
                  (active
                    ? "bg-accent-600 border-accent-500 text-ink-900"
                    : "bg-ink-700 border-ink-600 text-ink-200 hover:bg-ink-600")
                }
              >
                {m}
              </button>
            );
          })}
        </div>

        <Btn
          variant="bad"
          onClick={stopRun}
          disabled={run?.status === "done" || run?.status === "stopped"}
        >
          Stop
        </Btn>
      </div>

      {/* Per-model tabs --------------------------------------------- */}
      {slugTabs.length > 1 && (
        <div className="px-4 pt-2 bg-ink-800/40">
          <Tabs
            tabs={slugTabs as any}
            value={selectedSlug ?? slugTabs[0]?.key}
            onChange={(k) => setSelectedSlug(k)}
          />
        </div>
      )}

      {/* Main layout: left (history), center (outputs + prompt), right (overseer) */}
      <div className="flex-1 min-h-0 grid grid-cols-12 gap-0">
        {/* Left: iteration history list */}
        <aside className="col-span-2 border-r border-ink-700 bg-ink-800/40 overflow-hidden flex flex-col">
          <div className="px-3 py-2 text-xs uppercase tracking-wide text-ink-400 border-b border-ink-700">
            History
          </div>
          <Scroll className="flex-1">
            {currentSt?.iterations.length ? (
              <ul className="text-sm">
                {[...currentSt.iterations].reverse().map((it) => {
                  const active = it.index === selectedIter;
                  const dotTone =
                    it.decision === "accepted" || it.decision === "initial"
                      ? "bg-accent-500"
                      : it.decision === "rejected"
                        ? "bg-bad-500"
                        : "bg-ink-400";
                  return (
                    <li key={it.index}>
                      <button
                        onClick={() => setSelectedIter(it.index)}
                        className={
                          "w-full flex items-center gap-2 px-3 py-1.5 hover:bg-ink-700/50 " +
                          (active ? "bg-ink-700/70 text-ink-50" : "text-ink-200")
                        }
                      >
                        <span
                          className={`inline-block w-2 h-2 rounded-full ${dotTone}`}
                        />
                        <span className="mono text-xs">#{it.index}</span>
                        <span className="ml-auto mono text-xs">
                          {it.aggregate_score.toFixed(1)}
                        </span>
                      </button>
                    </li>
                  );
                })}
              </ul>
            ) : (
              <div className="p-3 text-xs text-ink-400 italic">
                Waiting for first iteration…
              </div>
            )}
          </Scroll>
        </aside>

        {/* Center: outputs grid + prompt */}
        <section className="col-span-7 overflow-hidden flex flex-col">
          <div className="px-4 py-2 border-b border-ink-700 bg-ink-800/30 flex items-center gap-3">
            <div className="text-sm">
              {iterDetail ? (
                <>
                  <span className="mono">#{iterDetail.summary.index}</span>{" "}
                  <Badge
                    tone={
                      iterDetail.summary.decision === "accepted" ||
                      iterDetail.summary.decision === "initial"
                        ? "good"
                        : iterDetail.summary.decision === "rejected"
                          ? "bad"
                          : "info"
                    }
                  >
                    {iterDetail.summary.decision}
                  </Badge>{" "}
                  <span className="text-ink-400 text-xs">
                    Δ {iterDetail.summary.delta.toFixed(2)} · cost $
                    {iterDetail.summary.cost_usd.toFixed(4)}
                  </span>
                </>
              ) : (
                <span className="text-ink-400 italic text-sm">
                  Select an iteration.
                </span>
              )}
            </div>
            <div className="ml-auto flex gap-4">
              {iterDetail && (
                <>
                  <Stat
                    label="Score"
                    value={iterDetail.summary.aggregate_score.toFixed(2)}
                  />
                  <Stat
                    label="Failed"
                    value={iterDetail.summary.failed_ids.length}
                  />
                </>
              )}
            </div>
          </div>
          <Scroll className="flex-1 p-4 flex flex-col gap-4">
            {iterDetail && (
              <>
                <OutputGrid rows={iterDetail.outputs} outputType={outputType} />
                <Card title="Prompt used">
                  <Code>{iterDetail.prompt}</Code>
                </Card>
              </>
            )}
          </Scroll>
        </section>

        {/* Right: overseer panel + signal controls */}
        <aside className="col-span-3 border-l border-ink-700 bg-ink-800/40 overflow-hidden flex flex-col">
          <div className="px-3 py-2 text-xs uppercase tracking-wide text-ink-400 border-b border-ink-700">
            Overseer / user signal
          </div>
          <Scroll className="flex-1 p-3 flex flex-col gap-3">
            {awaiting && (
              <Card title={<span className="text-warn-500">Awaiting signal</span>}>
                <div className="text-xs text-ink-300 mb-2">
                  Mode:{" "}
                  <Badge tone="warn">{awaiting.mode}</Badge>{" "}
                  · iter #{awaiting.iteration.index} scored{" "}
                  {awaiting.iteration.aggregate_score.toFixed(1)}
                </div>
                <div className="flex gap-2 mb-2">
                  <Btn variant="accent" onClick={() => sendSignal({ feedback: "👍 keep going" })}>
                    Approve
                  </Btn>
                  <Btn
                    variant="warn"
                    onClick={() =>
                      sendSignal({
                        feedback: "👎 regression — try a different approach",
                      })
                    }
                  >
                    Reject
                  </Btn>
                  <Btn variant="bad" onClick={() => sendSignal({ stop: true })}>
                    Stop
                  </Btn>
                </div>
                <textarea
                  value={manualFeedback}
                  onChange={(e) => setManualFeedback(e.target.value)}
                  rows={4}
                  placeholder="Free-text feedback for the Overseer…"
                  className="w-full bg-ink-900 border border-ink-700 rounded p-2 text-xs text-ink-100"
                />
                <div className="flex items-center justify-end mt-2">
                  <Btn
                    variant="accent"
                    onClick={() =>
                      sendSignal({ feedback: manualFeedback || "(no feedback)" })
                    }
                  >
                    Send feedback
                  </Btn>
                </div>
              </Card>
            )}

            {iterDetail?.overseer_analysis_md ? (
              <Card title="Overseer analysis">
                <pre className="lpo-pre mono text-xs text-ink-100">
                  {iterDetail.overseer_analysis_md}
                </pre>
              </Card>
            ) : (
              !awaiting && (
                <div className="text-xs text-ink-500 italic">
                  No overseer analysis for this iteration.
                </div>
              )
            )}

            {iterDetail && (
              <Card title="Per-scenario">
                {Object.keys(iterDetail.summary.per_scenario).length === 0 ? (
                  <div className="text-xs text-ink-500 italic">
                    No scenarios tagged on this eval set.
                  </div>
                ) : (
                  <ul className="text-xs text-ink-200 flex flex-col gap-0.5">
                    {Object.entries(iterDetail.summary.per_scenario).map(([k, v]) => (
                      <li key={k} className="flex justify-between">
                        <span>{k}</span>
                        <span className="mono">{v.toFixed(1)}</span>
                      </li>
                    ))}
                  </ul>
                )}
              </Card>
            )}

            {iterDetail?.summary.per_model && (
              <Card title="Per-model (Strategy C)">
                <ul className="text-xs text-ink-200 flex flex-col gap-0.5">
                  {Object.entries(iterDetail.summary.per_model).map(([k, v]) => (
                    <li key={k} className="flex justify-between">
                      <span className="mono">{k}</span>
                      <span className="mono">{v.toFixed(1)}</span>
                    </li>
                  ))}
                </ul>
              </Card>
            )}
          </Scroll>
        </aside>
      </div>
    </div>
  );
}

// --- reducer helpers --------------------------------------------------

function applyEvent(ev: WsEvent, state: Record<string, PerSlugState>) {
  if (ev.type === "hello") {
    const last = ev.data.last_iteration ?? {};
    for (const slug of Object.keys(last)) {
      const it = last[slug] as IterationSummary;
      ensureSlug(state, slug);
      upsertIter(state[slug], it);
    }
    return;
  }
  if (ev.type === "iteration") {
    const slug = ev.data.slug as string;
    const it = ev.data.iteration as IterationSummary;
    ensureSlug(state, slug);
    upsertIter(state[slug], it);
    state[slug].awaiting = null;
    return;
  }
  if (ev.type === "awaiting_signal") {
    const slug = ev.data.slug as string;
    ensureSlug(state, slug);
    state[slug].awaiting = {
      mode: ev.data.mode as string,
      iteration: ev.data.iteration as IterationSummary,
    };
    return;
  }
  if (ev.type === "done") {
    for (const s of Object.values(state)) s.awaiting = null;
  }
}

function ensureSlug(state: Record<string, PerSlugState>, slug: string) {
  if (!state[slug]) state[slug] = { iterations: [], awaiting: null };
}

function upsertIter(s: PerSlugState, it: IterationSummary) {
  const idx = s.iterations.findIndex((x) => x.index === it.index);
  if (idx >= 0) s.iterations[idx] = it;
  else {
    s.iterations.push(it);
    s.iterations.sort((a, b) => a.index - b.index);
  }
}
