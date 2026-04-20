import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import {
  api,
  type RunState,
  type StartRunRequest,
  type TaskDetail as TaskDetailT,
} from "../api";
import { Badge, Btn, Card, Code, Scroll, Stat, Tabs } from "../components/ui";
import { DiffView } from "../components/DiffView";

// The task hub: task.md, eval set, metric editor (with diff), start-run
// controls, links into per-slug run viewers and the comparison.

export default function TaskDetail() {
  const { name = "" } = useParams();
  const navigate = useNavigate();
  const [detail, setDetail] = useState<TaskDetailT | null>(null);
  const [state, setState] = useState<Record<string, RunState>>({});
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<"overview" | "eval" | "metric">("overview");

  // Start-run controls
  const [mutator, setMutator] = useState<StartRunRequest["mutator"]>("auto");
  const [mode, setMode] = useState<StartRunRequest["initial_mode"]>(undefined);
  const [fresh, setFresh] = useState(true);
  const [starting, setStarting] = useState(false);

  // Metric editor draft
  const [metricDraft, setMetricDraft] = useState<string>("");
  const [metricMsg, setMetricMsg] = useState<string | null>(null);

  const refresh = async () => {
    try {
      const [d, s] = await Promise.all([api.getTask(name), api.getTaskState(name)]);
      setDetail(d);
      setState(s);
      if (metricDraft === "") setMetricDraft(d.metric_yaml);
      setError(null);
    } catch (e: any) {
      setError(e.message ?? String(e));
    }
  };

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 4000);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [name]);

  const start = async () => {
    setStarting(true);
    try {
      const resp = await api.startRun({
        task_name: name,
        mutator,
        fresh,
        initial_mode: mode,
      });
      navigate(`/tasks/${encodeURIComponent(name)}/runs/${resp.run_id}`);
    } catch (e: any) {
      setError(e.message ?? String(e));
    } finally {
      setStarting(false);
    }
  };

  const saveMetric = async () => {
    setMetricMsg(null);
    try {
      await api.putMetric(name, metricDraft);
      setMetricMsg("Saved.");
      await refresh();
    } catch (e: any) {
      setMetricMsg(e.message ?? String(e));
    }
  };

  const slugs = useMemo(() => Object.keys(state), [state]);

  if (!detail) {
    return (
      <div className="p-6 text-sm text-ink-300">
        {error ? <span className="text-bad-500">{error}</span> : "Loading…"}
      </div>
    );
  }

  const s = detail.summary;

  return (
    <Scroll className="h-full">
      <div className="max-w-6xl mx-auto p-6 flex flex-col gap-4">
        <div className="flex items-start justify-between gap-4 flex-wrap">
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-xl font-semibold">{s.name}</h1>
              <Badge tone="info">{s.strategy}</Badge>
              <Badge>{s.output_type}</Badge>
              <Badge>default mode: {s.mode}</Badge>
            </div>
            <div className="text-xs text-ink-400 mt-1 mono">{s.path}</div>
          </div>
          <div className="flex items-center gap-2">
            {s.has_comparison && (
              <Link
                to={`/tasks/${encodeURIComponent(name)}/comparison`}
                className="text-sm"
              >
                <Btn variant="ghost">Comparison →</Btn>
              </Link>
            )}
          </div>
        </div>

        <Card title="Launch a run">
          <div className="grid grid-cols-1 sm:grid-cols-4 gap-4">
            <label className="flex flex-col gap-1 text-xs text-ink-400">
              Mutator
              <select
                value={mutator}
                onChange={(e) => setMutator(e.target.value as any)}
                className="bg-ink-900 border border-ink-700 rounded px-2 py-1 text-sm text-ink-100"
              >
                <option value="auto">auto</option>
                <option value="overseer">overseer</option>
                <option value="null">null</option>
              </select>
            </label>
            <label className="flex flex-col gap-1 text-xs text-ink-400">
              Starting mode
              <select
                value={mode ?? ""}
                onChange={(e) =>
                  setMode((e.target.value || undefined) as any)
                }
                className="bg-ink-900 border border-ink-700 rounded px-2 py-1 text-sm text-ink-100"
              >
                <option value="">(task default: {s.mode})</option>
                <option value="autonomous">autonomous</option>
                <option value="supervised">supervised</option>
                <option value="manual">manual</option>
                <option value="visual">visual</option>
              </select>
            </label>
            <label className="flex items-end gap-2 text-sm text-ink-200">
              <input
                type="checkbox"
                checked={fresh}
                onChange={(e) => setFresh(e.target.checked)}
              />
              Fresh (wipe prior artifacts)
            </label>
            <div className="flex items-end">
              <Btn variant="accent" onClick={start} disabled={starting}>
                {starting ? "Starting…" : "Start run"}
              </Btn>
            </div>
          </div>
        </Card>

        <Card
          title={
            <Tabs
              tabs={[
                { key: "overview", label: "Overview" },
                { key: "eval", label: `Eval set (${detail.eval_records.length})` },
                { key: "metric", label: `Metric (${detail.metric_type})` },
              ]}
              value={tab}
              onChange={(k) => setTab(k as any)}
            />
          }
        >
          {tab === "overview" && (
            <div className="flex flex-col gap-4">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <Stat label="Eval size" value={detail.eval_records.length} />
                <Stat label="Gold records" value={detail.gold_count} />
                <Stat label="Targets" value={s.targets.length} />
                <Stat label="Metric" value={detail.metric_type} />
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <div className="text-xs uppercase text-ink-400 mb-1">task.md</div>
                  <Code>{detail.task_md || "(empty)"}</Code>
                </div>
                <div>
                  <div className="text-xs uppercase text-ink-400 mb-1">prompt_seed.txt</div>
                  <Code>{detail.seed_prompt || "(empty)"}</Code>
                </div>
              </div>

              <div>
                <div className="text-xs uppercase text-ink-400 mb-1">
                  Per-target run state
                </div>
                {slugs.length === 0 ? (
                  <div className="text-sm text-ink-400 italic">
                    No on-disk run state yet.
                  </div>
                ) : (
                  <table className="w-full text-sm">
                    <thead className="text-ink-400 text-xs uppercase tracking-wide">
                      <tr>
                        <th className="text-left py-1">Slug</th>
                        <th className="text-left">Iterations</th>
                        <th className="text-left">Best</th>
                        <th className="text-left">Winner</th>
                        <th />
                      </tr>
                    </thead>
                    <tbody>
                      {slugs.map((slug) => {
                        const st = state[slug];
                        return (
                          <tr key={slug} className="border-t border-ink-700/50">
                            <td className="py-1 mono">{slug}</td>
                            <td>{st.iteration_count}</td>
                            <td>
                              {st.best_score !== null
                                ? st.best_score.toFixed(2)
                                : "–"}
                            </td>
                            <td>
                              {st.winner_ready ? (
                                <Badge tone="good">ready</Badge>
                              ) : (
                                <span className="text-ink-500">–</span>
                              )}
                            </td>
                            <td className="text-right">
                              {st.winner_ready && (
                                <Link
                                  to={`/tasks/${encodeURIComponent(name)}/winner/${encodeURIComponent(slug)}`}
                                  className="text-accent-500 hover:underline text-sm"
                                >
                                  Winner →
                                </Link>
                              )}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                )}
              </div>
            </div>
          )}

          {tab === "eval" && (
            <div className="flex flex-col gap-2">
              {detail.eval_records.map((r, idx) => (
                <div
                  key={(r.id as string) ?? idx}
                  className="border border-ink-700 rounded p-2 bg-ink-900/40"
                >
                  <div className="flex items-center justify-between">
                    <span className="mono text-xs text-ink-400">
                      #{(r.id as string) ?? idx}
                    </span>
                    {(r.scenario as string) && (
                      <Badge tone="info">{r.scenario as string}</Badge>
                    )}
                  </div>
                  <pre className="lpo-pre mono text-xs text-ink-100 mt-1">
                    {typeof r.input === "string"
                      ? (r.input as string)
                      : JSON.stringify(r.input, null, 2)}
                  </pre>
                </div>
              ))}
            </div>
          )}

          {tab === "metric" && (
            <div className="flex flex-col gap-3">
              <div className="text-xs text-ink-400">
                Edit <span className="mono">metric.yaml</span> in place. Saving
                validates by round-tripping through the real loader; invalid
                YAML or schema rolls back automatically.
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <div>
                  <div className="text-xs uppercase text-ink-400 mb-1">Draft</div>
                  <textarea
                    value={metricDraft}
                    onChange={(e) => setMetricDraft(e.target.value)}
                    rows={20}
                    className="w-full bg-ink-900 border border-ink-700 rounded p-2 mono text-xs text-ink-100 scroll-thin"
                  />
                </div>
                <div>
                  <div className="text-xs uppercase text-ink-400 mb-1">
                    Diff vs on-disk
                  </div>
                  <DiffView before={detail.metric_yaml} after={metricDraft} />
                </div>
              </div>
              <div className="flex items-center gap-3">
                <Btn variant="accent" onClick={saveMetric}>
                  Save metric
                </Btn>
                <Btn
                  variant="ghost"
                  onClick={() => setMetricDraft(detail.metric_yaml)}
                >
                  Reset
                </Btn>
                {metricMsg && (
                  <span className="text-xs text-ink-400">{metricMsg}</span>
                )}
              </div>
            </div>
          )}
        </Card>
      </div>
    </Scroll>
  );
}
