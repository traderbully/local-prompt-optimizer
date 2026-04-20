import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, type LiveRunInfo, type TaskSummary } from "../api";
import { Badge, Btn, Card, Scroll } from "../components/ui";

// Landing page. In ``showRuns`` mode the active/recent live runs panel is
// promoted to the top so the user can jump straight into a monitoring view.

export default function TaskList({ showRuns = false }: { showRuns?: boolean } = {}) {
  const [tasks, setTasks] = useState<TaskSummary[] | null>(null);
  const [runs, setRuns] = useState<LiveRunInfo[]>([]);
  const [error, setError] = useState<string | null>(null);

  const refresh = async () => {
    try {
      const [t, r] = await Promise.all([api.listTasks(), api.listRuns()]);
      setTasks(t);
      setRuns(r);
      setError(null);
    } catch (e: any) {
      setError(e.message ?? String(e));
    }
  };

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 3000);
    return () => clearInterval(id);
  }, []);

  return (
    <Scroll className="h-full">
      <div className="max-w-6xl mx-auto p-6 flex flex-col gap-6">
        {error && (
          <div className="text-bad-500 text-sm">
            API error: {error} — is the backend running?
          </div>
        )}

        {(showRuns || runs.length > 0) && (
          <Card
            title="Active & recent runs"
            right={<Btn variant="ghost" onClick={refresh}>Refresh</Btn>}
          >
            {runs.length === 0 ? (
              <div className="text-sm text-ink-400 italic">No runs yet.</div>
            ) : (
              <table className="w-full text-sm">
                <thead className="text-ink-400 text-xs uppercase tracking-wide">
                  <tr>
                    <th className="text-left py-1">Run</th>
                    <th className="text-left">Task</th>
                    <th className="text-left">Strategy</th>
                    <th className="text-left">Mode</th>
                    <th className="text-left">Status</th>
                    <th />
                  </tr>
                </thead>
                <tbody>
                  {runs.map((r) => (
                    <tr key={r.run_id} className="border-t border-ink-700/50">
                      <td className="py-1 mono text-xs">{r.run_id}</td>
                      <td>{r.task_name}</td>
                      <td><Badge tone="info">{r.strategy}</Badge></td>
                      <td>{r.current_mode}</td>
                      <td>
                        <Badge
                          tone={
                            r.status === "running" || r.status === "awaiting_signal"
                              ? "good"
                              : r.status === "error"
                                ? "bad"
                                : "info"
                          }
                        >
                          {r.status}
                        </Badge>
                      </td>
                      <td className="text-right">
                        <Link
                          to={`/tasks/${encodeURIComponent(r.task_name)}/runs/${r.run_id}`}
                          className="text-accent-500 hover:underline text-sm"
                        >
                          Open →
                        </Link>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </Card>
        )}

        <Card title="Task bundles">
          {tasks === null ? (
            <div className="text-sm text-ink-400">Loading…</div>
          ) : tasks.length === 0 ? (
            <div className="text-sm text-ink-400 italic">
              No task bundles found. Drop a bundle into the tasks/ directory.
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
              {tasks.map((t) => (
                <Link
                  key={t.name}
                  to={`/tasks/${encodeURIComponent(t.name)}`}
                  className="block border border-ink-700 hover:border-accent-500 rounded-md p-3 bg-ink-800/60 transition-colors"
                >
                  <div className="flex items-center justify-between">
                    <div className="font-medium text-ink-50">{t.name}</div>
                    <Badge tone="info">{t.strategy}</Badge>
                  </div>
                  <div className="text-xs text-ink-400 mt-1">
                    {t.targets.length} target{t.targets.length === 1 ? "" : "s"} ·{" "}
                    <span className="text-ink-300">{t.mode}</span> ·{" "}
                    {t.output_type}
                  </div>
                  <div className="text-xs text-ink-400 mt-2 flex flex-wrap gap-1">
                    {t.targets.map((tg) => (
                      <span
                        key={tg.slug}
                        className="border border-ink-700 rounded px-1.5 py-0.5 mono"
                      >
                        {tg.slug}
                      </span>
                    ))}
                  </div>
                  <div className="mt-2 flex items-center gap-2">
                    {t.has_runs && <Badge tone="good">has runs</Badge>}
                    {t.has_comparison && <Badge tone="info">comparison</Badge>}
                  </div>
                </Link>
              ))}
            </div>
          )}
        </Card>
      </div>
    </Scroll>
  );
}
