import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api, type ComparisonView } from "../api";
import { Badge, Card, Scroll } from "../components/ui";

export default function Comparison() {
  const { name = "" } = useParams();
  const [view, setView] = useState<ComparisonView | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.getComparison(name).then(setView).catch((e) => setError(String(e)));
  }, [name]);

  if (error) {
    return <div className="p-6 text-bad-500 text-sm">{error}</div>;
  }
  if (!view) {
    return <div className="p-6 text-sm text-ink-300">Loading…</div>;
  }
  if (!view.present) {
    return (
      <div className="p-6 text-sm text-ink-300">
        No comparison report yet. Run a multi-target strategy against this task
        to populate <span className="mono">comparison/summary.json</span>.{" "}
        <Link to={`/tasks/${encodeURIComponent(name)}`} className="text-accent-500 hover:underline">
          Back to task
        </Link>
      </div>
    );
  }

  const summary = view.summary ?? {};
  const perModel: any[] = (summary.per_model as any[]) ?? [];
  const winner = summary.winner_recommendation as any;

  return (
    <Scroll className="h-full">
      <div className="max-w-6xl mx-auto p-6 flex flex-col gap-4">
        <div className="flex items-start justify-between flex-wrap gap-3">
          <div>
            <h1 className="text-xl font-semibold">Comparison — {name}</h1>
            <div className="text-xs text-ink-400 mono">
              strategy: {String(summary.strategy)} · total cost $
              {Number(summary.total_cost_usd ?? 0).toFixed(4)}
            </div>
          </div>
          <Link
            to={`/tasks/${encodeURIComponent(name)}`}
            className="text-sm text-ink-300 hover:text-ink-50"
          >
            ← task
          </Link>
        </div>

        {winner && (
          <Card title="Recommended winner">
            <div className="flex items-center gap-3">
              <Badge tone="good">{String(winner.slug)}</Badge>
              <span className="text-sm text-ink-200">
                best_score = {Number(winner.best_score ?? 0).toFixed(2)}
              </span>
              <span className="text-xs text-ink-400 italic">
                {String(winner.reason ?? "")}
              </span>
              <Link
                to={`/tasks/${encodeURIComponent(name)}/winner/${encodeURIComponent(
                  winner.slug,
                )}`}
                className="ml-auto text-sm text-accent-500 hover:underline"
              >
                Open winner →
              </Link>
            </div>
          </Card>
        )}

        <Card title="Per-model results">
          <table className="w-full text-sm">
            <thead className="text-ink-400 text-xs uppercase tracking-wide">
              <tr>
                <th className="text-left py-1">Slug</th>
                <th className="text-left">Model</th>
                <th className="text-left">Best</th>
                <th className="text-left">Iters</th>
                <th className="text-left">Stop</th>
                <th className="text-left">Cost</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {perModel.map((r: any) => (
                <tr key={String(r.slug)} className="border-t border-ink-700/50">
                  <td className="py-1 mono">{String(r.slug)}</td>
                  <td className="mono text-xs">{String(r.model_id ?? "")}</td>
                  <td className="mono">{Number(r.best_score ?? 0).toFixed(2)}</td>
                  <td>{Number(r.iterations ?? 0)}</td>
                  <td className="text-xs text-ink-300">{String(r.stop_reason ?? "")}</td>
                  <td className="mono text-xs">
                    ${Number(r.cost_usd ?? 0).toFixed(4)}
                  </td>
                  <td className="text-right">
                    <Link
                      to={`/tasks/${encodeURIComponent(name)}/winner/${encodeURIComponent(
                        r.slug,
                      )}`}
                      className="text-accent-500 hover:underline text-sm"
                    >
                      Winner
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>

        {view.report_md && (
          <Card title="Report">
            <pre className="lpo-pre mono text-xs text-ink-100">
              {view.report_md}
            </pre>
          </Card>
        )}
      </div>
    </Scroll>
  );
}
