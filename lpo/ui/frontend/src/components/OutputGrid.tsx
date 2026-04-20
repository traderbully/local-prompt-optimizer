import type { EvalOutputRow } from "../api";
import { Badge, Card } from "./ui";

// Renders one iteration's eval outputs in a scenario-grouped grid.
// Works for text/JSON outputs today; image/base64 rendering is stubbed so
// Visual mode can light up without a schema change.

function renderOutput(output: unknown, outputType: string): JSX.Element {
  if (outputType === "image" && typeof output === "string" && output.startsWith("data:image")) {
    return (
      <img
        src={output}
        alt="model output"
        className="max-h-48 w-full object-contain rounded border border-ink-700 bg-ink-900"
      />
    );
  }
  const text =
    typeof output === "string" ? output : JSON.stringify(output, null, 2);
  return (
    <pre className="lpo-pre mono text-xs text-ink-100 bg-ink-900 rounded p-2 max-h-48 overflow-auto scroll-thin">
      {text || <span className="text-ink-500 italic">(empty)</span>}
    </pre>
  );
}

function renderInput(input: unknown): string {
  if (typeof input === "string") return input;
  try {
    return JSON.stringify(input);
  } catch {
    return String(input);
  }
}

export default function OutputGrid({
  rows,
  outputType = "text",
}: {
  rows: EvalOutputRow[];
  outputType?: string;
}) {
  if (!rows.length) {
    return (
      <div className="text-sm text-ink-400 italic">
        No outputs recorded yet. Outputs arrive after the first iteration completes.
      </div>
    );
  }

  // Group by scenario when present; otherwise one bucket.
  const groups: Record<string, EvalOutputRow[]> = {};
  for (const r of rows) {
    const key = (r.scenario as string) || "(no scenario)";
    (groups[key] ??= []).push(r);
  }

  return (
    <div className="flex flex-col gap-4">
      {Object.entries(groups).map(([scenario, grouped]) => (
        <Card
          key={scenario}
          title={
            <span className="flex items-center gap-2">
              <span className="text-ink-200">{scenario}</span>
              <Badge tone="info">{grouped.length}</Badge>
            </span>
          }
        >
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
            {grouped.map((r) => {
              const score = typeof r.score === "number" ? r.score : null;
              const tone =
                score === null
                  ? "info"
                  : score >= 90
                    ? "good"
                    : score >= 60
                      ? "warn"
                      : "bad";
              return (
                <div
                  key={`${r.model_slug ?? "_"}-${r.id}`}
                  className="border border-ink-700 rounded-md bg-ink-800/60 p-3 flex flex-col gap-2"
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-xs text-ink-400 mono">#{r.id}</span>
                    <div className="flex items-center gap-1">
                      {r.model_slug && <Badge>{r.model_slug}</Badge>}
                      <Badge tone={tone as any}>
                        {score === null ? "–" : score.toFixed(1)}
                      </Badge>
                    </div>
                  </div>
                  <div className="text-[11px] text-ink-400 uppercase tracking-wide">Input</div>
                  <div className="text-xs text-ink-200 line-clamp-3">
                    {renderInput(r.input)}
                  </div>
                  <div className="text-[11px] text-ink-400 uppercase tracking-wide">Output</div>
                  {renderOutput(r.output, outputType)}
                  {r.rationale && (
                    <div className="text-[11px] text-ink-400 italic">
                      {r.rationale}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </Card>
      ))}
    </div>
  );
}
