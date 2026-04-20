// Line-based diff renderer used by the Metric editor when the user is
// comparing the current metric.yaml to an edited draft (or, in a future
// stage, an Overseer-proposed diff). No external dependencies — we compute a
// Hunt-McIlroy LCS table on lines only. Good enough for YAML-sized files.

export type DiffRow =
  | { kind: "keep"; text: string }
  | { kind: "add"; text: string }
  | { kind: "del"; text: string };

export function computeDiff(before: string, after: string): DiffRow[] {
  const a = before.split("\n");
  const b = after.split("\n");
  const m = a.length;
  const n = b.length;
  const lcs: number[][] = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));
  for (let i = m - 1; i >= 0; i--) {
    for (let j = n - 1; j >= 0; j--) {
      if (a[i] === b[j]) lcs[i][j] = lcs[i + 1][j + 1] + 1;
      else lcs[i][j] = Math.max(lcs[i + 1][j], lcs[i][j + 1]);
    }
  }
  const rows: DiffRow[] = [];
  let i = 0;
  let j = 0;
  while (i < m && j < n) {
    if (a[i] === b[j]) {
      rows.push({ kind: "keep", text: a[i] });
      i++;
      j++;
    } else if (lcs[i + 1][j] >= lcs[i][j + 1]) {
      rows.push({ kind: "del", text: a[i] });
      i++;
    } else {
      rows.push({ kind: "add", text: b[j] });
      j++;
    }
  }
  while (i < m) rows.push({ kind: "del", text: a[i++] });
  while (j < n) rows.push({ kind: "add", text: b[j++] });
  return rows;
}

export function DiffView({ before, after }: { before: string; after: string }) {
  const rows = computeDiff(before, after);
  if (rows.every((r) => r.kind === "keep")) {
    return <div className="text-xs text-ink-400 italic">No changes.</div>;
  }
  return (
    <div className="mono text-xs bg-ink-900 border border-ink-700 rounded-md overflow-auto scroll-thin">
      {rows.map((r, idx) => {
        const base = "px-3 py-0.5 whitespace-pre-wrap break-words";
        if (r.kind === "keep") return <div key={idx} className={`${base} text-ink-300`}>{" "}{r.text}</div>;
        if (r.kind === "add")
          return (
            <div key={idx} className={`${base} bg-accent-600/20 text-accent-500`}>
              +{r.text}
            </div>
          );
        return (
          <div key={idx} className={`${base} bg-bad-500/20 text-bad-500`}>
            -{r.text}
          </div>
        );
      })}
    </div>
  );
}
