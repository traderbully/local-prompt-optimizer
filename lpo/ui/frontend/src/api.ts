// Thin, typed client for the LPO FastAPI backend.
//
// The dev server (npm run dev) proxies /api to the Python backend at
// 127.0.0.1:8787 (see vite.config.ts); the production build served by the
// Python app is single-origin so the same relative URLs just work.

export type TargetSummary = { slug: string; provider: string; model_id: string };

export type TaskSummary = {
  name: string;
  path: string;
  strategy: "single" | "parallel_independent" | "unified_portable" | string;
  mode: string;
  output_type: string;
  targets: TargetSummary[];
  has_runs: boolean;
  has_comparison: boolean;
};

export type TaskDetail = {
  summary: TaskSummary;
  task_md: string;
  seed_prompt: string;
  config_yaml: string;
  metric_yaml: string;
  eval_records: Array<Record<string, unknown>>;
  gold_count: number;
  metric_type: string;
};

export type IterationSummary = {
  index: number;
  aggregate_score: number;
  decision: string;
  delta: number;
  cost_usd: number;
  timestamp: string;
  failed_ids: string[];
  per_scenario: Record<string, number>;
  per_model?: Record<string, number> | null;
};

export type EvalOutputRow = {
  id: string;
  input: unknown;
  output: unknown;
  score?: number | null;
  rationale?: string | null;
  scenario?: string | null;
  model_slug?: string | null;
  [k: string]: unknown;
};

export type IterationDetail = {
  summary: IterationSummary;
  prompt: string;
  outputs: EvalOutputRow[];
  overseer_analysis_md: string | null;
  scores_full: Record<string, unknown>;
  decision_full: Record<string, unknown>;
};

export type RunState = {
  slug: string;
  exists: boolean;
  best_score: number | null;
  best_prompt: string | null;
  iteration_count: number;
  latest_iteration: number | null;
  iterations: IterationSummary[];
  winner_ready: boolean;
};

export type ComparisonView = {
  present: boolean;
  summary: Record<string, any> | null;
  report_md: string | null;
};

export type LiveRunInfo = {
  run_id: string;
  task_name: string;
  strategy: string;
  status:
    | "starting"
    | "running"
    | "awaiting_signal"
    | "done"
    | "error"
    | "stopped";
  started_at: string;
  finished_at: string | null;
  error: string | null;
  current_mode: string;
  slugs: string[];
  latest_iterations: Record<string, number>;
};

export type StartRunRequest = {
  task_name: string;
  mutator?: "auto" | "overseer" | "null";
  fresh?: boolean;
  initial_mode?: "autonomous" | "supervised" | "manual" | "visual";
};

export type SignalRequest = {
  slug?: string | null;
  mode?: "autonomous" | "supervised" | "manual" | "visual";
  feedback?: string;
  stop?: boolean;
};

async function req<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {}
    throw new Error(`${res.status} ${detail}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  listTasks: () => req<TaskSummary[]>(`/api/tasks`),
  getTask: (name: string) => req<TaskDetail>(`/api/tasks/${encodeURIComponent(name)}`),
  getTaskState: (name: string) =>
    req<Record<string, RunState>>(`/api/tasks/${encodeURIComponent(name)}/state`),
  getIteration: (name: string, slug: string, index: number) =>
    req<IterationDetail>(
      `/api/tasks/${encodeURIComponent(name)}/state/${encodeURIComponent(slug)}/iter/${index}`,
    ),
  getComparison: (name: string) =>
    req<ComparisonView>(`/api/tasks/${encodeURIComponent(name)}/comparison`),
  getWinner: (name: string, slug: string) =>
    req<{ slug: string; present: boolean; prompt: string | null; report_md: string | null }>(
      `/api/tasks/${encodeURIComponent(name)}/winner/${encodeURIComponent(slug)}`,
    ),
  putMetric: (name: string, yaml: string) =>
    req<{ status: string }>(`/api/tasks/${encodeURIComponent(name)}/metric`, {
      method: "PUT",
      body: JSON.stringify({ yaml }),
    }),
  putPromptSeed: (name: string, text: string) =>
    req<{ status: string }>(`/api/tasks/${encodeURIComponent(name)}/prompt_seed`, {
      method: "PUT",
      body: JSON.stringify({ text }),
    }),

  listRuns: () => req<LiveRunInfo[]>(`/api/runs`),
  getRun: (id: string) => req<LiveRunInfo>(`/api/runs/${id}`),
  startRun: (body: StartRunRequest) =>
    req<{ run_id: string; task_name: string; strategy: string; slugs: string[] }>(`/api/runs`, {
      method: "POST",
      body: JSON.stringify(body),
    }),
  stopRun: (id: string) =>
    req<{ status: string }>(`/api/runs/${id}/stop`, { method: "POST" }),
  signal: (id: string, body: SignalRequest) =>
    req<{ status: string }>(`/api/runs/${id}/signal`, {
      method: "POST",
      body: JSON.stringify(body),
    }),
};

export function wsUrl(runId: string): string {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}/api/runs/${runId}/ws`;
}
