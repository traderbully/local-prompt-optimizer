import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api } from "../api";
import { Badge, Btn, Card, Code, Scroll } from "../components/ui";

export default function Winner() {
  const { name = "", slug = "" } = useParams();
  const [data, setData] = useState<{
    slug: string;
    present: boolean;
    prompt: string | null;
    report_md: string | null;
  } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    api.getWinner(name, slug).then(setData).catch((e) => setError(String(e)));
  }, [name, slug]);

  const copy = async () => {
    if (!data?.prompt) return;
    try {
      await navigator.clipboard.writeText(data.prompt);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {}
  };

  if (error) return <div className="p-6 text-bad-500 text-sm">{error}</div>;
  if (!data) return <div className="p-6 text-sm text-ink-300">Loading…</div>;
  if (!data.present) {
    return (
      <div className="p-6 text-sm text-ink-300">
        No winner on disk yet for <span className="mono">{slug}</span>. Finish a
        run first.{" "}
        <Link to={`/tasks/${encodeURIComponent(name)}`} className="text-accent-500 hover:underline">
          Back to task
        </Link>
      </div>
    );
  }

  return (
    <Scroll className="h-full">
      <div className="max-w-4xl mx-auto p-6 flex flex-col gap-4">
        <div className="flex items-center gap-3 flex-wrap">
          <h1 className="text-xl font-semibold">Winner</h1>
          <Badge tone="good">{slug}</Badge>
          <Link
            to={`/tasks/${encodeURIComponent(name)}`}
            className="ml-auto text-sm text-ink-300 hover:text-ink-50"
          >
            ← {name}
          </Link>
        </div>

        <Card
          title="Winning prompt"
          right={
            <Btn variant="accent" onClick={copy} disabled={!data.prompt}>
              {copied ? "Copied!" : "Copy to clipboard"}
            </Btn>
          }
        >
          <Code>{data.prompt ?? "(empty)"}</Code>
        </Card>

        {data.report_md && (
          <Card title="Winner report">
            <pre className="lpo-pre mono text-xs text-ink-100">
              {data.report_md}
            </pre>
          </Card>
        )}
      </div>
    </Scroll>
  );
}
