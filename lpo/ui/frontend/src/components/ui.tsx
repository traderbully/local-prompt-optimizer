// Small grab-bag of low-level UI primitives. Keeping them in one file
// sidesteps a proper shadcn/ui setup for this stage while still giving every
// page a consistent visual vocabulary.
import { PropsWithChildren, ReactNode } from "react";

export function Card({
  title,
  right,
  children,
  className = "",
}: PropsWithChildren<{ title?: ReactNode; right?: ReactNode; className?: string }>) {
  return (
    <section
      className={`bg-ink-800 border border-ink-700 rounded-lg overflow-hidden ${className}`}
    >
      {title !== undefined && (
        <header className="flex items-center justify-between px-4 py-2 border-b border-ink-700 bg-ink-700/40">
          <div className="text-sm font-medium text-ink-100">{title}</div>
          {right}
        </header>
      )}
      <div className="p-4">{children}</div>
    </section>
  );
}

export function Btn({
  onClick,
  disabled,
  variant = "default",
  children,
  type = "button",
  className = "",
}: {
  onClick?: () => void;
  disabled?: boolean;
  variant?: "default" | "accent" | "warn" | "bad" | "ghost";
  children: ReactNode;
  type?: "button" | "submit";
  className?: string;
}) {
  const base =
    "px-3 py-1.5 rounded-md text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed";
  const byVariant: Record<string, string> = {
    default: "bg-ink-600 hover:bg-ink-500 text-ink-50",
    accent: "bg-accent-600 hover:bg-accent-500 text-ink-900",
    warn: "bg-warn-500 hover:bg-amber-400 text-ink-900",
    bad: "bg-bad-500 hover:bg-red-400 text-ink-50",
    ghost: "bg-transparent hover:bg-ink-700 text-ink-200",
  };
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={`${base} ${byVariant[variant]} ${className}`}
    >
      {children}
    </button>
  );
}

export function Badge({
  children,
  tone = "default",
}: {
  children: ReactNode;
  tone?: "default" | "good" | "warn" | "bad" | "info";
}) {
  const tones: Record<string, string> = {
    default: "bg-ink-600 text-ink-50",
    good: "bg-accent-600 text-ink-900",
    warn: "bg-warn-500 text-ink-900",
    bad: "bg-bad-500 text-ink-50",
    info: "bg-ink-700 text-ink-200 border border-ink-500",
  };
  return (
    <span
      className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${tones[tone]}`}
    >
      {children}
    </span>
  );
}

export function Stat({
  label,
  value,
  sub,
}: {
  label: string;
  value: ReactNode;
  sub?: ReactNode;
}) {
  return (
    <div className="flex flex-col gap-0.5">
      <div className="text-xs uppercase tracking-wide text-ink-400">{label}</div>
      <div className="text-lg font-semibold text-ink-50">{value}</div>
      {sub && <div className="text-xs text-ink-400">{sub}</div>}
    </div>
  );
}

export function Code({ children }: { children: ReactNode }) {
  return (
    <pre className="lpo-pre mono text-xs bg-ink-900 border border-ink-700 rounded-md p-3 overflow-auto scroll-thin text-ink-100">
      {children}
    </pre>
  );
}

export function Scroll({ children, className = "" }: PropsWithChildren<{ className?: string }>) {
  return <div className={`overflow-auto scroll-thin ${className}`}>{children}</div>;
}

export function Tabs({
  tabs,
  value,
  onChange,
}: {
  tabs: { key: string; label: ReactNode; badge?: ReactNode }[];
  value: string;
  onChange: (key: string) => void;
}) {
  return (
    <div role="tablist" className="flex border-b border-ink-700">
      {tabs.map((t) => {
        const active = t.key === value;
        return (
          <button
            key={t.key}
            role="tab"
            aria-selected={active}
            onClick={() => onChange(t.key)}
            className={
              "px-4 py-2 text-sm flex items-center gap-2 border-b-2 -mb-[1px] " +
              (active
                ? "border-accent-500 text-ink-50"
                : "border-transparent text-ink-300 hover:text-ink-50")
            }
          >
            <span>{t.label}</span>
            {t.badge}
          </button>
        );
      })}
    </div>
  );
}
