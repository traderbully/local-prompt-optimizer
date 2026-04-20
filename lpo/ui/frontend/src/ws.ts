import { useEffect, useRef, useState } from "react";
import { wsUrl } from "./api";

export type WsEvent = {
  type:
    | "hello"
    | "iteration"
    | "status"
    | "awaiting_signal"
    | "mode_changed"
    | "error"
    | "done";
  data: Record<string, any>;
};

// Minimal WebSocket hook — one run per mount. Reconnects once on transient
// disconnects; the backend sends a `done` terminal event and then closes.
export function useRunWebSocket(runId: string | null) {
  const [events, setEvents] = useState<WsEvent[]>([]);
  const [connected, setConnected] = useState(false);
  const ref = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (!runId) return;
    let cancelled = false;

    const connect = () => {
      const ws = new WebSocket(wsUrl(runId));
      ref.current = ws;
      ws.onopen = () => setConnected(true);
      ws.onmessage = (ev) => {
        try {
          const data = JSON.parse(ev.data) as WsEvent;
          setEvents((prev) => [...prev, data]);
        } catch {
          // ignore non-JSON heartbeats
        }
      };
      ws.onclose = () => {
        setConnected(false);
        // Final `done` already delivered; no reconnect after a clean close.
      };
      ws.onerror = () => {
        setConnected(false);
      };
    };

    connect();
    return () => {
      cancelled = true;
      void cancelled;
      ref.current?.close();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runId]);

  return { events, connected };
}
