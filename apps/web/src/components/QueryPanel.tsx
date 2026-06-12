import { useState } from "react";
import { useSessionStore } from "@/store/sessionStore";

const EXAMPLE_QUERIES = [
  "How is NVDA's data-center revenue concentration evolving vs AMD's, and what's the risk narrative in their latest 10-Ks?",
  "Compare MSFT and GOOGL cloud segment growth over the last 4 quarters.",
  "What liquidity risks does TSLA flag in its most recent 10-Q?",
];

export function QueryPanel() {
  const { query, status, setQuery, startSession, reset } = useSessionStore();
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!query.trim() || loading) return;

    setLoading(true);
    try {
      const res = await fetch("/api/sessions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, stream: true }),
      });
      if (!res.ok) throw new Error(await res.text());
      const { session_id } = (await res.json()) as { session_id: string };
      startSession(session_id);
      connectWebSocket(session_id);
    } catch (err) {
      console.error("Failed to start session:", err);
    } finally {
      setLoading(false);
    }
  }

  function connectWebSocket(sessionId: string) {
    const { handleEvent } = useSessionStore.getState();
    const ws = new WebSocket(`/ws/sessions/${sessionId}/stream`);
    ws.onmessage = (e) => {
      try {
        handleEvent(JSON.parse(e.data));
      } catch {
        /* ignore parse errors */
      }
    };
    ws.onerror = () => handleEvent({ session_id: sessionId, event_type: "error", payload: {} });
  }

  const isRunning = status === "running" || status === "connecting";

  return (
    <div className="p-4 border-b border-border space-y-3">
      <form onSubmit={handleSubmit} className="space-y-2">
        <textarea
          className="w-full h-28 px-3 py-2 text-sm rounded-md border border-border bg-card resize-none focus:outline-none focus:ring-1 focus:ring-ring placeholder:text-muted-foreground"
          placeholder="Ask about any public company or sector…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          disabled={isRunning}
        />
        <div className="flex gap-2">
          <button
            type="submit"
            disabled={!query.trim() || isRunning || loading}
            className="flex-1 py-1.5 text-sm font-medium rounded-md bg-primary text-primary-foreground disabled:opacity-40 hover:opacity-90 transition-opacity"
          >
            {loading ? "Starting…" : isRunning ? "Running…" : "Analyze"}
          </button>
          {status !== "idle" && (
            <button
              type="button"
              onClick={reset}
              className="px-3 py-1.5 text-sm rounded-md border border-border hover:bg-muted transition-colors"
            >
              Reset
            </button>
          )}
        </div>
      </form>

      {status === "idle" && (
        <div className="space-y-1">
          <p className="text-xs text-muted-foreground">Examples</p>
          {EXAMPLE_QUERIES.map((q, i) => (
            <button
              key={i}
              onClick={() => setQuery(q)}
              className="block w-full text-left text-xs px-2 py-1.5 rounded hover:bg-muted transition-colors text-muted-foreground hover:text-foreground truncate"
            >
              {q}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
