import { useEffect, useRef } from "react";
import { useSessionStore } from "@/store/sessionStore";
import type { AgentEvent, EventType } from "@/types/agent";

// ── Appearance maps ───────────────────────────────────────────────────────────

const EVENT_BADGE: Record<EventType, string> = {
  agent_start:            "bg-blue-500/20 text-blue-400",
  agent_end:              "bg-green-500/20 text-green-400",
  tool_call:              "bg-purple-500/20 text-purple-400",
  tool_result:            "bg-purple-500/10 text-purple-300",
  thought:                "bg-muted text-muted-foreground",
  evidence_added:         "bg-yellow-500/20 text-yellow-400",
  draft_chunk:            "bg-primary/20 text-primary",
  critique:               "bg-orange-500/20 text-orange-400",
  human_approval_required:"bg-red-500/20 text-red-400",
  session_done:           "bg-green-500/30 text-green-300",
  error:                  "bg-red-500/30 text-red-400",
};

const EVENT_LABEL: Record<EventType, string> = {
  agent_start:            "start",
  agent_end:              "done",
  tool_call:              "tool↑",
  tool_result:            "tool↓",
  thought:                "thought",
  evidence_added:         "evidence",
  draft_chunk:            "token",
  critique:               "critique",
  human_approval_required:"approval",
  session_done:           "session done",
  error:                  "error",
};

// ── Helpers ───────────────────────────────────────────────────────────────────

function payloadSummary(event: AgentEvent): string {
  const p = event.payload;
  switch (event.event_type) {
    case "agent_start":
    case "agent_end":
      return event.agent ?? "";
    case "evidence_added": {
      const ev = p.evidence as { source_type?: string; ticker?: string } | undefined;
      return ev ? `${ev.ticker ?? ev.source_type ?? ""}` : "";
    }
    case "draft_chunk":
      return typeof p.chunk === "string"
        ? p.chunk.slice(0, 40).replace(/\n/g, "↵")
        : "";
    case "critique": {
      const r = p.result as { verdict?: string } | undefined;
      return r?.verdict ?? "";
    }
    case "error":
      return typeof p.message === "string" ? p.message.slice(0, 60) : "unknown";
    case "session_done":
      return "✓";
    default:
      return "";
  }
}

// ── Component ─────────────────────────────────────────────────────────────────

export function EventLog() {
  const events = useSessionStore((s) => s.events);
  const bottomRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to newest event
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [events.length]);

  return (
    <div className="flex flex-col h-full">
      <div className="px-4 py-2 border-b border-border shrink-0">
        <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
          Events ({events.length})
        </span>
      </div>

      <div className="flex-1 overflow-y-auto px-3 py-2 space-y-0.5 font-mono text-[10px]">
        {events.length === 0 ? (
          <p className="text-muted-foreground px-1 py-4 text-center">No events yet.</p>
        ) : (
          events.map((ev, i) => (
            <EventRow key={i} event={ev} />
          ))
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}

function EventRow({ event }: { event: AgentEvent }) {
  const badgeClass = EVENT_BADGE[event.event_type] ?? "bg-muted text-muted-foreground";
  const label      = EVENT_LABEL[event.event_type] ?? event.event_type;
  const summary    = payloadSummary(event);

  return (
    <div className="flex items-baseline gap-2 py-0.5 hover:bg-muted/30 rounded px-1">
      <span className={`shrink-0 px-1.5 py-0.5 rounded text-[9px] font-semibold ${badgeClass}`}>
        {label}
      </span>
      {summary && (
        <span className="text-muted-foreground truncate leading-tight">{summary}</span>
      )}
    </div>
  );
}
