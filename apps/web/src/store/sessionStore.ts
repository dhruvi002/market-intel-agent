import { create } from "zustand";
import type { AgentEvent, AgentName, AgentNodeStatus, CritiqueResult, Evidence } from "@/types/agent";

interface SessionState {
  sessionId: string | null;
  status: "idle" | "connecting" | "running" | "done" | "error";
  query: string;
  draft: string;
  evidence: Evidence[];
  events: AgentEvent[];
  critique: CritiqueResult | null;
  agentStatuses: Record<AgentName, AgentNodeStatus>;
  humanApprovalRequired: boolean;
  error: string | null;

  // Actions
  setQuery: (q: string) => void;
  startSession: (sessionId: string) => void;
  handleEvent: (event: AgentEvent) => void;
  reset: () => void;
}

const DEFAULT_AGENT_STATUSES: Record<AgentName, AgentNodeStatus> = {
  supervisor: "idle",
  web_search: "idle",
  edgar_parser: "idle",
  retrieval: "idle",
  sql_generator: "idle",
  summarizer: "idle",
  critic: "idle",
};

export const useSessionStore = create<SessionState>((set) => ({
  sessionId: null,
  status: "idle",
  query: "",
  draft: "",
  evidence: [],
  events: [],
  critique: null,
  agentStatuses: { ...DEFAULT_AGENT_STATUSES },
  humanApprovalRequired: false,
  error: null,

  setQuery: (query) => set({ query }),

  startSession: (sessionId) =>
    set({
      sessionId,
      status: "connecting",
      draft: "",
      evidence: [],
      events: [],
      critique: null,
      agentStatuses: { ...DEFAULT_AGENT_STATUSES },
      humanApprovalRequired: false,
      error: null,
    }),

  handleEvent: (event) =>
    set((state) => {
      const events = [...state.events, event];
      let { draft, evidence, critique, agentStatuses, status, humanApprovalRequired, error } = state;

      switch (event.event_type) {
        case "agent_start":
          if (event.agent) {
            agentStatuses = { ...agentStatuses, [event.agent]: "active" };
            status = "running";
            // Reset draft each time the summarizer starts so that revise-loop
            // re-runs replace the previous draft rather than appending to it.
            if (event.agent === "summarizer") {
              draft = "";
            }
          }
          break;
        case "agent_end":
          if (event.agent) {
            agentStatuses = { ...agentStatuses, [event.agent]: "done" };
          }
          break;
        case "draft_chunk":
          draft += (event.payload.chunk as string) ?? "";
          break;
        case "evidence_added":
          evidence = [...evidence, event.payload.evidence as Evidence];
          break;
        case "critique":
          critique = event.payload.result as CritiqueResult;
          break;
        case "human_approval_required":
          humanApprovalRequired = true;
          break;
        case "session_done":
          status = "done";
          agentStatuses = Object.fromEntries(
            Object.entries(agentStatuses).map(([k, v]) => [k, v === "active" ? "done" : v])
          ) as Record<AgentName, AgentNodeStatus>;
          break;
        case "error":
          status = "error";
          error = (event.payload.message as string | undefined) ?? "An unknown error occurred.";
          break;
      }

      return { events, draft, evidence, critique, agentStatuses, status, humanApprovalRequired, error };
    }),

  reset: () =>
    set({
      sessionId: null,
      status: "idle",
      query: "",
      draft: "",
      evidence: [],
      events: [],
      critique: null,
      agentStatuses: { ...DEFAULT_AGENT_STATUSES },
      humanApprovalRequired: false,
      error: null,
    }),
}));
