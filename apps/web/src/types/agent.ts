/** Mirror of packages/shared/src/mia_shared/schemas.py — kept in sync manually. */

export type AgentName =
  | "supervisor"
  | "web_search"
  | "edgar_parser"
  | "retrieval"
  | "sql_generator"
  | "summarizer"
  | "critic";

export type CriticVerdict = "pass" | "revise" | "escalate";

export type EventType =
  | "agent_start"
  | "agent_end"
  | "tool_call"
  | "tool_result"
  | "thought"
  | "evidence_added"
  | "draft_chunk"
  | "critique"
  | "human_approval_required"
  | "session_done"
  | "error";

export interface Evidence {
  id: string;
  source_type: string;
  source_url?: string;
  ticker?: string;
  filing_type?: string;
  section?: string;
  text: string;
  relevance_score?: number;
  metadata: Record<string, unknown>;
}

export interface FailingClaim {
  claim: string;
  reason: string;
  suggested_fix?: string;
}

export interface CritiqueResult {
  verdict: CriticVerdict;
  failing_claims: FailingClaim[];
  summary: string;
}

export interface AgentEvent {
  session_id: string;
  event_type: EventType;
  agent?: AgentName;
  payload: Record<string, unknown>;
}

export type AgentNodeStatus = "idle" | "active" | "done" | "error";

export interface AgentNodeData {
  label: string;
  agent: AgentName;
  status: AgentNodeStatus;
  lastEvent?: string;
}
