"""Core Pydantic schemas shared across all packages."""

from __future__ import annotations

from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


# ─── Enums ────────────────────────────────────────────────────────────────────

class AgentName(str, Enum):
    SUPERVISOR = "supervisor"
    WEB_SEARCH = "web_search"
    EDGAR_PARSER = "edgar_parser"
    RETRIEVAL = "retrieval"
    SQL_GENERATOR = "sql_generator"
    SUMMARIZER = "summarizer"
    CRITIC = "critic"


class CriticVerdict(str, Enum):
    PASS = "pass"
    REVISE = "revise"
    ESCALATE = "escalate"


class EventType(str, Enum):
    """WebSocket event types streamed to the frontend."""
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    THOUGHT = "thought"
    EVIDENCE_ADDED = "evidence_added"
    DRAFT_CHUNK = "draft_chunk"
    CRITIQUE = "critique"
    HUMAN_APPROVAL_REQUIRED = "human_approval_required"
    SESSION_DONE = "session_done"
    ERROR = "error"


# ─── Evidence & Citations ─────────────────────────────────────────────────────

class Evidence(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    source_type: str  # "edgar_filing" | "web" | "sql_result" | "rag_chunk"
    source_url: str | None = None
    ticker: str | None = None
    filing_type: str | None = None  # "10-K" | "10-Q" | "8-K"
    section: str | None = None  # "MD&A" | "Risk Factors" etc.
    text: str
    relevance_score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Citation(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    evidence_id: UUID
    claim_text: str  # the exact claim this citation supports
    is_verified: bool = False
    nli_score: float | None = None  # entailment score from Critic


# ─── Critic output ────────────────────────────────────────────────────────────

class FailingClaim(BaseModel):
    claim: str
    reason: str
    suggested_fix: str | None = None


class CritiqueResult(BaseModel):
    verdict: CriticVerdict
    failing_claims: list[FailingClaim] = Field(default_factory=list)
    summary: str = ""


# ─── LangGraph AgentState ─────────────────────────────────────────────────────

class AgentState(BaseModel):
    """Typed state carried through the LangGraph StateGraph."""

    session_id: UUID = Field(default_factory=uuid4)
    query: str
    plan: str = ""
    messages: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    draft: str = ""
    critique: CritiqueResult | None = None
    iteration_count: int = 0
    human_approval_required: bool = False
    # Which agent is currently active (for DAG visualization)
    active_agent: AgentName | None = None
    error: str | None = None


# ─── WebSocket events ─────────────────────────────────────────────────────────

class AgentEvent(BaseModel):
    """Envelope for events streamed over WebSocket."""

    session_id: UUID
    event_type: EventType
    agent: AgentName | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


# ─── API request/response ─────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=5, max_length=2000)
    tickers: list[str] = Field(default_factory=list, description="Optional ticker hints")
    stream: bool = Field(True)


class SessionResponse(BaseModel):
    session_id: UUID
    status: str  # "queued" | "running" | "done" | "error"
    ws_url: str
