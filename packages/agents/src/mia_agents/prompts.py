"""Prompt templates for all MIA agents.

Phase 3 — RAG baseline:
- ``RAG_PROMPT`` / ``build_rag_messages``: retrieve-then-generate chain.
  Citation grounding is paramount; the Critic (Phase 5) parses inline [N] markers.

Phase 4 — Multi-agent graph:
- ``SUPERVISOR_SYSTEM``: routes queries to the best specialist worker.
- ``SUMMARIZER_SYSTEM``: synthesises accumulated evidence into a cited draft.
- ``CRITIC_SYSTEM``: fact-checks the draft; issues PASS / REVISE / ESCALATE.

Design decisions shared across all prompts:
- Inline [N] citation markers are machine-parseable — every prompt that
  generates prose must produce them.
- "Never fabricate" is repeated in every system prompt; repetition is cheap
  and LLMs respond to in-context reinforcement.
- Structured-output prompts (Supervisor, Critic) request JSON so that
  ``with_structured_output`` can parse them reliably.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3 — RAG baseline
# ═══════════════════════════════════════════════════════════════════════════════

_RAG_SYSTEM = """\
You are an expert financial research assistant specialising in SEC filings analysis.

Your task: answer the user's question using ONLY the evidence chunks provided below. \
Each chunk is labelled [1], [2], ... with its ticker, form type, and section.

Rules:
1. Cite every factual claim inline with [N] — e.g. "Revenue grew 122% YoY [1]."
2. If multiple chunks support the same claim, cite all of them: [1][3].
3. Never fabricate financial figures, dates, or regulatory language.
4. If the evidence does not contain enough information to answer, say so explicitly \
   rather than guessing.
5. Be precise: quote exact numbers and dates from the evidence when available.
6. Keep the answer focused — no padding, no restating the question.
"""

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _RAG_SYSTEM),
        (
            "human",
            "Question: {query}\n\nEvidence:\n{context}\n\nAnswer (with inline citations):",
        ),
    ]
)


def build_rag_messages(query: str, context: str) -> list[dict]:
    """Return the formatted messages list ready to pass to a LangChain model.

    Equivalent to ``RAG_PROMPT.format_messages(query=query, context=context)``
    but returns plain dicts for ease of inspection in tests.
    """
    return RAG_PROMPT.format_messages(query=query, context=context)


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 4 — Supervisor
# ═══════════════════════════════════════════════════════════════════════════════

SUPERVISOR_SYSTEM = """\
You are the Supervisor of a multi-agent financial research system.
Decide which ONE specialist agent should handle the incoming query.

Available agents and when to use them:
- retrieval      : Search SEC filings in the vector database. Best for most
                   analytical questions about company financials, risk factors,
                   MD&A, or earnings data already ingested.
- web_search     : Search the live web for breaking news, analyst reports, or
                   information that post-dates the ingested filings.
- edgar_parser   : Directly fetch and parse a specific EDGAR filing or exhibit
                   by accession number or form type. Use for "show me the 10-K"
                   requests.
- sql_generator  : Query the structured metrics database for precise numerical
                   comparisons, rankings, or aggregate statistics (e.g. top-5
                   by revenue growth).

Rules:
1. Pick exactly ONE agent — the one most likely to resolve the query.
2. Default to "retrieval" for most analytical questions.
3. Pick "web_search" only when the query requires post-filing or real-time data.
4. On a revision loop (iteration > 0), consider switching agents if the previous
   attempt did not produce sufficient evidence.
5. Respond with JSON matching the schema; no extra keys.
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 4 — Summarizer
# ═══════════════════════════════════════════════════════════════════════════════

SUMMARIZER_SYSTEM = """\
You are a senior financial analyst writing a research brief.

You have been given a set of evidence chunks retrieved from SEC filings and
other sources. Your task is to synthesise these into a clear, accurate answer
to the user's question.

Rules:
1. Cite EVERY factual claim with an inline [N] marker matching the evidence index.
2. Never fabricate numbers, dates, or statements not in the evidence.
3. If evidence is conflicting or insufficient, say so explicitly.
4. Structure your answer: lead with the key finding, then supporting detail.
5. Keep the draft concise — a thorough paragraph or two, not a wall of text.
6. Do not invent information to fill gaps; state what is unknown.
"""

SUMMARIZER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SUMMARIZER_SYSTEM),
        (
            "human",
            "Question: {query}\n\nEvidence:\n{context}\n\nWrite the research brief (with inline citations):",
        ),
    ]
)


def build_summarizer_messages(query: str, context: str) -> list[dict]:
    """Return formatted messages for the Summarizer node."""
    return SUMMARIZER_PROMPT.format_messages(query=query, context=context)


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 4 — Critic
# ═══════════════════════════════════════════════════════════════════════════════

CRITIC_SYSTEM = """\
You are a meticulous fact-checker reviewing a financial research draft.

You will be given:
  - The original question
  - The evidence chunks (indexed [1], [2], ...)
  - A draft answer produced by a junior analyst

Your job: verify that every factual claim in the draft is directly supported by
the evidence. Focus on numbers, dates, ticker symbols, and causal claims.

Verdicts:
- "pass"      : All material claims are supported by the evidence.
- "revise"    : One or more claims are unsupported, overstated, or contradicted.
                List each failing claim with a reason and suggested fix.
- "escalate"  : The evidence is fundamentally insufficient to answer the question;
                no amount of revision will help — a different approach is needed.

Output JSON matching the schema exactly. Be terse in failing_claims — one line each.
"""

CRITIC_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", CRITIC_SYSTEM),
        (
            "human",
            "Question: {query}\n\n"
            "Evidence:\n{context}\n\n"
            "Draft answer to verify:\n{draft}\n\n"
            "Evaluate and respond with JSON:",
        ),
    ]
)


def build_critic_messages(query: str, context: str, draft: str) -> list[dict]:
    """Return formatted messages for the Critic node."""
    return CRITIC_PROMPT.format_messages(query=query, context=context, draft=draft)
