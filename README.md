# Autonomous Enterprise Market Intelligence Agent

> **Status:** In active development. Phase 3 (single-agent RAG baseline) complete. Phase 4 (LangGraph multi-agent skeleton) next.

A multi-agent system that autonomously gathers, synthesizes, and verifies competitive intelligence from SEC filings, financial data, and open-web sources — streaming agent reasoning live to a React dashboard.

## Architecture

**Supervisor–Worker** agentic graph (LangGraph) with 6 specialized agents:
- **Supervisor / Planner** — decomposes queries, routes, decides when the answer is credible
- **Web Search** — open-web evidence gathering with source credibility scoring
- **EDGAR / Financial Parser** — 10-K/10-Q/8-K ingestion + XBRL structured extraction
- **Retrieval / RAG** — hybrid search (BM25 + dense) with cross-encoder reranking
- **SQL Generator** — NL→SQL over Postgres warehouse of XBRL facts
- **Critic / Verifier** — claim-level grounding check, triggers self-correction loops

## Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph, LangChain |
| LLMs | Gemini 2.0 Flash, Groq Llama 3.3 70B, Cerebras (fallback) |
| Embeddings | `bge-large-en-v1.5` (local) |
| Reranker | `bge-reranker-v2-m3` (local) |
| Vector DB | Qdrant (self-hosted) |
| Relational DB | Postgres 16 + pgvector |
| Cache / Queue | Redis 7 + ARQ |
| Object Store | MinIO |
| Backend | FastAPI + WebSockets |
| Frontend | React 18 + TypeScript + Vite + React Flow |
| Observability | Langfuse (self-hosted) |
| Eval | RAGAS + custom agent harness |
| Infra | Docker Compose |

## Quick Start

```bash
# Coming in Phase 0
docker compose up -d
```

## Project Phases

See [`docs/PLAN.md`](docs/PLAN.md) for the full build plan.

| Phase | Focus | Status |
|---|---|---|
| 0 | Bootstrap & DevOps foundation | 🔄 In progress |
| 1 | Ingestion pipelines (EDGAR + XBRL + PDF) | ✅ Complete |
| 2 | Retrieval stack (Qdrant + BM25 + reranker) | ✅ Complete |
| 3 | Single-agent RAG baseline | ✅ Complete |
| 4 | LangGraph multi-agent skeleton | ⬜ Pending |
| 5 | Critic agent + self-correction loops | ⬜ Pending |
| 6 | FastAPI + WebSocket streaming | ⬜ Pending |
| 7 | React dashboard | ⬜ Pending |
| 8 | Observability + eval harness + ablations | ⬜ Pending |
| 9 | Hardening + cloud deploy + writeup | ⬜ Pending |

## Evaluation

Ablation matrix: `{BM25 | dense | hybrid} × {no rerank | bge-rerank} × {no critic | critic}` evaluated on 50 hand-authored SEC Q/A pairs. Full results in [`docs/EVAL.md`](docs/EVAL.md) (populated in Phase 8).
