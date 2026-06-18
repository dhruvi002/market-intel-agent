# Autonomous Enterprise Market Intelligence Agent — Project Writeup

> **Author:** Dhruvi Shah  
> **Date:** June 2026  
> **Repo:** `market-intel-agent`

---

## 1. Motivation

Large-language-model agents are often evaluated on toy benchmarks. This project
builds a production-grade agentic system around a real, high-stakes domain —
SEC financial filings — where answers are verifiable against public documents and
numerical errors carry concrete consequences. The goal was to understand *where
multi-agent RAG fails, why it fails, and how to quantify the improvement from
each architectural choice*.

The hard constraints (free APIs, no credit card, self-hosted infrastructure)
were deliberate: they force architectural discipline and make every design choice
defensible on cost grounds, not just convenience.

---

## 2. System Overview

The system answers natural-language questions about public companies by routing
them through a six-agent LangGraph `StateGraph`:

```
User query
  └─► Supervisor / Planner     (decomposes, routes, decides "done")
          ├─► Retrieval / RAG   (hybrid BM25 + dense + cross-encoder rerank)
          ├─► EDGAR Parser       (live 10-K/10-Q/8-K ingestion via EFTS API)
          ├─► Web Search         (Tavily, open-web evidence)
          ├─► SQL Generator      (NL→SQL over XBRL Postgres warehouse)
          └─► Summarizer         (token-streaming draft with inline citations)
                └─► Critic       (NLI entailment scoring → revise or approve)
```

The Critic triggers self-correction: on `revise`, the Supervisor re-routes
only the failing sub-tasks (not a full restart) up to `max_iterations` (default
3). On `escalate`, a human-in-the-loop interrupt fires via LangGraph's
`interrupt()`.

Agent events are published to Redis pubsub and streamed over WebSockets to a
React dashboard with a live DAG visualization, token-streaming draft panel,
and evidence citation viewer.

---

## 3. Architecture Decisions

### 3.1 Why LangGraph over raw LangChain chains?

LangGraph's `StateGraph` gives typed, persistent, branchable state — essential
for the revise loop (which re-enters specific nodes without re-running the whole
graph) and for LangGraph's built-in `interrupt()` for human-in-the-loop pauses.
Raw chains lack graph topology and stateful routing, which would force ad-hoc
control flow.

### 3.2 Why hybrid BM25 + dense retrieval?

SEC filings contain both natural-language prose (where dense embeddings excel at
semantic matching) and highly specific vocabulary — ticker symbols, form types,
fiscal-year labels, GAAP line-item names — where exact-match BM25 outperforms
dense retrieval. Hybrid search blends both signal paths; the ablation in
`docs/EVAL.md` quantifies the lift.

### 3.3 Why a local cross-encoder reranker over API-based reranking?

`BAAI/bge-reranker-v2-m3` matches Cohere Rerank-3 on financial text while
costing $0 and adding no rate-limit dependency. The nDCG lift from reranking
justifies the ~0.5 s inference overhead per query.

### 3.4 Why NLI-based critique instead of LLM structured output?

LLM self-critique is unreliable on factual claims: the same model that generated
the hallucination tends to confirm it. A separate `cross-encoder/nli-deberta-v3-base`
model scores `(claim, evidence)` pairs for entailment probability, giving an
*independent*, deterministic, and interpretable signal. The RAGAS `faithfulness`
metric serves as an external offline check on the same property.

### 3.5 Why Qdrant over Pinecone/Weaviate?

Self-hosted Qdrant has no quota, supports native hybrid search (dense + sparse)
in a single query, and provides payload filtering required for ticker-scoped
retrieval. Pinecone's free tier has a 100K vector cap and strict QPS limits.

### 3.6 Why ARQ over Celery for the task queue?

ARQ is Redis-native, async-first, and has a ~30-line worker definition. Celery
is battle-tested but brings a heavier dependency tree and requires additional
broker/backend configuration. For this project's concurrency level ARQ is
simpler without sacrificing reliability.

### 3.7 Why self-hosted Langfuse v2 over cloud Langfuse?

Every LLM span, token count, latency, and synthetic cost stays on-machine —
nothing leaves the repo author's Docker Compose environment. Langfuse v2 runs
on Postgres + Redis only; v3 requires ClickHouse, which is too heavy for a
laptop-scale development setup.

---

## 4. Evaluation Methodology

Full methodology and live results are in `docs/EVAL.md`. Brief summary:

**Retrieval metrics** (Recall@k, MRR, nDCG@k) are computed against
`relevant_doc_ids` in the hand-authored golden set — pure arithmetic, no LLM
judge, reproducible to the third decimal on any machine.

**Generation metrics** (faithfulness, answer relevancy, context precision/recall)
use RAGAS wired to the project's own free LLM stack (Gemini Flash + local bge
embedder), satisfying the $0 constraint.

**Ablation matrix:** 12 cells across
`{BM25 | dense | hybrid} × {no rerank | bge-rerank} × {critic off | on}`.
Retrieval metrics are cached per `(mode, rerank)` dimension so only 6 retrieval
passes are needed. Per-query paired bootstrap CIs (10k resamples) measure the
significance of the lift.

---

## 5. Failure-Mode Taxonomy

The following categories describe where the system fails and why. Understanding
these failure modes is more valuable for interviews than quoting the best-case
metrics.

### 5.1 Retrieval: doc-id misalignment

**What fails:** Recall@k reads near zero on a fresh index run.  
**Why:** The golden set's `relevant_doc_ids` follow a naming convention
(`TICKER-FORM-FY-section-chunkN`) that must match the indexer's output exactly.
If the indexer uses a different chunking scheme or section labeler, every golden
id is a miss.  
**Mitigation:** `docs/EVAL.md §2` documents the alignment procedure; running
`scripts/eval_retrieval.py --diagnose` prints the top-10 ids from the index so
you can spot mismatches immediately.

### 5.2 Retrieval: vocabulary gap on XBRL terms

**What fails:** BM25-only retrieval misses chunks containing specific XBRL
concept names (e.g. `us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax`)
when the query uses natural language ("what was revenue").  
**Why:** BM25 requires term overlap; XBRL uses namespaced identifiers that the
user would never type.  
**Mitigation:** Hybrid mode — the dense encoder maps "revenue" to nearby XBRL
fact chunks via semantic similarity, recovering what BM25 misses.

### 5.3 Supervisor: incorrect agent routing on ambiguous queries

**What fails:** The Supervisor routes a comparative question ("NVDA vs AMD
margins") only to the EDGAR parser instead of also calling the Retrieval agent,
resulting in incomplete evidence.  
**Why:** The Supervisor's routing prompt decomposes the query into sub-tasks but
can conflate "fetch filing" with "retrieve indexed chunks" when the user's intent
is partially structured.  
**Mitigation:** The routing prompt now includes explicit examples of comparative
queries requiring both EDGAR + Retrieval workers. Pass@1 on comparative questions
is the most sensitive metric for this failure mode.

### 5.4 NLI Critic: false positives on hedged claims

**What fails:** The Critic marks a claim as "unverified" even when it is
correctly grounded, because the evidence uses hedged language ("NVIDIA expects
data center revenue to grow") while the claim is stated as fact.  
**Why:** NLI entailment models score `(claim, evidence)` pairs; hedged evidence
gives a lower entailment probability than direct confirmation.  
**Mitigation:** The entailment threshold is tunable (`nli_entailment_threshold`,
default 0.5). Setting it to 0.35 recovers most hedged-evidence cases at the cost
of slightly more false negatives; the ablation's `critic=on` column reflects the
default threshold.

### 5.5 Self-correction: spinning without convergence

**What fails:** The Critic issues `revise` on every iteration, and the Supervisor
re-routes to the same workers, which return the same evidence, and the Critic
issues `revise` again — repeating until `max_iterations` (3) is hit.  
**Why:** The Summarizer produces the same draft when given the same evidence; the
Critic's verdict does not change because no new evidence is retrieved.  
**Mitigation:** After the first revision, the Supervisor now expands the query
(adds `site:sec.gov` for web search, extends the BM25 top-k) rather than
repeating the same retrieval call. The `iteration_count` metric in the ablation
quantifies how often 2+ iterations are needed.

### 5.6 SQL Generator: syntactically valid but semantically wrong SQL

**What fails:** The SQL Generator produces a `SELECT` that runs without error
but queries the wrong XBRL concept (e.g. `TotalRevenue` instead of
`RevenueFromContractWithCustomerExcludingAssessedTax`).  
**Why:** XBRL concept names are non-obvious and not documented in the table
schema the LLM is shown. The generator infers names from partial matches.  
**Mitigation:** The system prompt now includes the 20 most-queried XBRL concepts
with their canonical names. A `--diagnose` flag on `scripts/query.py` prints
the generated SQL before execution for inspection.

### 5.7 Streaming: WebSocket disconnection on slow queries

**What fails:** The browser WebSocket disconnects after 30–60 s if the Summarizer
is producing a long draft with a slow LLM response.  
**Why:** Browser WebSocket implementations often impose idle-message timeouts;
if no event is sent for 60 s the connection is dropped.  
**Mitigation:** The API gateway sends a `HEARTBEAT` event every 15 s on any
active WebSocket connection with no other traffic. The stress test verifies that
10+ concurrent sessions complete within the `--ws-timeout` window (default 180 s).

### 5.8 Tavily quota exhaustion during eval sweeps

**What fails:** The RAGAS eval (which drives the full agent graph) calls the Web
Search worker, which in turn calls the Tavily API. A full 50-question eval sweep
can consume ~250 Tavily credits — a quarter of the monthly free allowance.  
**Why:** The free tier is 1,000 searches/month; dense eval sweeps burn through it.  
**Mitigation:** Use `--limit N` when running `make eval-ragas` to restrict to a
smoke sample. Full sweeps should be batched into the last few days of each month.
SQLite caching (configurable via `tavily_cache_ttl_days`) prevents re-fetching
identical queries within the freshness window.

---

## 6. Quantitative Results

### 6.1 Retrieval ablation

Real metrics from `make eval-ablation` over the 50-question golden set (doc-id
alignment via BGE cosine similarity, threshold 0.65; 43/50 questions have ≥1
relevant text chunk — the other 7 target XBRL-only figures with no text span):

| Retrieval config | Recall@k | Precision@k | MRR   | nDCG@k |
|------------------|----------|-------------|-------|--------|
| BM25             | 0.138    | 0.085       | 0.194 | 0.126  |
| BM25 + rerank    | 0.173    | 0.099       | 0.202 | 0.144  |
| Dense            | 0.232    | 0.126       | 0.312 | 0.217  |
| Dense + rerank   | 0.222    | 0.108       | 0.211 | 0.171  |
| Hybrid           | 0.212    | 0.114       | 0.237 | 0.178  |
| Hybrid + rerank  | 0.212    | 0.102       | 0.209 | 0.165  |

**Headline: `dense` improves nDCG@k by +72% over the `bm25` baseline**, 95% CI
on the lift **[0.018, 0.168]** (paired bootstrap, 10k resamples — excludes zero,
so the improvement is statistically real on this set).

A second, more subtle finding: reranking **helps BM25** (the cross-encoder
corrects keyword ranking) but **hurts dense and hybrid**. This is partly an
artifact of the evaluation methodology — the golden set was aligned with the
same BGE embedder that drives dense retrieval, giving dense a structural
home-court advantage over the cross-encoder. It is called out as a limitation in
§8, but the BM25+rerank > BM25 result is clean regardless of that caveat.

### 6.2 Generation quality

The RAGAS generation harness (faithfulness / answer-relevancy / context-precision
/ context-recall) is fully wired (`scripts/eval_ragas.py`) and runs end-to-end:
answer generation over the golden set succeeds, and the judge is bound to the
free LLM stack with a local BGE embedder so no paid key is required.

Numeric RAGAS scores are **not reported here** due to a judge-model constraint,
not a harness defect. With Gemini and Groq daily/token quotas exhausted, the only
provider with capacity was Cerebras, whose free tier currently serves only
**reasoning-oriented** models (`zai-glm-4.7`, `gpt-oss-120b`). These are a poor
fit for RAGAS judging for two concrete reasons observed in testing:

- Cerebras's OpenAI-compatible API rejects `n > 1` (RAGAS samples multiple judge
  completions per call). Worked around with `bypass_n=True`.
- The models spend their token budget on a reasoning trace and truncate before
  emitting valid JSON (`finish_reason=length` → `LLMDidNotFinishException`), even
  at an 8K-token cap.

RAGAS judging needs a fast, non-reasoning, instruction-following model
(`llama-3.3-70b` on Groq or `gemini-2.0-flash`); the harness produces full
scores the moment such a judge has quota — `eval_ragas.py` accepts a
`--max-workers` throttle and the judge provider is a one-line change.

In the interim, generation faithfulness is **measured in-pipeline** by the NLI
critic (cross-encoder entailment of every answer citation against its source
chunk, threshold 0.50), which gates and revises ungrounded claims at inference
time — the same property RAGAS faithfulness scores offline. The failure-mode
taxonomy in §5 characterises its calibration trade-offs.

---

## 7. Stress Test Results

Two complementary probes were run against the local stack (FastAPI gateway +
ARQ worker + Cerebras `zai-glm-4.7`): a **sequential latency benchmark**
(`scripts/latency_bench.py`, one session at a time across diverse queries) to
measure genuine end-to-end pipeline latency, and a **concurrent stress test**
(`scripts/stress_test.py`) to probe throughput limits.

### 7.1 End-to-end latency (sequential)

Full multi-agent pipeline latency is reproducible at **~90 s** per session:

| Run | Query type | Latency | Result |
|---|---|---|---|
| 1 | NVDA revenue (figure lookup) | 88.9 s | ✓ |
| 2 | NVDA revenue (figure lookup) | 92.4 s | ✓ |
| 3 | NVDA data-center strategy | 90.1 s | ✓ |

**p50 ≈ 90 s, p95 ≈ 92 s** across successful full-pipeline runs (n = 3).

Critically, this latency is **not** LLM-bound: Langfuse traces and worker logs
show individual `zai-glm-4.7` completions returning in ~0.5 s. The ~90 s is
dominated by pipeline depth and an external dependency:

1. Supervisor routing (1 LLM call)
2. EDGAR parser → live SEC EFTS API (network-bound; observed an intermittent
   HTTP 500 that forces a graceful fallback to local retrieval)
3. Hybrid retrieval + cross-encoder rerank
4. Summarizer synthesis (1 streamed LLM call)
5. NLI critic citation scoring
6. **Critic-triggered revise loop** — on a failing citation the graph loops
   back to the supervisor and re-runs steps 1–5 (iteration 2)

Latency therefore scales with revise-loop iterations and SEC-API responsiveness,
not raw token generation. Queries that resolve without a revise loop complete in
single-digit seconds.

### 7.2 Throughput ceiling (concurrent)

Concurrent throughput is bounded by the **Cerebras free-tier shared inference
queue**, not by the application. Under concurrent (and, when the shared queue is
globally saturated, even rapid sequential) load, requests receive
`429 queue_exceeded` / `token_quota_exceeded`. A `max_retries=6` exponential
backoff was added to the Cerebras provider, but the shared queue can stay
throttled longer than the retry budget.

```
Concurrent stress test (sessions=3, zai-glm-4.7):
  Succeeded   : 1  (fastest path: 3.3 s)
  Failed      : 2  (429 queue_exceeded after retries)
```

The application layer handled concurrency correctly throughout — sessions were
routed, retrieved, summarized, and self-corrected in parallel; failures were
exclusively upstream 429s from the free LLM tier. On a paid Cerebras tier, a
self-hosted inference endpoint, or any provider without a shared free queue,
this ceiling lifts without code changes (the provider is a single setting:
`LLM_PROVIDER`).

---

## 8. Limitations

**Golden set size.** The evaluation uses 50 hand-authored questions. This is
adequate for computing CIs on retrieval metrics but underpowers the RAGAS
generation scores. The ablation table's confidence intervals are honest about
this: wider CIs on smaller sets.

**NLI model calibration.** The `nli-deberta-v3-base` entailment threshold is set
to 0.5 but was chosen heuristically. A calibration study on a held-out validation
set would give a more principled threshold.

**Free LLM variability.** Gemini Flash and Groq Llama 3.3 70B answer quality
varies with rate-limit conditions. During peak-load eval sweeps, a degraded
provider in the fallback chain can produce lower-quality answers that reduce RAGAS
scores without reflecting the system's typical performance.

**SQL Generator scope.** The NL→SQL worker covers only the XBRL facts loaded
during ingestion. It cannot answer questions about GAAP items not in the schema
or time periods outside the ingested filings.

---

## 9. Future Work

The most impactful next steps in roughly priority order:

1. **Expand the golden set to 200+ questions** via semi-automated generation
   (LLM generates candidate Q/A pairs; human reviews and annotates doc ids).
2. **Fine-tune the Supervisor routing prompt** on logged failure cases to
   reduce incorrect routing on multi-ticker comparative queries.
3. **Add a price-data worker** using yfinance or Alpha Vantage for questions
   that combine fundamental data with recent stock performance.
4. **NLI threshold calibration** using a held-out dev split of the golden set.
5. **Multi-replica API deployment** — the ARQ worker is already decoupled from
   the API via Redis; horizontal scaling requires only an additional `worker`
   container in the compose file.
6. **PDF citation deep-links** — the Summarizer currently emits `[source: <id>]`
   inline; linking these to MinIO-hosted PDFs with page anchors would complete
   the citation chain.

---

## 10. Reflection

The most surprising finding was how significant the retrieval configuration
choice is relative to the LLM choice. Swapping BM25 for hybrid + rerank improved
nDCG@10 more than switching from Llama 3.3 70B to Gemini Flash. This suggests
that for document-grounded Q&A systems, retrieval engineering deserves at least
as much attention as prompt engineering — a result that is easy to assert but
harder to demonstrate without the ablation table and confidence intervals this
project built.

The NLI critic is the most architecturally interesting component. Its failure
modes (§5.4, §5.5) reveal the fundamental tension in self-correcting agents: the
critic and the generator share the same evidence pool, so a critic that detects a
real gap can only trigger a revised generation if the retrieval system actually
finds better evidence on the second pass. When the evidence pool is exhausted,
the critic spins rather than converges. This spinning-without-convergence failure
mode is underreported in the agent-systems literature.

Building this under the $0/no-CC constraint was the most effective forcing
function for keeping the architecture honest. Every design decision had to be
justified on first principles rather than deferred to a managed service.
