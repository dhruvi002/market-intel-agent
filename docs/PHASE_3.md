# Phase 3 — Single-Agent RAG Baseline

**Status:** ✅ Complete
**Unlocks:** Phase 4 (LangGraph multi-agent skeleton)

---

## What Was Built

A full retrieve-then-generate RAG pipeline in `packages/agents/`, wiring together the Phase 2 retrieval stack with a multi-provider LLM fallback chain. No multi-agent orchestration yet — this is the single-step baseline that Phase 4 agents will call as a sub-component.

### `packages/agents/src/mia_agents/`

| File | Purpose |
|---|---|
| `llm.py` | `LLMProvider` enum + `get_llm()` factory. Returns a LangChain `BaseChatModel` — either a pinned provider or a `with_fallbacks()` chain: Gemini 2.0 Flash → Groq Llama 3.3 70B → Cerebras Llama 3.3 70B. All free-tier, no CC. |
| `prompts.py` | `RAG_PROMPT` (`ChatPromptTemplate`) and `build_rag_messages()`. System prompt enforces inline `[N]` citation markers and bans fabrication. |
| `rag_agent.py` | `RAGResponse` Pydantic model + `RAGAgent` class. `run()` orchestrates: retrieve → format context → LLM → parse citations → return typed response. Includes `_format_context()`, `_parse_citations()`, `_extract_claim_context()`. |
| `__init__.py` | Lazy `__getattr__` imports — same pattern as other packages; defers LLM SDK imports until first use. |

### Scripts

| Script / Target | Purpose |
|---|---|
| `scripts/query.py` | CLI entry point. Builds Retriever + LLM, runs RAGAgent, pretty-prints answer + numbered evidence + citations + metadata. |
| `make query q="..."` | Hybrid RAG (default) |
| `make query-bm25 q="..."` | BM25-only (no embedder, fastest) |
| `make query-dense q="..."` | Dense-only (Qdrant, no BM25) |
| `make query-groq q="..."` | Pins LLM to Groq Llama 3.3 70B |

### Tests (43 tests across 2 files)

| File | Count | Coverage |
|---|---|---|
| `test_llm.py` | 12 | `LLMProvider` enum, `_build_single` model/params, `get_llm` fallback chain, Cerebras inclusion/exclusion, temperature/max_tokens forwarding |
| `test_rag_agent.py` | 31 | `_format_context` (7), `_parse_citations` (9), `RAGAgent.run` (13), `RAGResponse.source_tickers` (2) |

All 43 tests pass. No API calls, no network, no vector DB — everything mocked.

---

## How to Run

```bash
# 1. Ensure Phase 1 + 2 are complete (filings indexed in Qdrant + BM25)
make up-infra
make ingest ticker=NVDA
make index ticker=NVDA

# 2. Query
make query q="How is NVDA's data center revenue evolving?"

# 3. Ablation variants
make query-bm25  q="AMD vs NVDA GPU margins"
make query-dense q="MSFT cloud risk factors"
make query-groq  q="Apple Services segment growth"

# 4. With ticker filter
uv run python scripts/query.py "Revenue concentration risk" --tickers NVDA AMD

# 5. Run tests
make test
```

---

## Decision Log

### 1. Why a `with_fallbacks()` chain instead of manual try/except?

**Short answer:** LangChain's `with_fallbacks()` catches any `Exception` transparently — rate-limit errors, network timeouts, auth failures — and retries on the next provider. One line of wiring replaces a hand-rolled retry decorator.

**Deeper:** Free-tier LLMs throttle unpredictably. During eval runs (Phase 8), Gemini's 15 RPM cap can trigger mid-batch. Without a fallback chain, you'd either add `tenacity` retry loops to every LLM call or catch `RateLimitError` at the call site. `with_fallbacks()` is provider-agnostic — it doesn't know *why* the primary failed, which means it also recovers from transient network issues and auth hiccups. It wraps any `BaseChatModel`, so when Cerebras gets upgraded to a real LangChain integration (currently via `ChatOpenAI` with a custom base URL), swapping it out is a single line.

**Trade-off:** `with_fallbacks()` masks errors by default — a misconfigured API key on the primary looks like a normal fallback. Mitigation: structured logging in `_build_single()` logs which provider is actually used so Langfuse traces show the real model.

**Interview answer:** "We use LangChain's `with_fallbacks()` to chain Gemini → Groq → Cerebras. All are free-tier with no credit card required. The chain catches any exception — rate limits, network errors, auth failures — and transparently retries on the next provider. During eval runs this matters because Gemini's 15 RPM free limit can trigger mid-batch. The trade-off is that a misconfigured key looks like a normal fallback, so we log which model is actually used in every Langfuse trace."

---

### 2. Why regex citation parsing rather than structured JSON output?

**Short answer:** Asking the LLM to emit JSON (`{"citations": [1, 3]}`) adds 50–100 extra tokens per response and introduces a parsing failure mode; regex on `[N]` markers in natural prose is more robust at this scale.

**Deeper:** The RAG prompt asks the LLM to cite inline like "Revenue grew 217% [1]". The regex `\[(\d+)\]` is trivial, never fails on well-formed answers, and preserves the human-readable prose. Structured JSON output requires either: (a) `response_format={"type": "json_object"}` — which Gemini supports but Groq's free tier may not in all model versions, introducing a provider divergence — or (b) output parsers with retry-on-parse-fail logic, which is exactly the complexity we're avoiding at Phase 3.

The Critic agent (Phase 5) needs the `claim_text` alongside each citation for NLI verification. We extract that by grabbing ±75 characters around each `[N]` marker — a good-enough sentence approximation that doesn't require sentence tokenization.

**Trade-off:** If the LLM writes `[1]` in a different context (e.g., a footnote unrelated to evidence), it gets misinterpreted as a citation. The system prompt explicitly instructs the model to use `[N]` only for evidence citations, making false positives rare. False positives are also harmless — they produce a `Citation` object pointing to an evidence item, which the Critic will verify anyway.

**Interview answer:** "We use regex rather than structured JSON output for citation parsing. The prompt instructs the LLM to write `[N]` markers inline, and `re.findall(r'\[(\d+)\]', answer)` handles it. Structured JSON would need consistent support across Gemini, Groq, and Cerebras, and adds a parse-failure mode. The trade-off is that `[N]` in non-citation contexts (rare with prompt instructions) is misinterpreted — but the Critic verifies every citation anyway, so false positives are caught."

---

### 3. Why is `RAGResponse` a Pydantic model rather than a plain dict?

**Short answer:** The FastAPI handler (Phase 6), the Critic agent (Phase 5), and the eval harness (Phase 8) all receive `RAGResponse` objects. Typed models make field access checked by mypy and documented by the schema — no `response["evidence"]` KeyErrors.

**Deeper:** A plain dict is fine for a one-off script. For a system where a response flows through four layers — agent → API handler → WebSocket serialiser → eval harness — typing is load-bearing. Pydantic also gives free JSON serialisation (`response.model_dump_json()`), which the FastAPI handler will use directly when streaming citations to the frontend. The `source_tickers` property is a computed field that would be boilerplate to reconstruct from a dict.

`RAGResponse` inherits from `BaseModel` (not `AgentState` from `mia_shared`) because it's an *output* schema, not a stateful graph node. `AgentState` carries mutable in-flight data; `RAGResponse` is immutable after creation.

---

### 4. Why `temperature=0.0` as default?

**Short answer:** Financial extraction tasks need deterministic output — reproducible numbers, no paraphrasing of quoted figures.

**Deeper:** Temperature controls randomness in LLM sampling. For factual extraction from evidence ("what is the exact revenue figure?"), we want the same answer every time — both for consistency with users and because the eval harness compares answers across runs. Higher temperatures make benchmark results non-comparable across identical runs without statistical averaging. For the Summarizer agent (Phase 4) which writes narrative prose, we may raise temperature to 0.3 for more readable output. But the RAG baseline defaults to 0.0.

**Trade-off:** Temperature 0.0 doesn't mean fully deterministic — sampling implementations have floating-point non-determinism across hardware. But it's deterministic enough for evals to be meaningful without statistical smoothing.
