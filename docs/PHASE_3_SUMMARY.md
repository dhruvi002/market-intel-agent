# Phase 3 — Design Decisions & Interview Prep

> **Use this doc to:** challenge design decisions, prep interview answers, and restore context in a new Claude session.
> **Phase:** 3 — Single-Agent RAG Baseline (LLM factory, RAG prompt, RAGAgent, citation parsing)
> **Status:** ✅ Complete — 43/43 tests green
> **Project:** Autonomous Enterprise Market Intelligence Agent
> **Repo:** `/Users/dhruvishah/Documents/Projects/MarketIntelAgent/`

---

## What Phase 3 Built

Every file with its purpose — know this cold, you'll be asked "walk me through what you built."

| File | What it does |
|---|---|
| `packages/agents/src/mia_agents/llm.py` | `LLMProvider` enum (gemini / groq / cerebras) + `get_llm()` factory. Returns a LangChain `BaseChatModel` — either pinned to one provider or a `with_fallbacks()` chain: Gemini 2.0 Flash → Groq Llama 3.3 70B → Cerebras Llama 3.3 70B. `_build_single()` instantiates each via its LangChain SDK with lazy imports. |
| `packages/agents/src/mia_agents/prompts.py` | `RAG_PROMPT` (`ChatPromptTemplate`) and `build_rag_messages()`. System prompt enforces inline `[N]` citation markers, bans fabrication of financial figures, requests precise quotes of numbers/dates from evidence. |
| `packages/agents/src/mia_agents/rag_agent.py` | `RAGResponse` Pydantic model + `RAGAgent` class. `run()` orchestrates: retrieve evidence → `_format_context()` (numbered chunks) → `build_rag_messages()` → `llm.ainvoke()` → `_parse_citations()` (regex `[N]` extraction) → return typed `RAGResponse` with answer, evidence, citations, model_used, latency_ms. |
| `packages/agents/src/mia_agents/__init__.py` | Lazy `__getattr__` imports — same pattern as other packages. LLM SDKs not imported until `get_llm()` is first called. |
| `scripts/query.py` | CLI entry point. Argparse (query, --tickers, --mode, --no-rerank, --top-k, --provider). Builds Retriever + LLM, runs RAGAgent, pretty-prints answer / numbered evidence / citations / latency. |

**Tests:** 43 unit tests across 2 files. No API calls, no Qdrant, no network — LLM and Retriever fully mocked.

| Test file | Count | Coverage |
|---|---|---|
| `test_llm.py` | 12 | `LLMProvider` enum, `_build_single` model names + params, `get_llm` fallback chain construction, Cerebras inclusion/exclusion, temperature + max_tokens forwarding, missing-key error |
| `test_rag_agent.py` | 31 | `_format_context` (7 cases), `_parse_citations` (9 cases), `RAGAgent.run` (13 cases: query forwarding, ticker/mode pass-through, citation mapping, latency, session ID uniqueness, model_used extraction, reranked flag), `RAGResponse.source_tickers` (2 cases) |

---

## How to Run

```bash
# Prerequisites: Phase 1 + 2 complete (filings ingested and indexed)
make up-infra
make ingest ticker=NVDA          # Phase 1
make index ticker=NVDA           # Phase 2

# Phase 3 queries
make query q="How is NVDA's data center revenue evolving?"
make query-bm25  q="AMD supply chain risk factors"
make query-dense q="Apple Services segment growth"
make query-groq  q="MSFT Azure cloud margin trends"

# With ticker filter + options
uv run python scripts/query.py "Revenue concentration risk" \
    --tickers NVDA AMD --mode hybrid --top-k 15

# Run all tests
make test
```

---

## Decision Log — The "Why" Behind Every Choice

Use the **3-part pattern** to answer design questions in interviews:
1. **One sentence** — the choice made
2. **Two sentences** — the problem it solves + concrete example
3. **One sentence** — the trade-off you accepted

---

### 1. Why `with_fallbacks()` instead of manual try/except for the LLM chain?

**Short answer:** LangChain's `with_fallbacks()` catches any `Exception` transparently — rate-limit errors, network timeouts, auth failures — and retries on the next provider. One line of wiring replaces a hand-rolled retry decorator.

**Deeper:** Free-tier LLMs throttle unpredictably. During eval runs (Phase 8), Gemini's 15 RPM cap can trigger mid-batch. Without a fallback chain, you'd add `tenacity` retry loops to every LLM call *and* catch provider-specific exceptions (`google.api_core.exceptions.ResourceExhausted`, `groq.RateLimitError`) at each call site. `with_fallbacks()` is provider-agnostic — it doesn't inspect why the primary failed. This means it also silently recovers from a misconfigured API key on the primary, which is a downside: you might run an entire eval on Groq without realising Gemini never responded. Mitigation: log the actual model name from `response.response_metadata` in every RAGResponse so Langfuse traces reveal the real provider.

**Trade-off:** `with_fallbacks()` wraps the primary and all fallbacks into a single `BaseChatModel` — the caller has no visibility into which provider executed without checking `response.response_metadata`. For most use cases this is fine; for cost attribution per provider, you need the metadata inspection.

**Interview answer:** "We use LangChain's `with_fallbacks()` to chain Gemini → Groq → Cerebras. All free-tier, no CC. It catches any exception — rate limits, network errors, auth failures — and transparently retries on the next provider. Gemini's 15 RPM free limit can trigger mid-eval run, so the fallback is load-bearing, not just defensive. The trade-off is that a misconfigured Gemini key silently falls through to Groq — we log which model actually responded in every trace so Langfuse shows the real provider."

---

### 2. Why regex citation parsing (`\[(\d+)\]`) rather than structured JSON output?

**Short answer:** Asking the LLM to emit JSON adds ~50–100 tokens per response and introduces a parse-failure mode; regex on `[N]` markers in natural prose is more robust and provider-agnostic.

**Deeper:** Structured JSON output requires either `response_format={"type":"json_object"}` — which Gemini supports but Groq's free-tier Llama may behave inconsistently on — or output parsers with `retry_on_parse_fail` logic. Both introduce a provider divergence: your prompt works perfectly on Gemini but breaks on the Groq fallback because the smaller model doesn't reliably emit valid JSON under the same instruction. The `[N]` convention sidesteps this: it's a format instruction baked into the system prompt ("cite inline like [1], [2]"), and every model in the fallback chain obeys it. The regex `\[(\d+)\]` is three characters and never fails.

The Critic agent (Phase 5) needs `claim_text` alongside each citation for NLI verification. We extract ±75 characters around each `[N]` match as a sentence approximation — good enough for the cross-encoder NLI model without needing `nltk.sent_tokenize`.

**Trade-off:** If the LLM writes `[1]` in a context unrelated to evidence (footnote numbering, list notation), it gets misinterpreted as a citation. The system prompt mitigates this by instructing the model to use `[N]` *only* for evidence citations. False positives still produce a `Citation` object pointing to an evidence item — the Critic verifies every citation anyway, so false positives are caught at the next stage.

**Interview answer:** "We use regex rather than JSON output for citation parsing. The prompt instructs the LLM to write `[N]` markers inline; `re.findall(r'\[(\d+)\]', answer)` extracts them. Structured JSON would need consistent support across Gemini, Groq, and Cerebras — the Llama models in particular can be inconsistent with format instructions under rate pressure. The trade-off is that `[N]` in non-citation contexts is misinterpreted, but the Critic verifies every citation anyway."

---

### 3. Why is `RAGResponse` a Pydantic model rather than a plain dict?

**Short answer:** Four downstream consumers — FastAPI handler (Phase 6), Critic agent (Phase 5), WebSocket serialiser, and eval harness (Phase 8) — all receive `RAGResponse`. Typed models give mypy checking, auto-serialisation, and self-documenting fields.

**Deeper:** A plain dict is fine for a single-layer script. For a system where a response flows through four processing stages, typing is load-bearing: a typo on `response["evdience"]` is a silent `KeyError` at runtime; `response.evidence` fails at import-time mypy check. Pydantic also gives `model_dump_json()` for free — the FastAPI handler will call this directly when streaming the response to the frontend as a WebSocket JSON frame.

`RAGResponse` is separate from `AgentState` (in `mia_shared.schemas`) by design. `AgentState` is a *mutable, in-flight* graph node — it carries partial work-in-progress data through LangGraph's StateGraph. `RAGResponse` is an *immutable output* — it represents a completed single-agent run. Mixing them would make Phase 4's graph nodes responsible for emitting API-shaped output, coupling the orchestration layer to the API layer.

The `source_tickers` property (`sorted({ev.ticker for ev in self.evidence if ev.ticker})`) is a convenience accessor used by the frontend and eval harness. It's three lines — trivial to reconstruct from a dict, but having it on the model means callers don't repeat the set comprehension.

**Interview answer:** "RAGResponse is a Pydantic model because four consumers receive it downstream — the FastAPI handler, the Critic, the WebSocket serialiser, and the eval harness. Typed models give mypy checking on field access and free JSON serialisation via `model_dump_json()`. I kept it separate from `AgentState` — which is the mutable in-flight LangGraph node — because RAGResponse is an immutable output, and mixing them would couple the orchestration layer to the API response format."

---

### 4. Why `temperature=0.0` as the default?

**Short answer:** Financial extraction is a deterministic task — reproducible numbers, no paraphrasing of quoted figures, stable evals across multiple runs.

**Deeper:** Temperature controls the entropy of the sampling distribution. At 0.0, the model (mostly) samples the highest-probability token at each step — equivalent to greedy decoding. For financial RAG, you want the same revenue figure every time: "NVDA data center revenue was $18.4B in FY2024 [1]" should not vary run to run. If it did, eval results would be non-comparable across identical runs without statistical averaging over 3–5 samples per query, tripling your LLM cost.

The caveat: "temperature 0.0" is not perfectly deterministic due to floating-point non-determinism in GPU matrix multiplications. Different hardware may produce different results. But in practice, the variation is in low-probability completions (trailing words), not in high-salience numerical extractions.

For Phase 4 agents that write narrative prose (Summarizer), we'll likely raise temperature to 0.2–0.3 for more natural writing. The RAG baseline defaults to 0.0 and lets callers override.

**Interview answer:** "Temperature 0.0 because financial RAG is an extraction task — we want the same revenue figure every run for consistent evals. Higher temperatures make benchmark results non-comparable without averaging 3+ samples per query, tripling LLM cost. The caveat is that 0.0 isn't perfectly deterministic due to GPU floating-point non-determinism, but it's stable enough for our purposes. The Summarizer in Phase 4 will likely use 0.2 for more natural prose."

---

### 5. Why format context as numbered chunks `[1] TICKER FORM_TYPE — Section` rather than raw text?

**Short answer:** The citation system requires the LLM to know which evidence item is which. Numbered headers make the mapping unambiguous; TICKER + FORM_TYPE + Section tell the model the provenance without it having to read the full text.

**Deeper:** Without numbered headers, the LLM has no reliable way to cite evidence. You could ask it to quote the source URL, but the evidence chunks don't have canonical URLs yet (those come from Phase 1's MinIO storage path). You could ask it to cite by ticker + accession number, but that's fragile (LLMs sometimes invent accession numbers). A simple `[1]`, `[2]` sequence is unambiguous, machine-parseable, and convention the model already knows from academic citation notation.

The `TICKER FORM_TYPE — Section` header serves a second purpose: it focuses the model's attention. When the LLM sees `[3] AMD 10-K — Risk Factors`, it primes itself to interpret the following text as a risk disclosure from an AMD annual report, rather than as financial projections or MD&A commentary. This reduces cross-chunk confusion when multiple filings have similar language.

The empty-evidence fallback `"(no evidence retrieved)"` is also explicit rather than an empty string. An empty string could cause the model to hallucinate ("no evidence was provided so I'll answer from my training knowledge"), while the explicit placeholder signals "the retriever found nothing" which the model is instructed to acknowledge.

**Interview answer:** "We format context as numbered chunks — `[1] NVDA 10-K — MD&A` followed by the text. The number is what the citation regex parses; the TICKER + FORM + SECTION header tells the model the provenance upfront so it doesn't have to infer it from the text. Without numbered headers the citation system breaks: the LLM has no stable way to reference a specific chunk. The section name also primes attention — `Risk Factors` text is interpreted differently from `MD&A` text even if the words overlap."

---

### 6. Why separate `_format_context()` and `_parse_citations()` as module-level functions rather than methods?

**Short answer:** They're pure functions with no state — taking them out of the class makes them independently testable and reusable by other agents.

**Deeper:** `_format_context(evidence: list[Evidence]) -> str` and `_parse_citations(answer: str, evidence: list[Evidence]) -> list[Citation]` have no dependency on `self`. Making them methods would mean test code has to instantiate a full `RAGAgent` (with a mocked Retriever and LLM) just to test string formatting. As module-level functions, the tests import them directly and test with plain Python objects — no mocking required. 7 of the 31 `test_rag_agent.py` tests target `_format_context` and `_parse_citations` directly; they run in milliseconds with zero setup.

This also makes them available to Phase 4 nodes. The Summarizer agent needs to format evidence into a prompt; the Critic needs to parse citations from the Summarizer's output. Both can import `_format_context` and `_parse_citations` from `mia_agents.rag_agent` without coupling to `RAGAgent`.

**Interview answer:** "They're module-level functions because they're pure — no `self` dependency. This makes them directly importable and testable without instantiating a RAGAgent. It also means Phase 4 nodes (Summarizer, Critic) can reuse them without coupling to RAGAgent. Seven of the tests target these functions directly and run in under 10ms with zero mocking setup."

---

### 7. Why lazy-import the LLM SDKs inside `_build_single()`?

**Short answer:** Importing `langchain_google_genai` at module load time imports the Google Cloud SDK transitively — several hundred milliseconds and ~50MB of imports that are wasted if you're running BM25-only tests.

**Deeper:** `langchain_google_genai` transitively imports `google-auth`, `google-api-core`, `protobuf`, and `grpcio`. `langchain_groq` imports the Groq SDK. Neither is needed until an LLM is actually instantiated. In a test suite that mocks `_build_single`, those SDKs never load. In a script that only uses `make retrieve` (Phase 2 retrieval, no LLM), importing `mia_agents` would still pull in all LLM SDKs if they were top-level imports — adding 200–400ms to startup time.

This follows the same pattern as `mia_retrieval`: `get_embedder()` and `get_reranker()` do `from sentence_transformers import SentenceTransformer` inside the function, avoiding the 1GB torch import until the model is actually needed.

**Interview answer:** "LLM SDK imports are deferred to `_build_single()` via local imports. `langchain_google_genai` transitively imports google-auth, protobuf, and grpcio — hundreds of milliseconds of startup for code that might never run if you're using BM25-only mode. The pattern is consistent with the retrieval package: all heavy model imports (sentence-transformers, torch) are deferred to factory functions so the package imports light and tests run fast."

---

### 8. Why deduplicate citations by `evidence_id` rather than by `[N]` marker?

**Short answer:** The LLM often writes `[1][1]` or `[1] ... [1]` when a claim is strongly supported — deduplicating by marker index would still give duplicates if the same evidence appears at `[1]` in two separate places.

**Deeper:** Consider an answer like: "NVDA data center revenue was $18.4B [1]. This represents 78% of total revenue [1][3]. The growth rate was 217% [1]." The marker `[1]` appears three times. Deduplicating by marker index (deduplicate on the `int` value) would give one citation for `[1]` and one for `[3]`. Deduplicating by `evidence_id` does the same thing. But consider a case where the same evidence item appears in two positions — e.g., evidence[0] and evidence[2] are identical (duplicate retrieval). The LLM might write `[1]` and `[3]`, both pointing to the same filing chunk. Deduplicating by `evidence_id` catches this; deduplicating by marker integer doesn't.

In practice, the retriever deduplicates by chunk ID in the RRF fusion step (Phase 2), so the same chunk shouldn't appear twice. But `evidence_id` dedup is the logically correct invariant: "one Citation per unique source", not "one Citation per unique marker integer".

**Interview answer:** "We deduplicate by `evidence_id` — the UUID of the Evidence object — rather than by the `[N]` integer. The correct invariant is 'one Citation per unique source', not 'one Citation per unique marker'. The LLM might write `[1]` three times for the same chunk; deduplicating by marker integer would still give one Citation, which is correct. But if the same source appeared at two different indices — which shouldn't happen given RRF's dedup, but could in edge cases — dedup by marker integer would miss it."

---

## Gotchas from Real Implementation

**1. `response.content` not `response.text`**
LangChain's `BaseChatModel.ainvoke()` returns an `AIMessage` object. The text content is `response.content`, not `response.text`. The `response_metadata` dict (containing model name) is `response.response_metadata`. This trips up developers used to the OpenAI SDK's `response.choices[0].message.content`.

**2. `with_fallbacks()` doesn't retry — it falls through on first exception**
If the primary LLM raises `RateLimitError`, `with_fallbacks()` immediately tries Groq. It does NOT retry the primary. If you want exponential backoff on the primary before falling through, wrap the primary in `tenacity.retry` before passing to `with_fallbacks()`. Phase 3 does not do this — accepted trade-off for simplicity.

**3. LangChain `ChatGoogleGenerativeAI` uses `max_output_tokens`, not `max_tokens`**
`ChatGroq` and `ChatOpenAI` use `max_tokens`. `ChatGoogleGenerativeAI` uses `max_output_tokens`. If you pass `max_tokens` to the Google model, it silently ignores it. The `_build_single()` function handles this per-provider.

**4. Empty context does not mean LLM stays silent**
If `evidence=[]`, `_format_context()` returns `"(no evidence retrieved)"`. Without this explicit placeholder, the LLM receives an empty context and may answer from its training weights — exactly the hallucination risk we're building this system to prevent. Always pass an explicit "no evidence" signal.

**5. Citation index is 1-based in the prompt, 0-based in the list**
The formatted context labels chunks `[1]`, `[2]`, etc. (1-based, matching human-readable convention). `_parse_citations()` converts: `idx = int(match.group(1)) - 1`. Off-by-one here produces silent wrong citations — the test `test_parse_citations_zero_index_ignored` specifically checks that `[0]` is discarded.

**6. `asyncio.run()` in tests is fine for Python 3.10+ without `pytest-asyncio`**
All `RAGAgent.run()` tests use `asyncio.run(agent.run(...))` rather than `pytest.mark.asyncio`. This avoids `pytest-asyncio` version compatibility issues and is perfectly valid — `asyncio.run()` creates a fresh event loop per test, which is what you want. The project's `pyproject.toml` sets `asyncio_mode = "auto"` for the parts that use `pytest.mark.asyncio`; the agent tests bypass it entirely.

**7. `_extract_model_name()` handles both Gemini and Groq metadata formats**
Gemini puts the model name in `response_metadata["model_name"]`; Groq puts it in `response_metadata["model"]`. The helper function checks both keys with `or` — if neither exists (mocked LLM in tests), it returns `""` rather than raising `KeyError`.

**8. `build_retriever()` from Phase 2 is the standard factory — don't import `Retriever` directly**
`build_retriever(bm25_path=...)` loads the BM25 index from disk, creates the Qdrant client, and instantiates the Embedder/Reranker factories. Importing `Retriever` directly and constructing it requires instantiating all four components manually. The `scripts/query.py` CLI uses `build_retriever()` correctly; tests mock the `Retriever` directly since they don't need real components.

---

## Questions to Be Ready For

Use the pattern: **choice → problem it solves + example → trade-off**.

---

**"Walk me through the Phase 3 pipeline end to end."**
> "A query comes in — say, 'How is NVDA's data center revenue growing?' The RAGAgent first calls the Phase 2 Retriever in hybrid mode: BM25 + dense search → RRF fusion → bge-reranker → top-10 Evidence objects. Then it formats those chunks as a numbered list: `[1] NVDA 10-K — MD&A\n<text>`, `[2] NVDA 10-Q — MD&A\n<text>`, etc. That numbered list becomes the `{context}` in the RAG prompt. The LLM — Gemini 2.0 Flash by default — generates an answer with inline `[N]` citation markers. A regex extracts every `[N]` marker, maps the 1-based index to the Evidence list, and builds Citation objects with the surrounding sentence as `claim_text`. The result is a `RAGResponse` with the answer, the evidence list, and the citations — all typed Pydantic objects ready for the downstream Critic or API layer."

---

**"What's the difference between `retrieve.py` (Phase 2) and `query.py` (Phase 3)?"**
> "`make retrieve` tests the retrieval stack in isolation — it returns raw chunks with scores, no LLM involved. Useful for debugging why a specific chunk didn't surface. `make query` runs the full RAG pipeline: retrieval plus LLM synthesis plus citation parsing. The output is a natural-language answer with inline citations, not a ranked list of chunks. Phase 2 is the information retrieval component; Phase 3 is the generation component that consumes it."

---

**"Why not just prompt the LLM with the question and let it answer from its training data?"**
> "Two failure modes: first, hallucination — LLMs confidently invent financial figures. NVDA's FY2024 data center revenue was $47.5B; an LLM might say $30B or $60B depending on what fragment it's blending. Second, staleness — training cutoffs are months to over a year behind. An analyst asking about Q3 2025 earnings gets no useful answer from training data. RAG grounds every claim in retrieved, timestamped evidence and forces citation. The Critic (Phase 5) then verifies citations against the evidence using NLI scoring, giving a second check on hallucination."

---

**"How does the fallback chain affect latency?"**
> "The fallback chain only kicks in on failure — normal path hits Gemini and returns in 500ms–2s depending on response length. If Gemini throws `RateLimitError`, the SDK raises immediately (no wait), `with_fallbacks()` routes to Groq, and Groq returns in ~300ms (it's the fastest inference endpoint we use at ~800 tok/s). In the worst case — Gemini rate-limited, Groq also rate-limited — we fall through to Cerebras at ~2000 tok/s. Total added latency on a fallback is one failed HTTP request (~100ms) plus the second provider's response time. For a RAG pipeline that already takes 500ms+ for retrieval + reranking, that's acceptable."

---

**"How do you know which LLM actually ran?"**
> "`RAGResponse.model_used` is extracted from `response.response_metadata` — Gemini puts it under `model_name`, Groq puts it under `model`. This gets logged by Langfuse (Phase 8) with every trace, so you can see per-query which provider handled it and whether a fallback occurred. Without this, you'd only see that `with_fallbacks()` returned a result, with no visibility into which branch ran."

---

**"What happens if the LLM cites `[99]` when you only have 10 evidence chunks?"**
> "`_parse_citations()` bounds-checks: `if idx < 0 or idx >= len(evidence): continue`. Out-of-range citations are silently dropped. The test `test_parse_citations_out_of_range` covers exactly this. It happens occasionally when the LLM hallucinates a citation — writes `[11]` in an answer backed by 10 chunks. The citation is dropped rather than raising an error; the Critic will flag the uncited claim when it checks all factual assertions against the evidence."

---

**"How will RAGAgent fit into Phase 4's multi-agent graph?"**
> "In Phase 4, the LangGraph StateGraph has a Retrieval worker node. That node will call `RAGAgent.run()` or directly call `Retriever.retrieve()` and attach the evidence to `AgentState.evidence`. The Summarizer worker then calls the LLM with the accumulated evidence from all workers (web search + EDGAR + RAG), not just from the RAG node. The Phase 3 baseline is the Retrieval worker in isolation — useful for evaluating retrieval quality independently of the multi-agent overhead."

---

**"Why is the RAG prompt in a `ChatPromptTemplate` rather than an f-string?"**
> "Three reasons. First, `ChatPromptTemplate` produces `HumanMessage`/`SystemMessage` objects that the LangChain chat model APIs expect natively — no manual message construction. Second, it separates the system prompt (role instruction, citation rules) from the human turn (query + context) structurally, which matters for models that treat system vs human messages differently (Gemini and Claude use system prompts very differently from the human turn). Third, it's composable — in Phase 4, we can swap in a different template for the Summarizer or Critic node without changing the agent logic."

---

**"What does `source_tickers` give you and why is it a property?"**
> "`RAGResponse.source_tickers` returns `sorted({ev.ticker for ev in self.evidence if ev.ticker})` — the unique tickers that actually contributed to the answer, alphabetically sorted. It's a computed property rather than a stored field because it's always derivable from `evidence`; storing it separately would create a consistency hazard if `evidence` is modified. The frontend uses it to display which companies were sourced, and the eval harness uses it to check whether the correct ticker's documents were retrieved for a given query."

---

## Connections to Other Phases

| This decision... | ...is load-bearing for Phase... |
|---|---|
| `RAGResponse` Pydantic model with `evidence: list[Evidence]` | Phase 4: `AgentState.evidence` populated by the Retrieval node from `RAGResponse.evidence` |
| `_parse_citations()` returns `Citation(claim_text=..., is_verified=False)` | Phase 5: Critic receives citations with `is_verified=False` and sets them to `True` after NLI scoring |
| `_format_context(evidence)` as a module-level function | Phase 4+5: Summarizer and Critic nodes reuse it to format their own evidence prompts |
| `RAGResponse.model_used` from `response_metadata` | Phase 8: Langfuse traces use this to report actual provider per query, enabling per-provider latency/cost analysis |
| Fallback chain (Gemini → Groq → Cerebras) | Phase 8: Ablation evals run many queries in sequence — fallback chain prevents a single rate-limit from breaking the entire eval run |
| `temperature=0.0` | Phase 8: Deterministic output needed for reproducible eval scores — same question must produce comparable answers across ablation cells |

---

## Red Flags to Avoid

- Don't say "we just prompt the LLM and ask it to cite" — explain what `[N]` markers are, how they map to the evidence list, and how `_parse_citations()` extracts them with a regex
- Don't confuse `RAGResponse` with `AgentState` — `RAGResponse` is an immutable output, `AgentState` is the mutable in-flight LangGraph state
- Don't say the fallback chain "retries" — it falls *through* to the next provider on first failure, it does not retry the same provider
- Don't say temperature 0.0 is "fully deterministic" — acknowledge GPU floating-point non-determinism; say it's "stable enough for reproducible evals"
- Don't say "we use the OpenAI library for Cerebras" — we use `langchain_openai.ChatOpenAI` with `openai_api_base` pointed at Cerebras's OpenAI-compatible endpoint. The library is OpenAI's; the endpoint is Cerebras's
- Don't skip why `evidence=[]` produces `"(no evidence retrieved)"` — the explicit placeholder prevents the LLM from silently hallucinating answers from training data
- Don't say Phase 3 "calls the LLM directly" — it calls it via a LangChain `BaseChatModel`, which abstracts the provider. The `with_fallbacks()` chain only works because every provider returns the same `BaseChatModel` interface
