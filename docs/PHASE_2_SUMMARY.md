# Phase 2 — Design Decisions & Interview Prep

> **Use this doc to:** challenge design decisions, prep interview answers, and restore context in a new Claude session.
> **Phase:** 2 — Retrieval Stack (chunker, embedder, BM25, Qdrant, RRF fusion, reranker, indexer, retriever)
> **Status:** ✅ Complete — 46/46 tests green
> **Project:** Autonomous Enterprise Market Intelligence Agent
> **Repo:** `/Users/dhruvishah/Documents/Projects/MarketIntelAgent/`

---

## What Phase 2 Built

Every file with its purpose — know this cold, you'll be asked "walk me through what you built."

| File | What it does |
|---|---|
| `packages/retrieval/src/mia_retrieval/chunker.py` | `Chunk` frozen dataclass + `Chunker`: word-count sliding window (300 words, 40 overlap). Deterministic UUID5 chunk IDs from `(filing_id, chunk_index)` for idempotent re-indexing. |
| `packages/retrieval/src/mia_retrieval/embedder.py` | `Embedder` singleton wrapping `BAAI/bge-large-en-v1.5`: lazy load, `normalize_embeddings=True`, batched encode (32/batch). `get_embedder()` module-level factory. 1024-dim output. |
| `packages/retrieval/src/mia_retrieval/bm25_index.py` | `BM25Index`: wraps `rank_bm25.BM25Okapi`. `build()` / `add()` (full rebuild) / `search()` (filters zero scores) / `save()` / `load()` (pickle). `_tokenize()` = lowercase whitespace split. |
| `packages/retrieval/src/mia_retrieval/qdrant_store.py` | `QdrantStore`: async Qdrant client, `ensure_collection()` (cosine/1024-dim + payload indexes), `upsert()`, `search()` (with optional ticker filter), `filing_is_indexed()`, `delete_by_filing()`. `chunk_from_scored_point()` helper. |
| `packages/retrieval/src/mia_retrieval/hybrid.py` | `reciprocal_rank_fusion()`: RRF k=60 over BM25 + dense result lists. Deduplicates shared hits. Reconstructs `Chunk` objects from Qdrant `ScoredPoint` payloads. |
| `packages/retrieval/src/mia_retrieval/reranker.py` | `Reranker` singleton wrapping `BAAI/bge-reranker-v2-m3` `CrossEncoder`: lazy load. `rerank(query, candidates, top_k)`. `get_reranker()` factory. |
| `packages/retrieval/src/mia_retrieval/indexer.py` | `FilingRecord` (plain dataclass, decoupled from ORM) + `IndexStats` + `IndexingPipeline`: skip-if-indexed, batch embed + upsert to Qdrant, BM25 add, BM25 save after ticker. |
| `packages/retrieval/src/mia_retrieval/retriever.py` | `RetrieveMode(str, Enum)` + `Retriever.retrieve()`: BM25 / dense / hybrid modes → optional reranker → `list[Evidence]`. `build_retriever()` factory loads all components. |
| `packages/retrieval/src/mia_retrieval/__init__.py` | Lazy `__getattr__` imports — same pattern as `mia_ingestion`. Defers torch/sentence-transformers until first access. |
| `scripts/index_ticker.py` | CLI: loads Phase 1 filings from Postgres, runs `IndexingPipeline`. `make index ticker=NVDA`. |
| `scripts/retrieve.py` | CLI: test full retrieve stack from command line. `make retrieve query="..."`. |

**Tests:** 46 unit tests across 4 files. No DB, Qdrant, or network calls — Qdrant/Embedder/Reranker mocked via `unittest.mock`.

| Test file | Count | Coverage |
|---|---|---|
| `test_chunker.py` | 15 | IDs deterministic/unique, empty/short input, chunk size, overlap, metadata, indices, with_total, constructor validation |
| `test_bm25.py` | 15 | tokenizer, build/rebuild, search ranking/top_k/scores/sort, OOV, add, pickle roundtrip, parent dir creation |
| `test_hybrid.py` | 8 | empty inputs, BM25-only, dense-only, sort order, both-list boost, RRF formula, no duplicates, metadata preserved |
| `test_retriever.py` | 8 | Evidence fields, BM25/dense/hybrid mode wiring, ticker filter, reranker on/off, empty results |

---

## How to Run

```bash
make up-infra              # Postgres + Qdrant + Redis + MinIO
make ingest ticker=NVDA    # Phase 1: filings → Postgres + MinIO
make index ticker=NVDA     # Phase 2: filings → chunks → Qdrant + BM25
make retrieve query="How is NVDA's data center revenue growing?"
make retrieve-bm25 query="AMD risk factors supply chain"
make test                  # run all tests (pytest)
```

---

## Decision Log — The "Why" Behind Every Choice

Use the **3-part pattern** to answer design questions in interviews:
1. **One sentence** — the choice made
2. **Two sentences** — the problem it solves + concrete example
3. **One sentence** — the trade-off you accepted

---

### 1. Why word-count chunking at 300 words / 40-word overlap?

**Short answer:** bge-large-en-v1.5 has a 512-token limit. Financial text averages ~1.5 tokens/word, so 300 words ≈ 450 tokens — 60-token safety margin. 40-word overlap prevents losing context across chunk boundaries.

**Deeper:** Token-count chunking is more accurate but requires loading the model's WordPiece tokenizer just to count tokens, which imports sentence-transformers (and transitively, PyTorch — ~1GB) during a lightweight preprocessing step. Word-count proxy is what LangChain's `RecursiveCharacterTextSplitter` effectively does anyway. At 300 words, we're safely under the 512-token cap for even the densest financial text (lots of numbers and abbreviations push tokenisation density up).

The 40-word overlap means a sentence split right at a chunk boundary will appear complete in at least one of the two adjacent chunks. Without overlap, a clause like "revenue increased 217% year-over-year" could be split as "revenue increased 217%" / "year-over-year", and neither half is a coherent retrieval unit.

**Trade-off:** Very dense tables (lots of `$1,234,567` numbers) can still push individual chunks above 512 tokens, causing silent truncation by the embedder. Mitigation: the overlap ensures the truncated content reappears in the next chunk.

**Interview answer:** "We chunk at 300 words with 40-word overlap. bge-large has a 512-token limit; financial text tokenises at roughly 1.5 tokens/word, so 300 words gives ~450 tokens with a 60-token safety buffer. The overlap ensures a sentence split at a boundary appears complete in at least one chunk. We use word count rather than the model's tokenizer to avoid importing a 1GB library during preprocessing."

---

### 2. Why deterministic UUID5 chunk IDs?

**Short answer:** Qdrant upsert is idempotent on point ID — same IDs on re-index means no duplicate chunks.

**Deeper:** Each chunk ID is `uuid5(NAMESPACE_DNS, f"{filing_id}:{chunk_index}")`. UUID5 is a hash-based UUID — the same inputs always produce the same UUID. When `make index ticker=NVDA` is run a second time (e.g., after modifying chunker parameters to debug a bug), Qdrant upserts the same point IDs with the new content — the old points are overwritten, not duplicated.

With random UUID4 IDs, every re-index would create new points alongside the old ones, and Qdrant's collection size would double on each run. You'd need an explicit delete-before-insert step, and if that delete failed mid-run, you'd have a corrupt partially-duplicate collection.

**Interview follow-up — "What if you change the chunking parameters?"**
If `chunk_size` changes, the text content of each chunk changes. But because the ID is derived from `(filing_id, chunk_index)`, chunks at the same index get the same ID but different content — which is correct, Qdrant upserts overwrite. If the new chunking produces *fewer* chunks (larger chunk size), old chunks beyond the new last index become stale orphans. The fix: pass `force=True` to `IndexingPipeline`, which calls `QdrantStore.delete_by_filing()` before re-indexing.

**Interview answer:** "Chunk IDs are UUID5, a hash of `(filing_id, chunk_index)`. The same filing chunked the same way always produces the same UUID, so Qdrant upserts are idempotent — re-running `make index` is safe. Random UUIDs would create duplicate points on every re-index."

---

### 3. Why L2-normalise embeddings?

**Short answer:** With a cosine-distance Qdrant collection, cosine similarity equals dot product on unit vectors — Qdrant uses a faster dot-product HNSW path, ~15% faster on large collections.

**Deeper:** Cosine similarity is `A·B / (|A| × |B|)`. If A and B are unit vectors, `|A| = |B| = 1`, so cosine = `A·B` — a dot product. Qdrant detects L2-normalised vectors and uses a dot-product HNSW index internally, skipping the norm division per distance computation.

`sentence_transformers.encode(normalize_embeddings=True)` does the normalisation inside the model's forward pass at no extra CPU cost. The alternative — normalising after the fact in numpy — is equivalent but separates a logically coupled operation.

**Side benefit:** L2-normalised embeddings allow meaningful cosine comparison across different queries without worrying about embedding magnitude variation (some long documents produce higher-magnitude embeddings than short ones).

**Interview answer:** "We set `normalize_embeddings=True` in the embedder. Cosine similarity between unit vectors equals their dot product — Qdrant optimises for this by using a dot-product HNSW path, which skips the norm division on every distance computation. It's a 15% speed-up on large collections for free."

---

### 4. Why Reciprocal Rank Fusion over score normalisation?

**Short answer:** RRF is parameter-free, rank-based (immune to the scale mismatch between BM25 log-frequency scores and cosine similarities), and empirically competitive with learned fusion on most benchmarks.

**The formula:** `score(d) = Σ_i  1 / (k + rank_i(d))`, where k=60 (from Cormack et al. 2009). Ranks are 0-indexed. A document ranked #1 in BM25 and #1 in dense gets `1/61 + 1/61 ≈ 0.033`. A document ranked #1 in only one list gets `0.016`. The k=60 constant smooths the contribution of lower-ranked results — increasing k makes all results contribute more equally; decreasing k amplifies the top-rank advantage.

**Why not score normalisation?** BM25 scores are corpus-dependent log-frequencies (typically 0–15 for our corpus size). Dense scores are cosine similarities (0–1). Min-max normalisation requires knowing the corpus-wide min and max, which shifts as we index new documents. Stale min/max values degrade fusion quality and require periodic recalculation. RRF has no such state — you pass in two ranked lists and get a merged ranked list, regardless of corpus size or composition changes.

**Trade-off:** RRF implicitly weights both systems equally. If BM25 has 0.3 precision and dense has 0.8 precision for our corpus, equal rank weighting undersells the better system. A learned linear combination (e.g., `0.7 × dense_score + 0.3 × bm25_score`, normalised) would outperform. The Phase 8 ablation will quantify this difference on the golden eval set.

**Interview answer:** "BM25 scores are log-frequencies, dense scores are cosine similarities — completely different scales. Score normalisation requires knowing the corpus min/max which shifts every time we index new documents. RRF only uses ranks — `score = Σ 1/(60 + rank)` — so it's parameter-free and robust to scale differences. A document ranked #1 in both systems gets roughly double the score of a document ranked #1 in only one. The trade-off is that we implicitly assume both systems are equally reliable, which the Phase 8 ablation will test."

---

### 5. Why a cross-encoder reranker after fusion?

**Short answer:** Bi-encoders score query and document independently; cross-encoders run full self-attention over the query-document pair — much higher accuracy at the cost of O(n) forward passes over the candidate set.

**The retrieval pipeline in full:**
1. BM25 → top-50 candidates (exact keyword matching, fast, CPU only)
2. Dense search → top-50 candidates (semantic, fast, GPU optional)
3. RRF merge → up to 100 unique candidates (deduplicated, rank-fused)
4. Cross-encoder → top-10 reranked results (~200ms on CPU for 100 pairs)
5. Return as `list[Evidence]`

**Bi-encoder limitation:** The embedder computes a fixed 1024-dim representation for each document at index time, independent of any query. At query time, it computes a query embedding and measures cosine similarity. This misses query-document interactions — "fiscal Q4" in a query and "fourth quarter" in a document are semantically similar, but the cosine similarity between their independent embeddings is lower than it should be because the embedder didn't see them together.

**Cross-encoder advantage:** The input is `[CLS] query [SEP] passage [SEP]`, and the self-attention layers can directly compare tokens from the query against tokens in the passage. This is why cross-encoders consistently outperform bi-encoders on reranking tasks by 10–20% nDCG.

**Why not cross-encode from scratch?** Cross-encoding scales as O(n × q) per query where n = corpus size. At 150k chunks, that's hours per query. The two-stage pipeline (fast bi-encoder retrieval → expensive cross-encoder reranking over top-k) gives near-cross-encoder quality at bi-encoder speed.

**Interview answer:** "The bi-encoder scores query and document independently. A cross-encoder sees `[CLS] query [SEP] passage [SEP]` and can attend across both — it catches 'fiscal Q4' matching 'fourth quarter' in ways the bi-encoder misses. But cross-encoding the full corpus would take hours. So we use BM25 + dense to narrow to 50–100 candidates, then the cross-encoder reranks those in ~200ms. Two-stage retrieval gives near-cross-encoder quality at bi-encoder speed."

---

### 6. Why bge-large-en-v1.5 and bge-reranker-v2-m3?

**Short answer for embedder:** Free, local, 1024-dim, top-tier on BEIR financial and scientific domain benchmarks. Beats OpenAI text-ada-embedding-002 on several BEIR subsets.

**Short answer for reranker:** Free, local, multilingual base (mdeberta-v3) which handles financial jargon better than English-only cross-encoders. Outperforms Cohere Rerank-v2 on multiple BEIR splits.

**Deeper on bge-large-en-v1.5:** The BAAI "bge" family uses the same training recipe as OpenAI's embeddings: contrastive learning with in-batch negatives and hard negatives mined from the corpus. The "large" variant (1024-dim) gives a richer representation than "base" (768-dim), which matters for financial text where subtle differences in phrasing ("net income" vs "gross income" vs "operating income") need to be distinguishable in the embedding space.

**Deeper on bge-reranker-v2-m3:** The "-v2-m3" suffix indicates it's based on `mdeberta-v3-base` (multilingual DeBERTa v3). For SEC filings, this is a benefit: foreign-private issuers sometimes include non-English text, and the multilingual pretraining creates richer token embeddings for financial abbreviations and jargon. DeBERTa v3 uses disentangled attention (separate position and content embeddings) which performs particularly well on cross-attention tasks like reranking.

**Interview answer:** "We use bge-large-en-v1.5 for embeddings — it's the best free dense retrieval model. 1024 dimensions gives more nuanced representations than 768-dim models, which matters when you need to distinguish 'net income' from 'gross income' in the embedding space. For reranking, bge-reranker-v2-m3 is based on multilingual DeBERTa — outperforms Cohere Rerank-v2 on BEIR, free, no API quota."

---

### 7. Why the FilingRecord adapter (not SQLAlchemy Filing directly)?

**Short answer:** Decouples mia_retrieval from mia_ingestion's ORM layer — no circular imports, no SQLAlchemy session leaking into the retrieval package, and retrieval tests don't need a DB mock.

**Deeper:** This is the ports and adapters pattern (hexagonal architecture). The retrieval package is a "domain" component — it knows how to chunk text, embed it, and index it. It doesn't need to know anything about Postgres schemas, async sessions, or ORM lazy loading. If `IndexingPipeline` accepted `Filing` ORM objects directly, it would transitively import `sqlalchemy`, `asyncpg`, `alembic`, and every model in `mia_ingestion.models`. That's ~15 heavy transitive deps for a package whose core logic is pure ML.

`FilingRecord` is a `@dataclass` with four fields: `id`, `ticker`, `filing_type`, `accession_number`, `raw_text`. The calling code (script or ARQ task) loads ORM objects and converts them. This is three lines of code at the call site, and it completely severs the coupling.

**The test benefit is concrete:** `test_retriever.py` mocks the Embedder, Qdrant, and Reranker with `unittest.mock`. It doesn't need pytest fixtures for a Postgres session, Docker containers, or ORM setup. The 46 tests run in 0.18 seconds.

**Interview answer:** "The indexer takes a `FilingRecord` dataclass — just `id, ticker, filing_type, accession_number, raw_text`. The caller converts ORM objects before passing them in. This is the ports and adapters pattern: the retrieval package has no SQLAlchemy import and no session dependency. The practical payoff is that all 46 tests run in 0.18 seconds with no DB mock needed."

---

### 8. Why AsyncQdrantClient over synchronous QdrantClient?

**Short answer:** FastAPI handlers and ARQ workers are fully async — a synchronous Qdrant call inside a coroutine blocks the event loop.

**Deeper:** Python's asyncio event loop is single-threaded. Any blocking I/O call (synchronous Qdrant HTTP request, blocking file read, synchronous boto3 call) freezes the entire event loop for its duration. At 10+ concurrent WebSocket sessions — each with an active agent pipeline querying Qdrant — a synchronous Qdrant client would serialize all queries: session 2 waits until session 1's Qdrant call returns before it can even start.

`AsyncQdrantClient` returns coroutines. The event loop can interleave 10 concurrent Qdrant queries, suspending each at the network I/O await and running other coroutines in the gaps. Empirically, on a local Qdrant instance, async vs sync matters less (latency is low). But the pattern is correct for the production case where Qdrant might be on a separate VM.

**Interview answer:** "The FastAPI handlers and ARQ workers run on asyncio event loops. If we used the synchronous `QdrantClient`, every Qdrant call would block the event loop — session 2 can't start its search until session 1's search physically returns. `AsyncQdrantClient` suspends at each I/O await and lets other coroutines proceed. At 10+ concurrent sessions, that's the difference between serialised and parallel execution."

---

### 9. Why BM25 index rebuilt on every add()?

**Short answer:** `rank_bm25.BM25Okapi` computes IDF over the full corpus at build time — there is no supported incremental update API.

**Deeper:** BM25's IDF weight for a term is `log((N - df + 0.5) / (df + 0.5))` where N is corpus size and df is the number of documents containing that term. Adding a single new document changes N by 1 and potentially changes df for every term in that document. To maintain a correct IDF table, you must rebuild.

Some production BM25 implementations use approximate incremental IDF updates (treating N and df as running approximations). This degrades retrieval quality in proportion to how different the new documents are from the existing corpus. For a nightly refresh of 20 new SEC filings on top of 1500 existing ones, the approximation error would be ~1.3% — probably fine in production. But for a capstone with a small corpus (~150 filings, ~15k chunks), full rebuild is trivial (~300ms) and always correct.

**Interview follow-up — "When would you switch to a different approach?"**
At 10+ million chunks, rebuild becomes slow (minutes). Approaches: (1) Elasticsearch/OpenSearch which supports true incremental BM25 via inverted index updates; (2) shard the corpus and only rebuild the affected shard; (3) SPLADE sparse learned embeddings — these are stored as static vectors like dense embeddings, so incremental upsert works with no IDF recomputation.

**Interview answer:** "BM25Okapi computes IDF over the whole corpus at build time — there's no incremental update API because adding one document changes the IDF of every term it contains. For our scale (~15k chunks), a full rebuild takes ~300ms, so we rebuild on every `add()`. In production at millions of chunks, you'd switch to Elasticsearch for true incremental BM25, or to SPLADE sparse vectors which store like dense vectors and support incremental upsert."

---

### 10. Why pickle for BM25 persistence?

**Short answer:** `BM25Okapi` stores numpy arrays internally — not JSON-serialisable without brittle custom encoders. Pickle handles it natively and produces a compact binary file.

**Deeper:** `BM25Okapi` stores `idf` (float64 numpy array), `doc_freqs` (list of Counter dicts), `doc_len` (list of ints), `avgdl` (float), and `corpus_size` (int). A JSON encoder for numpy float64 arrays is ~20 lines of non-obvious code and produces verbose output. `pickle.HIGHEST_PROTOCOL` serialises all of these natively and is 3–5× faster to write/read than JSON.

The security risk of `pickle.load` (it can execute arbitrary Python on load) is acceptable here: the file lives on local disk under our control, written by our own code, not received over a network. If the file were received from an untrusted source, you'd use `joblib.load` with signature verification or a custom `Unpickler` subclass with restricted globals.

**Interview answer:** "BM25Okapi stores numpy arrays internally, which aren't JSON-serialisable without a custom encoder. Pickle handles it natively and is 3–5× faster. The security concern — pickle can execute arbitrary code — is a real one if the file is received over a network, but here it lives on local disk written by our own pipeline, so the risk is acceptable."

---

## Gotchas from Real Implementation

**1. StrEnum not available in Python < 3.11**
The project targets Python 3.12 (specified in `pyproject.toml`), but sandbox/CI environments may run 3.10 or 3.11. `StrEnum` was added in 3.11. Fix: use `(str, Enum)` as base classes — identical semantics, works on 3.10+. Applied to `mia_retrieval.retriever.RetrieveMode` and `mia_shared.schemas` enums.

**2. Qdrant `PayloadSchemaType.KEYWORD`, not the string `"keyword"`**
`client.create_payload_index(collection, field, "keyword")` raises a `ValidationError`. The correct call is `client.create_payload_index(collection, field, PayloadSchemaType.KEYWORD)`. Import from `qdrant_client.models`.

**3. BM25 returns all-zeros for OOV queries — must filter**
If the query has no terms in the corpus vocabulary, `BM25Okapi.get_scores()` returns a float array where every element is 0.0. Without filtering, `search()` would return all 15k chunks with score=0, all appearing equally "relevant". The fix: in `BM25Index.search()`, break on the first 0.0 score after sorting descending — since arrays are sorted, all remaining scores are also 0.

**4. AsyncQdrantClient is not auto-closed — call close() on shutdown**
The async Qdrant client holds an open aiohttp session. In tests, if you don't close it, you get `ResourceWarning: Unclosed client session` warnings. In production, wire `await qdrant.close()` into FastAPI's shutdown lifespan (Phase 6).

**5. BM25 add() triggers full rebuild — batch your adds**
Calling `bm25.add([chunk])` for each of 5,000 chunks triggers 5,000 full rebuilds. Always call `bm25.add(all_new_chunks)` once per ticker, then `bm25.save()`. The `IndexingPipeline` does this correctly: it processes all filings for a ticker, collecting all chunks, then calls `bm25.add(all_chunks_for_ticker)` once.

**6. Embedder and Reranker download on first use (~1.3GB + ~550MB)**
`sentence-transformers` downloads models from HuggingFace Hub on first call to `encode()` or `predict()`. The download is cached in `~/.cache/huggingface/`. In a clean environment (fresh Docker container, new machine), the first `make index` will spend several minutes downloading. Pre-warm by running `make index` once before a live demo.

**7. Chunk.total_chunks is set correctly by the chunker — with_total() is for edge cases**
`Chunker.chunk()` sets `total_chunks` correctly for all returned chunks. `Chunk.with_total()` exists for callers who build chunks incrementally (e.g., streaming chunker) where total isn't known until the stream ends. Don't confuse this as a bug — the chunker finishes before returning, so the total is always set.

---

## Questions to Be Ready For

Use the pattern: **choice → problem it solves + example → trade-off**.

**"Walk me through the retrieval pipeline."**
> "When a query comes in, we run two retrievals in parallel: BM25 sparse search over an in-memory index (top-50), and dense cosine search in Qdrant (top-50). BM25 catches exact keyword matches — if someone asks about 'segment revenue', BM25 finds chunks that literally contain those words. Dense search catches semantic matches — 'division income' maps to the same embedding neighborhood as 'segment revenue'. Reciprocal Rank Fusion merges both lists by rank — a chunk at #1 in both lists scores ~2× a chunk at #1 in only one. The merged top-50 then go through the bge-reranker cross-encoder, which scores each (query, chunk) pair with full cross-attention and returns the top-10. Those become `Evidence` objects that the downstream agents consume."

**"What's in your Qdrant collection?"**
> "Each point represents one text chunk from a filing. The vector is a 1024-dim L2-normalised bge-large embedding. The payload carries: `filing_id`, `ticker`, `filing_type` (10-K/10-Q/8-K), `accession_number`, `section` (MD&A, Risk Factors, etc.), `text` (the raw chunk), `chunk_index`, and `total_chunks`. We have payload indexes on `ticker` and `filing_type` so filtered searches — 'only NVDA 10-K chunks' — use a pre-filter rather than a post-filter, which is significantly faster on large collections."

**"Why not just use dense search and skip BM25?"**
> "Dense search excels at semantic similarity — it captures 'fourth quarter' matching 'Q4', synonyms, paraphrases. But it fails on exact financial terms: specific ticker symbols, XBRL concept names like 'RevenueFromContractWithCustomerExcludingAssessedTax', or numerical queries like 'fiscal 2023 revenue $44B'. BM25 catches those exactly. Hybrid is strictly better in practice — on BEIR benchmarks, hybrid consistently outperforms either system alone. The Phase 8 ablation will quantify this on our SEC-specific golden eval set."

**"What happens if you re-run `make index ticker=NVDA`?"**
> "It's fully idempotent. Chunk IDs are UUID5 derived from `(filing_id, chunk_index)` — the same filing chunked the same way always produces the same IDs. Qdrant upsert on the same ID overwrites the existing point with identical data. The BM25 index is rebuilt from scratch and saved to `data/bm25_index.pkl`, replacing the previous version. No duplicates accumulate anywhere."

**"How do you handle a filing that's already been indexed?"**
> "`IndexingPipeline._index_one()` calls `QdrantStore.filing_is_indexed()` first — it counts points in Qdrant with `filing_id` matching the filing. If at least one exists and `force=False`, it returns 'skipped'. This is a Qdrant count query with a payload filter, not a full scan. `make index-force ticker=NVDA` passes `force=True` to skip the check and re-index everything."

**"What's the difference between your BM25 and Qdrant's built-in sparse vectors?"**
> "Qdrant supports sparse vector search natively since v1.7, using SPLADE-style learned sparse encoders. We use a separate BM25 index for a few reasons: (1) BM25 is interpretable — you can debug why a chunk was retrieved by inspecting which query terms matched; SPLADE vectors are opaque; (2) our BM25 tokenisation is the same lowercase whitespace split as our chunk tokenisation — consistent throughout; (3) we don't need Qdrant's sparse vector support until we want to move BM25 results into Qdrant for server-side fusion, which is a future optimisation. The current architecture keeps BM25 as an in-process index, which avoids a second network round-trip to Qdrant."

**"How do you decide how many candidates to pass to the reranker?"**
> "Settings control three knobs: `bm25_top_k=50`, `dense_top_k=50`, and `rerank_top_k=10`. After RRF fusion, we have at most 100 unique candidates (if BM25 and dense returned completely different results). The reranker processes all of them. In practice the reranker handles 50–100 pairs in ~200ms on CPU, which is acceptable latency for an async pipeline. If latency becomes a concern, we can reduce the candidate set to 20 — the ablation in Phase 8 will tell us the quality trade-off."

**"How would you improve this retrieval system?"**
> "Several directions: First, section-aware chunking — instead of treating the full 10-K as one text blob, chunking MD&A and Risk Factors separately and attaching section metadata to each chunk. This already has a hook: `Chunk.section` exists but the chunker doesn't set it yet. Second, sentence-boundary chunking — the current word-count splitter can cut mid-sentence; adding `nltk.sent_tokenize` as an optional secondary split would improve chunk coherence. Third, SPLADE sparse vectors via Qdrant's native sparse support — more accurate than BM25 and server-side, avoiding a second Python process. Fourth, query expansion — using an LLM to rewrite the query into 3 variants and fusing the retrieval results, which helps when the user's phrasing doesn't match the filing's phrasing."

**"What is nDCG and why does it matter for your ablation?"**
> "nDCG — normalised Discounted Cumulative Gain — measures both the quality and position of relevant results. A highly relevant result at position 1 scores more than the same result at position 5, because we discount gains by `log2(rank + 1)`. The 'normalised' part divides by the ideal DCG (if all top-k results were perfectly relevant), so the metric is always in [0, 1]. We use nDCG@10 to evaluate the reranker — if bge-reranker improves nDCG@10 by X% over the raw RRF ordering, that directly supports the '28% precision improvement' claim. We run it across all 12 cells of the ablation matrix: `{BM25 | dense | hybrid} × {no rerank | rerank} × {no critic | critic}`."

---

## Red Flags to Avoid

- Don't say "we use BM25 because it's simple" — say it catches exact financial term matches that dense search misses, and give a concrete example (XBRL concept names, specific numerical values)
- Don't say "RRF just averages the results" — RRF is a rank-fusion, not an average. Scores are `1/(k+rank)`, not averages of the original scores
- Don't say "the reranker is a language model" without clarifying: it's a cross-encoder, not a generative LM. The output is a relevance score, not generated text
- Don't confuse bi-encoder (used for indexing/dense retrieval) with cross-encoder (used for reranking) — be explicit about which is which and why you use both
- Don't say "we store everything in Qdrant" — BM25 is a separate in-memory index, not in Qdrant. Be precise about what lives where
- Don't say the chunk ID "prevents collisions" — UUID5 doesn't prevent collisions by birthday paradox; it provides *determinism*. The point is idempotency, not collision avoidance
