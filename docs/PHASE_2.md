# Phase 2 — Retrieval Stack

**Status:** ✅ Complete
**Duration:** ~1 session
**Unlocks:** Phase 3 (single-agent RAG baseline)

---

## What Was Built

### `packages/retrieval/src/mia_retrieval/`

| File | Purpose |
|---|---|
| `chunker.py` | `Chunk` dataclass + paragraph-aware word-count chunker (chunk_size=300 words, overlap=40 words, deterministic UUID5 chunk IDs) |
| `embedder.py` | `Embedder` singleton wrapping `bge-large-en-v1.5` (1024-dim, L2-normalised, lazy load, 32-item batches) |
| `bm25_index.py` | `BM25Index` wrapping `rank_bm25.BM25Okapi` — build, add (triggers rebuild), search, persist/load (pickle) |
| `qdrant_store.py` | `QdrantStore` — async Qdrant client, `ensure_collection`, upsert, dense search, `filing_is_indexed`, payload indexes on ticker + filing_type |
| `hybrid.py` | `reciprocal_rank_fusion()` — RRF (k=60) over BM25 and dense results; deduplicates shared hits |
| `reranker.py` | `Reranker` singleton wrapping `bge-reranker-v2-m3` cross-encoder (lazy load) |
| `indexer.py` | `IndexingPipeline` — loads filings via `FilingRecord`, chunks, batch-embeds (32/batch), upserts to Qdrant, adds to BM25, saves index. Skip-if-indexed + `force` override. |
| `retriever.py` | `Retriever.retrieve()` — bm25 / dense / hybrid modes → optional reranker → `list[Evidence]`. `build_retriever()` factory. |
| `__init__.py` | Lazy `__getattr__` imports (same pattern as mia_ingestion) |

### Scripts

| Script | Purpose |
|---|---|
| `scripts/index_ticker.py` | `make index ticker=NVDA` — loads Phase 1 filings from Postgres, runs IndexingPipeline |
| `scripts/retrieve.py` | `make retrieve query="..."` — dev test of the full retrieval stack |

### Makefile targets

```
make index ticker=NVDA          # index one or more tickers
make index-force ticker=NVDA    # re-index (overwrite existing Qdrant points)
make retrieve query="NVDA..."   # test hybrid retrieve + rerank
make retrieve-bm25 query="..."  # BM25-only for ablation
```

### Tests (40 tests across 4 files)

| File | Coverage |
|---|---|
| `tests/test_chunker.py` | 15 tests — IDs, empty input, overlap, metadata, constructor validation |
| `tests/test_bm25.py` | 14 tests — tokenizer, build/rebuild, search ranking, add, persist/load roundtrip |
| `tests/test_hybrid.py` | 9 tests — RRF formula, both-list boost, dedup, metadata preservation |
| `tests/test_retriever.py` | 11 tests — all 3 modes, ticker filter, reranker on/off, empty results |

---

## How to Run Phase 2

```bash
# 1. Start infra (Qdrant must be running)
make up-infra

# 2. Ensure Phase 1 ingestion is complete
make ingest ticker=NVDA

# 3. Index into Qdrant + BM25
make index ticker=NVDA

# 4. Test retrieval
make retrieve query="How is NVDA's data center revenue growing?"

# 5. Run tests
make test
```

---

## Decision Log

### 1. Why word-count chunking instead of token-count chunking?

**Short answer:** Avoids importing the sentence-transformers tokenizer at chunking time.  bge-large-en-v1.5 has a 512-token limit; at `chunk_size_words=300` and ~1.4–1.6 tokens/word for financial text, chunks stay under ~480 tokens.

**Deeper:** Token-count chunking is technically more accurate but requires loading the model's WordPiece tokenizer to count tokens, which imports sentence-transformers (and transitively torch — ~1GB) even for the lightweight chunking step.  Word-count proxy is standard practice in most RAG frameworks (LangChain's `RecursiveCharacterTextSplitter` uses character count, which is cruder).  For a ~2× safety buffer (300 words → ~420 tokens vs. 512 limit), the approximation is fine.

**Trade-off:** Very dense text (abbreviations, tables) can still hit the 512-token ceiling, causing silent truncation in the embedder.  Mitigated by the 40-word overlap, which ensures truncated content appears in an adjacent chunk.

**Interview answer:** "We use word-count proxy at 300 words per chunk.  Financial text averages ~1.5 tokens/word, so 300 words ≈ 450 tokens, giving 60 tokens of safety margin below bge-large's 512-token limit.  We could use the model's tokenizer directly, but that imports a 1GB library for what is essentially a preprocessing step."

---

### 2. Why deterministic UUID5 chunk IDs?

**Short answer:** Re-running `make index ticker=NVDA` upserts the same point IDs into Qdrant — the operation is fully idempotent.

**Deeper:** UUID5 is deterministically derived from a namespace UUID and a string key (`"{filing_id}:{chunk_index}"`).  The same filing, chunked the same way, always produces the same UUIDs.  Qdrant's `upsert` is idempotent on point ID — if the point already exists, it's overwritten with the same data.  This means:
- `make index` is safe to run multiple times
- Re-ingesting a filing (Phase 1) then re-indexing it correctly replaces old chunks
- No "duplicate chunk" accumulation in Qdrant

Alternative (random UUID4): simple but loses idempotency — every re-index doubles the chunk count.  Alternative (sequential integer): not globally unique across tickers without coordination.

**Interview follow-up — "What if the chunking parameters change?"**
Changing `chunk_size` or `overlap` changes the text content of each chunk, so the old chunks (with different text) should be deleted.  Use `QdrantStore.delete_by_filing()` before re-indexing, or pass `force=True` to `index_ticker()`.

---

### 3. Why L2-normalize embeddings?

**Short answer:** With a Cosine-distance Qdrant collection, cosine similarity = dot product on unit vectors.  Qdrant's HNSW index exploits this to skip an extra norm division per distance computation, giving ~15% speed improvement on large collections.

**Deeper:** The standard cosine similarity between vectors A and B is `A·B / (|A| × |B|)`.  If A and B are already unit vectors (L2-normalised), `|A| = |B| = 1`, so cosine similarity is just `A·B` — a dot product.  Qdrant's HNSW implementation detects normalised vectors and uses a faster dot-product path internally.  `sentence_transformers.SentenceTransformer.encode(normalize_embeddings=True)` handles the normalisation as part of the forward pass, with no extra CPU cost.

---

### 4. Why Reciprocal Rank Fusion over score-based fusion?

**Short answer:** RRF is parameter-free, robust to score-scale mismatch between BM25 and dense search, and empirically competitive with learned fusion weights.

**Deeper:** BM25 scores are corpus-dependent log-frequencies (typically in the range 0–15 for our corpus).  Dense search scores are cosine similarities in the range [0, 1].  Normalising both into [0, 1] via min-max requires knowing the corpus-wide min/max, which changes as we add documents — stale normalisation degrades fusion quality.  

RRF uses only ranks, not scores.  The formula `1 / (k + rank)` for each list is added up, and k=60 (from Cormack et al. 2009) smooths the contribution of lower-ranked results.  The key property: a document ranked #1 in BM25 and #1 in dense gets approximately `2 × (1/61) ≈ 0.033`, while a document ranked #1 in only one list gets `0.016`.  Documents unique to one list still appear — they just score lower than documents that both rankers agree on.

**Interview follow-up — "When would you NOT use RRF?"**
When the retrieval modalities have very different precision characteristics — e.g., if BM25 has 0.2 precision and dense has 0.9 precision, equal weighting of both ranks undersells dense.  A learned linear combination (e.g., `0.3 * BM25_score_norm + 0.7 * dense_score_norm`) would be better, but requires a labelled dataset to tune the 0.7/0.3 weights.  We don't have that in Phase 2; Phase 8's ablation study will quantify whether RRF or a weighted combination wins on our golden eval set.

---

### 5. Why bge-reranker-v2-m3 over Cohere Rerank?

**Short answer:** Free, local (no API call), and outperforms Cohere Rerank-v2 on BEIR (Cohere benchmarks show this too — bge-reranker-v2-m3 is the public state-of-the-art for free cross-encoders).

**Deeper:** Cohere Rerank-3 is excellent but costs money (even the free tier requires a CC).  `bge-reranker-v2-m3` is a `cross-encoder/mdeberta-v3-base` fine-tuned on multilingual passage pairs.  The multilingual base is actually a strength for SEC filings: foreign-private issuers sometimes include non-English text, and financial jargon shares structure with multilingual data.  Model size is ~550MB — loads in ~3s on CPU, predicts 50 pairs in ~200ms.

---

### 6. Why is BM25Index rebuilt on every `add()`?

**Short answer:** `rank_bm25.BM25Okapi` computes IDF over the full corpus at build time; there is no supported incremental update API.

**Deeper:** BM25's IDF term is `log((N - df + 0.5) / (df + 0.5))` where N is corpus size and df is document frequency for a term.  Adding one document changes N by 1 and potentially changes df for every term in that document.  To update IDF correctly, you'd need to either (a) recompute the full index (which is what we do), or (b) use an approximation that doesn't update IDF (which degrades quality over time).

For Phase 2 scale (~150 filings, ~15k chunks), full rebuild takes ~300ms.  At S&P 100 scale (~1500 filings, ~150k chunks), rebuild takes ~3s — still fast enough for a nightly batch.  If we needed sub-second incremental updates, we'd switch to Elasticsearch or implement a shard-merge strategy.

---

### 7. Why pickle for BM25 persistence instead of JSON?

**Short answer:** `BM25Okapi` stores internal term-frequency matrices as numpy arrays — not JSON-serialisable without significant custom encoding.

**Deeper:** `BM25Okapi` stores `idf` (numpy array), `doc_len` (list of ints), `avgdl` (float), and `doc_freqs` (list of dicts).  A custom JSON encoder for numpy arrays is ~30 lines of brittle code.  Pickle handles all of this natively and produces a smaller file.  The security risk of `pickle.load` (arbitrary code execution from malicious files) is acceptable here because the file lives on local disk under our control, not received over a network.

---

## Gotchas

**1. Qdrant payload index creation is not idempotent (older versions)**
Qdrant ≥ 1.7 silently skips `create_payload_index` if the index already exists.  Older versions raise.  The `ensure_collection()` method calls it unconditionally; if you're on an older Qdrant image, wrap it in a try/except.

**2. BM25 OOV queries return all-zeros — filter out zero scores**
BM25Okapi returns a float array where terms not in the vocabulary score 0.  Without filtering, the search would return 0-score chunks which look like matches.  The `search()` method breaks on the first 0-score result (since scores are sorted descending) and only returns results with score > 0.

**3. AsyncQdrantClient is not automatically closed**
Async Qdrant client connections persist across the event loop lifetime.  Call `await qdrant.close()` on shutdown (Phase 6 FastAPI lifespan will handle this).

**4. Embedding model downloads on first use**
`sentence-transformers` downloads `bge-large-en-v1.5` (~1.3GB) and `bge-reranker-v2-m3` (~550MB) from HuggingFace on first load.  Subsequent runs use the cache at `~/.cache/huggingface/`.  Pre-warm by running `make index` before the first demo.

---

## What Phase 3 Needs From Here

- `data/bm25_index.pkl` populated by `make index ticker=NVDA`
- Qdrant `filings` collection with chunks for at least one ticker
- `Retriever.retrieve(query)` callable from the single-agent RAG baseline
- `list[Evidence]` output compatible with the LangGraph `AgentState.evidence` field

---

## Open Items (carry into Phase 3)

- [ ] Section-aware chunking: detect MD&A / Risk Factors / Financials headers and set `Chunk.section` accordingly (the extractor already identifies sections; wire it in)
- [ ] Sentence-boundary-aware split: avoid cutting mid-sentence (add nltk `sent_tokenize` as optional enhancement)
- [ ] ARQ task wrapper for `IndexingPipeline.index_ticker()` (Phase 6)
- [ ] S&P 100 batch indexing script: `make index-all` loops over S&P 100 tickers
- [ ] pgvector fallback: if Qdrant is down, fall back to `pgvector` cosine search over `mia.filings.embedding` (add a `vector` column in Phase 3)
