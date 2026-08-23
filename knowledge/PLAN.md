# Semantic Cache: Build Plan

High level execution plan. Full reasoning and context lives in the project brief (`semantic-cache-project-brief.md`), this file is the working checklist for actually building the thing, in order.

## Ground rules for this build

- Local first. No Bedrock, no AWS, no billed API calls until the core system already works. Embeddings come from a local model (sentence-transformers), the "LLM call" on cache miss is stubbed or run through a local model (Ollama) until we deliberately decide to wire in the real thing.
- Linear search comes before HNSW, on purpose. It is not just a warmup exercise, it becomes the ground truth we check HNSW's approximate results against later (recall = how often HNSW returns the same neighbor the brute-force search does).
- Each phase below produces something runnable before moving to the next. No phase starts until the previous one is working.

## Phase 1: Linear (brute-force) vector search

Goal: given a query vector and a set of stored vectors, find the nearest neighbor(s) by cosine similarity, correctly, with no approximation.

- `src/linear_search.py`: a small index class, insert(id, vector) and search(query_vector, k) -> ranked list of (id, similarity).
- Vector math only at this stage, no embeddings wired in yet. Test with small synthetic vectors we can check by hand.
- `tests/test_linear_search.py`: correctness checks (self-similarity is 1.0, ordering is descending by similarity, top-k matches hand-computed values on a tiny fixed dataset).

Status: in progress, iterating with Claude in this session first, then continuing in VS Code.

## Phase 2: Eval harness + key metrics

Goal: a way to measure the linear search index against a real test set, so we have baseline numbers before HNSW exists.

- Build the paraphrase / near-miss test set (`data/`): pairs of queries that should match and pairs that should not.
- Embed the test set with sentence-transformers.
- Run the linear index across a range of similarity thresholds, record hit rate and wrong-match rate at each threshold.
- This produces the first real numbers for the project, and the harness gets reused unchanged for HNSW later, only the index underneath swaps out.

## Phase 3: HNSW from scratch

Goal: an approximate nearest neighbor index (hierarchical navigable small world graph) with insert and search, built without a library.

- `src/hnsw.py`: layered graph, insert with probabilistic layer assignment, greedy search for entry point, neighbor selection.
- Correctness check: for a given query, HNSW's top-k should closely match linear search's top-k on the same dataset. This is where Phase 1's brute-force index earns its keep, it is the answer key.

## Phase 4: Compare linear vs HNSW

Goal: the actual "built it myself vs baseline" comparison that makes this a system, not just an algorithm exercise.

- Recall@k: how often does HNSW return the same nearest neighbor as linear search.
- Speed: query latency, linear search vs HNSW, as index size grows. This is where HNSW's reason to exist becomes visible (linear search degrades linearly with index size, HNSW should not).
- Re-run the Phase 2 threshold sweep on HNSW, confirm hit rate / wrong-match rate numbers hold up against the approximate index, not just the exact one.

## Phase 5: Cache routing + storage

Goal: wire the index into the actual cache decision logic (hit -> return stored response, miss -> call LLM, store result).

- Cache storage: in-memory dict is fine to start, SQLite if persistence matters.
- Swap the stubbed/local LLM for the real Bedrock call only once this is stable, deliberately, not by accident during earlier debugging.

## Phase 6 (stretch, cut first if time is short): AWS wiring

API Gateway + Lambda in front of Bedrock, DynamoDB for cache storage, CloudWatch dashboard. Only after Phase 1-5 work locally.

## Where we are right now

Phase 1 through Phase 4 are complete. Linear search is built and tested, the eval harness is built and verified against a 194-pair test set, the operating threshold (0.80 on all-mpnet-base-v2) is chosen and documented in `knowledge/learned.md`, `src/hnsw.py` is built from scratch and validated against `LinearIndex` (194/194 top-1 agreement on the real eval set), and Phase 4's recall@k + latency comparison at scale (1k/10k/50k synthetic vectors) is measured and documented, including a real O(n^2) bug found and fixed in `LinearIndex.insert` along the way. Starting Phase 5 next: cache routing (hit -> return stored response, miss -> call LLM, store result).

## Agent usage guardrails

- Keep the project aligned to the active phase in this plan.
- Update `Progress.md` after meaningful milestones or blockers.
- Prefer the smallest useful verification step before moving on.
- Do not expand into AWS, dashboard, or speculative architecture work until the local core is fully working and measured.
- If a task appears to drift into a rabbit hole, pause, summarize the tradeoff, and return to the approved roadmap.
