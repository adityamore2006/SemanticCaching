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

## Phase 6: AWS wiring

Decided architecture (see knowledge/learned.md section 16 for the cost reasoning): API Gateway (HTTP API) + Lambda + DynamoDB (on-demand billing, not provisioned) + Bedrock (called only on cache miss) + CloudWatch dashboard. Fully serverless and chosen specifically to minimize idle cost -- nothing runs, or costs anything, when it's not being called. OpenSearch Serverless is explicitly ruled out (real idle billing floor, conflicts with the economical-to-host goal); the from-scratch HNSW index runs inside Lambda itself rather than a managed vector DB.

Decided persistence strategy (see knowledge/learned.md section 19): Lambda is stateless between invocations, so the in-memory HNSW graph gets rebuilt from DynamoDB on every cold start (reusing insert() as-is) and kept in memory for that container's lifetime. Provisioned concurrency is explicitly rejected (standing 24/7 cost). Estimated $0-1/month at demo-level traffic; cold-start latency (~16.5s at n=10,000) is the honest non-dollar cost. index_kind is always "hnsw" for the deployed system -- LinearIndex never gets deployed.

Design for this phase is complete; implementation was deliberately deferred to its own dedicated session. See Progress.md's "Handoff for Phase 6" section for the exact starting checklist. Only start implementation after Phase 5's cache-routing logic is built and validated locally -- same reasoning as linear-before-HNSW, prove the logic somewhere cheap and fast to iterate on before wiring it to live, billed AWS resources. (Phase 5 is complete as of this writing.)

## Where we are right now

Phase 1 through Phase 5 are complete. Linear search and HNSW are both built, tested, and benchmarked against each other at a realistic 1k-10k scale (two real bugs found and fixed along the way: an O(n^2) insert bug and an unrecoverable upper-layer routing bug, see `knowledge/learned.md`). `src/cache_router.py` wires embedding, index, and storage into the real hit/miss decision, verified by 34 passing tests and a live end-to-end run with the real embedding model. Phase 6's architecture and persistence strategy are fully decided and documented (see the Phase 6 section above and `Progress.md`'s "Handoff for Phase 6"), but implementation is deliberately deferred to its own dedicated session.

## Agent usage guardrails

- Keep the project aligned to the active phase in this plan.
- Update `Progress.md` after meaningful milestones or blockers.
- Prefer the smallest useful verification step before moving on.
- Do not expand into AWS, dashboard, or speculative architecture work until the local core is fully working and measured.
- If a task appears to drift into a rabbit hole, pause, summarize the tradeoff, and return to the approved roadmap.
