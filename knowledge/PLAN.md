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

**Current architecture: one small EC2 instance + DynamoDB + Bedrock.** The service runs as a long-lived FastAPI process on a `t4g.medium`, started before a demo and stopped afterwards. DynamoDB (on-demand) is the durable store for cached responses and their vectors; Bedrock (Claude Haiku) is called only on a cache miss. index_kind is always "hnsw" for the deployed system -- LinearIndex never gets deployed, it stays the local ground-truth tool HNSW is graded against.

**This replaced a fully serverless design (API Gateway + Lambda + container image + ECR), and the reasoning is worth keeping.** The original choice optimized for zero idle cost, which is right in general and wrong for this workload specifically: Lambda recycles idle containers on AWS's schedule, so an unpredictable multi-second cold start lands on an arbitrary request. A cache whose entire pitch is latency cannot absorb that. Keeping a Lambda warm costs about $21.90/month, more than just running the instance, and switching the instance off makes it an order of magnitude cheaper (~$2-3/month at demo usage). Full cost comparison and reasoning in knowledge/learned.md section 23.

**What removing Lambda deleted:** Lambda, API Gateway, ECR, the Docker image, and SAM packaging. All five existed only to work around Lambda's 250MB package limit, which was also the sole reason embedding had been pushed to a managed model (Bedrock Titan). With no size ceiling, embedding runs locally on `all-mpnet-base-v2` again, which keeps the locked 0.80 threshold and all of Phase 2's eval work valid rather than needing re-derivation (learned.md sections 20 and 23).

**Persistence:** the built HNSW graph is snapshotted to a local file on the instance's EBS volume at shutdown and reloaded at boot. DynamoDB remains the source of truth, and rebuilding from it via `CacheRouter.restore()` is the fallback whenever the snapshot is missing, corrupt, or was built by a different embedding model. Section 19 had deferred graph snapshotting as too complex; that complexity was entirely a property of concurrent stateless Lambdas and disappears on a single instance with a persistent disk.

**Still ruled out, unchanged:** OpenSearch Serverless (real idle billing floor) and Lambda provisioned concurrency (standing 24/7 cost, and now also simply more expensive than the instance).

## Where we are right now

Phase 1 through Phase 5 are complete. Linear search and HNSW are both built, tested, and benchmarked against each other at a realistic 1k-10k scale (two real bugs found and fixed along the way: an O(n^2) insert bug and an unrecoverable upper-layer routing bug, see `knowledge/learned.md`). `src/cache_router.py` wires embedding, index, and storage into the real hit/miss decision.

Phase 6's application code is built and green (51 tests): `src/api.py` (FastAPI), `src/dynamodb_cache_store.py`, `src/bedrock_llm.py`, graph snapshot save/load, and a swappable embedder interface. Four more real bugs were found and fixed while building it, three of which would have shipped silently -- see learned.md sections 21, 22, and 24. What remains is provisioning the actual AWS resources (EC2 instance, IAM role, DynamoDB table) and deploying, plus a Bedrock quota increase for the miss-path LLM call. See `Progress.md` for the exact next steps.

## Agent usage guardrails

- Keep the project aligned to the active phase in this plan.
- Update `Progress.md` after meaningful milestones or blockers.
- Prefer the smallest useful verification step before moving on.
- Do not expand into AWS, dashboard, or speculative architecture work until the local core is fully working and measured.
- If a task appears to drift into a rabbit hole, pause, summarize the tradeoff, and return to the approved roadmap.
