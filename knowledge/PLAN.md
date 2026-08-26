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

## Phase 7: making the miss path safe, and a demo surface

Everything here exists because storage changed what a bad answer costs. Through Phase 5 the miss path called a stub and nothing persisted, so a wrong or missing answer was thrown away. Once entries are written down and replayed, a single bad answer becomes permanent.

**Never cache a bad answer.** An LLM that *raises* was always safe, since `route()` writes nothing on an exception. The hole was an LLM that returns successfully with unusable content: an empty string is not `None`, so it passed the orphan check and would be served as a confident HIT to every future paraphrase, permanently. `BedrockLLM` now raises on a refusal, on a `max_tokens` truncation (a half-finished answer looks fine and would be replayed forever), and on empty text. `CacheRouter` independently rejects a falsy or whitespace-only response, because `llm` is caller-supplied and the router should not trust it. Same principle as refusing to serve a `None` found in the store (learned.md section 21).

**Fail usefully at the boundary.** `api.py` returns 429 for the daily cap (a policy decision), 503 for an unusable answer or an upstream failure, and 422 for a query that is empty or over 2000 characters. Every one of those paths leaves the cache untouched, which is the actual guarantee worth having.

**A daily ceiling on LLM calls** (`src/usage_limiter.py`). The API is deliberately public so the demo URL always works, and every miss costs a real Bedrock call, so a stranger sending novel queries spends real money. The counter lives in DynamoDB rather than memory: an in-memory count resets on restart, so anyone who could restart the process could clear the cap. Its row carries no `vector` attribute, which is what makes it invisible to `all_items()` and therefore to the cold-start rebuild. This is the innermost of three cost layers, bounding spend *before* it happens rather than hours later when billing data catches up (learned.md section 23c).

**The stub was removed outright, not improved.** `llm` is now a required argument with no default, joining `index_kind` and `embedder` for the same reason: too load-bearing to assume. A miss with no model configured returns **501** rather than placeholder text. The old default cached its stub and then served it back as a confident hit, so the cache accumulated answers nothing had ever answered. Third instance of the same shape in this project (learned.md section 23e): a default that was harmless in one phase became a defect in the next, silently.

**A curated starter corpus** (`data/seed_answers.json`): 67 real answers for the collision-verified anchors, seeded automatically when the cache comes up empty. The cache is fully demonstrable with no model at all -- hits, rewordings, and refused near-misses all work -- because the only operation needing a model is answering something genuinely new. Entries carry a `source` field (`seed` or `llm`), without which a plausible curated answer is indistinguishable from a generated one, and a purge cannot avoid deleting answers that were paid for.

**One-command reset** (`scripts/reset_cache.py` -> `POST /admin/reset`). Reset lives in the service because the in-memory index is what actually needs clearing and only the process holds it; purging storage from outside leaves the running index serving entries that no longer exist. It clears the store, **deletes the on-disk snapshot** (skipping that is the silent failure: the reset looks fine until a restart reloads the old graph), rebuilds, and re-seeds. Guarded by `RESET_TOKEN` and disabled entirely when unset. `scripts/purge_cache.py` remains for the one-off cleanup of entries written before this existed.

**A demo page served by FastAPI** (`src/static/index.html` at `/`), showing hit/miss, similarity against the threshold, latency, and a running count of model calls avoided.

**S3 + CloudFront was considered and rejected** for that page, on the same grounds as OpenSearch Serverless (section 16), provisioned concurrency (section 19), and Lambda (section 23): a service has to earn its place. The page is useless without this backend, so splitting them across origins buys CORS configuration and a page that loads but errors whenever the instance is stopped, which is most of the time by design. It also demonstrates nothing the existing stack does not already cover.

## What is left

Everything below is either deployment or blocked on AWS. No local feature work remains: 82 tests pass, and every part of the system except answering a genuinely new question is exercisable on a laptop.

**Sequencing decision: nothing deploys until Bedrock works.** The instance stays stopped and on old code deliberately. One deploy with a finished system beats several partial ones, and the things only AWS can verify (the instance role's permissions, systemd, real cold-start timing, DynamoDB across a stop/start) are worth checking against final code rather than something about to be replaced. Items 1-3 below are ready whenever item 4 unblocks.

**Ready to ship, gated on Bedrock by choice rather than by dependency:**
1. **Deploy to the instance.** It is nine commits behind, still running the version that fabricated stub answers and had no seed corpus, reset, spend cap, or demo page. `git pull` plus a service restart.
2. **Purge the deployed table.** It holds 73 items: the 67 anchors plus six left by testing, and every one still carries `[stub response for: ...]`. Those do not disappear on their own -- they keep being served as hits, which look like success. `scripts/purge_cache.py --yes`, then the service re-seeds the canonical 67 on restart.
3. **Set `RESET_TOKEN`** in `/etc/systemd/system/semantic-cache.service` so the one-command reset works against the deployed instance. Currently commented out, which leaves the endpoint returning 404.

**Blocked on AWS, not on us:**
4. **Bedrock quota.** `L-CCA5DF70` and `L-58BE175A` (Claude Haiku 4.5 requests and tokens per minute) are both adjustable and still read **0**. Until they are granted, a miss returns 501 by design. Once granted: set `LLM_MODEL_ID`, restart, reset, and let the miss path produce real answers. The daily cap in `usage_limiter.py` becomes active at the same moment.

**Small cleanup, no urgency:**
5. **The local snapshot is written but never read.** `build_router()` loads a snapshot only when a durable store is configured, but the shutdown handler writes one unconditionally. Locally that means ~200KB of dead I/O on every stop, and a file on disk that looks meaningful and is not.

**Deliberately not doing** (recorded so they are decisions rather than omissions):
- **Cache eviction / LRU** -- learned.md section 22b. At the measured 1k-10k range nothing is under memory pressure, and HNSW cannot delete in place, so this is real work against a problem the measurements say does not exist.
- **CloudWatch dashboard** -- `/stats` and the demo page already report hit rate, entry count, and calls avoided. A dashboard would be a second implementation of the same numbers.
- **S3 + CloudFront** for the frontend -- see Phase 7.

**One open lead worth chasing eventually:** learned.md section 22b found that HNSW's edge pruning orphans nodes on clustered data (245 of 2,000 with zero in-edges), and the eval benchmark generates exactly that shape. It may explain the remaining half of section 15's unclosed recall ceiling, and would explain why recall stayed flat across a 32x wider `ef_search`. Recorded as a hypothesis with the test that would confirm it, not as a finding.

## Where we are right now

Phase 1 through Phase 5 are complete. Linear search and HNSW are both built, tested, and benchmarked against each other at a realistic 1k-10k scale (two real bugs found and fixed along the way: an O(n^2) insert bug and an unrecoverable upper-layer routing bug, see `knowledge/learned.md`). `src/cache_router.py` wires embedding, index, and storage into the real hit/miss decision.

Phases 6 and 7's application code is built and green (82 tests): FastAPI, DynamoDB storage, graph snapshotting, a swappable embedder, a curated seed corpus, spend caps, a demo page, and a one-command reset. Nine real bugs were found and fixed along the way, most of which would have shipped silently -- see learned.md sections 21, 22, 23c, 23d, 23e, and 24. The stack is deployed on AWS but the instance is stopped and running older code. See "What is left" above for the exact remaining steps.

## Agent usage guardrails

- Keep the project aligned to the active phase in this plan.
- Update `Progress.md` after meaningful milestones or blockers.
- Prefer the smallest useful verification step before moving on.
- Do not expand into AWS, dashboard, or speculative architecture work until the local core is fully working and measured.
- If a task appears to drift into a rabbit hole, pause, summarize the tradeoff, and return to the approved roadmap.
