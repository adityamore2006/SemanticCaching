# Semantic Cache for LLM APIs

A caching layer that recognizes when a new question **means** the same as one already
answered, and serves the stored response instead of paying for another model call.

Unlike an exact-match cache, "How do I merge my accounts?" and "Is it possible to combine my
two separate accounts?" are the same question, and this serves one answer for both. The vector
index that makes that possible is written from scratch, no FAISS and no hnswlib.

```
MISS  sim=n/a     "Can I merge two accounts into one?"          model called
HIT   sim=0.9404  "Is it possible to combine my two accounts?"  no model call
MISS  sim=0.7266  "Can I delete one of my two accounts?"        correctly refused
```

That third line is the point. Same topic, opposite intent, and serving the merge answer there
would be a confidently wrong response with no failure signal.

## The actual engineering problem

Not "build a cache". **Where do you put the similarity threshold?**

Too loose and the cache serves a wrong answer with total confidence and no error anywhere. Too
strict and it never fires, saving nothing. That is a precision/recall tradeoff, so it has to be
measured rather than guessed.

I built a 194-pair labeled evaluation set (67 true paraphrases, 67 deceptive near-misses, 60
unrelated) and swept the threshold across it:

| threshold | hit rate | wrong-match rate |
|---|---|---|
| 0.70 | 0.55 | 0.21 |
| 0.75 | 0.40 | 0.12 |
| **0.80 (chosen)** | **0.22** | **0.07** |
| 0.85 | 0.04 | 0.04 |

**0.80, and deliberately not 0.85.** The costs are asymmetric: a missed hit costs one extra API
call, while a wrong hit serves an incorrect answer silently. So the threshold weights false
positives heavily. But at 0.85 the hit rate collapses to 4%, which is not "safer", it is a
cache that no longer functions.

## Index performance

From-scratch HNSW against brute force, same data, `all-mpnet-base-v2`:

| n | linear query | HNSW query | recall@1 |
|---|---|---|---|
| 1,000 | 0.055 ms | 0.371 ms | **99.5%** |
| 5,000 | 0.234 ms | 0.287 ms | 86.5% |
| 10,000 | 0.632 ms | 0.419 ms | 70.0% |

**Read honestly: brute force wins outright at 1,000.** It is faster *and* effectively exact.
The crossover lands in the low-to-mid thousands, above which HNSW pulls ahead. Recall falls as
a fixed-width search covers less of a growing graph, which is the tradeoff HNSW exists to make,
not a defect being hidden.

Two real bugs surfaced only at scale, both found by measurement and fixed the same way:

- **O(n²) insert.** `np.vstack` reallocating the whole array every call. 443s to 0.37s at
  n=50,000. Invisible to correctness tests, because small n cannot distinguish O(n) from O(n²).
- **Unrecoverable upper-layer routing.** Recall was flat across a 32x wider search beam, which
  killed the obvious hypothesis and pointed at a zero-backtracking greedy walk through the
  graph's upper layers. Recall 54% to 65.5% at n=50,000.

## Architecture

One long-lived local process holds the embedding model, the HNSW graph, and the hit/miss
decision. Entries persist in a SQLite file; the graph is rebuilt from it at startup.

```
client -> FastAPI ---> embed (mpnet, in-process)
                  ---> search HNSW (in RAM)
                  ---> similarity >= 0.80 ?
                         yes -> return stored answer        (no model call)
                         no  -> LLM -> write SQLite -> insert in index -> return
```

Every part behind a seam is swappable: `VectorIndex` (linear, hnsw), `Embedder`, `CacheStore`
(in-memory, SQLite) and `LLM` (OpenAI-compatible HTTP, fake). Each has a small factory.

The embedder decides hit or miss on every request, so the 0.80 threshold belongs to it. The LLM
only runs on a miss and does not affect the threshold. Changing the embedder means re-deriving
the threshold with `eval/threshold_sweep.py`.

An earlier version was deployed to AWS (EC2, DynamoDB, Bedrock). That code is removed; the
decisions and measurements from that phase are kept in `knowledge/learned.md`, and the code is
in git history.

## Running it

Needs Python 3.9 or newer. The first run downloads the embedding model (`all-mpnet-base-v2`,
about 400 MB) once.

```bash
make setup                          # once: creates .venv and installs dependencies
ollama pull qwen2.5:7b-instruct     # once: the model that answers new questions
make run                            # starts http://127.0.0.1:8000
```

Open http://127.0.0.1:8000 for the demo page, or http://127.0.0.1:8000/docs for the API. On
first start the service loads 67 curated support answers into `data/cache.db`, so a cached
question answers immediately. A question the cache does not know goes to the model, and the
answer is stored.

**No Ollama?** `make run-fake` starts the same service with a fake model. Cached questions work
as normal, and a new question gets a placeholder answer. Everything below except real answers
works with it.

### Try it

With the service running, in a second terminal:

| Command | What it shows |
|---|---|
| `make tour` | Labelled queries: which ones hit, which near-misses are refused, and the similarity of each |
| `make loadtest` | 20 identical questions, 20 different ones, and a flood of 60 slow ones, with the numbers |
| `make reset` | Restores the cache to the 67 curated entries |
| `make test` | The 154 tests. No service and no Ollama needed |
| `.venv/bin/python eval/threshold_sweep.py` | Regenerates the threshold table above |

`make loadtest` starts its own copy of the service on port 8765 with a fake 3-second model, so
it does not touch the one on port 8000. Run `make help` for the full list.

Or by hand:

```bash
# a question the cache knows: hit, no model call
curl -s -X POST localhost:8000/query -H 'Content-Type: application/json' \
  -d '{"query": "Is it possible to combine my two separate accounts?"}'

# a new question: miss, the model answers, the answer is stored. Ask it again and it hits.
curl -s -X POST localhost:8000/query -H 'Content-Type: application/json' \
  -d '{"query": "What is the capital of France?"}'

curl -s localhost:8000/metrics        # counters, model calls, latency percentiles
```

### Endpoints

| Endpoint | |
|---|---|
| `POST /query` | `{"query": "..."}`. Returns `response`, `hit`, `coalesced`, `similarity`, `matched_id`, `latency_ms`. A request that shared an identical in-flight request's model call has `hit: false, coalesced: true`. |
| `GET /metrics` | Request counts (hits, misses, coalesced, shed, errors), model calls, in-flight gauge, p50/p95/p99 latency, `queue_wait`, miss-path usage |
| `GET /stats` | The smaller view the demo page uses |
| `GET /health` | Liveness. Answers even when every worker thread is busy |
| `POST /admin/reset` | Restores the curated corpus. Needs `X-Reset-Token`; returns 404 unless `RESET_TOKEN` is set. `make run` sets it to `dev-token` |
| `GET /` | The demo page |

Errors: `503` if the model is unreachable or returned nothing usable, `504` if it timed out
(nothing is cached in either case, because a cache replays whatever it stores), and `503` with
`Retry-After: 1` if too many misses are in flight (`MAX_CONCURRENT_MISSES`). Cache hits are
never refused for that reason. `/metrics` counts the three separately.

### Settings

Environment variables, all optional. Numbers and booleans are parsed strictly: a typo such as
`MAX_CONCURRENT_MISSES=0` or `SEED_ON_EMPTY=flase` stops the boot with a message instead of
being guessed at.

| Variable | Default | |
|---|---|---|
| `CACHE_DB_PATH` | `data/cache.db` | SQLite file |
| `LLM_BACKEND` | `openai_compat` | or `fake` (no model) |
| `LLM_MODEL` | `qwen2.5:7b-instruct` | any model the server has |
| `LLM_BASE_URL` | `http://localhost:11434/v1` | Ollama; any OpenAI-compatible server works |
| `LLM_API_KEY` | none | for hosted providers |
| `LLM_TIMEOUT_SECONDS` | `30` | give up on a model call after this long (504) |
| `MAX_CONCURRENT_MISSES` | `24` | most requests allowed in the miss path at once; more get a 503 with `Retry-After` |
| `OPERATING_THRESHOLD` | `0.80` | |
| `RESET_TOKEN` | unset | enables `POST /admin/reset` |
| `SEED_ON_EMPTY` | `true` | `false` starts with an empty cache |

Run one server process. The cache lives in that process's memory, so extra `--workers` would
each build their own index and mint colliding ids against the same SQLite file.

## Under concurrent load

Measured with a fake model that takes 3 seconds (real embedder, index and SQLite), 10-core
laptop, synthetic traffic. Reproduce with `scripts/loadtest.py`.

| | Before | After |
|---|---|---|
| 20 identical new questions at once: model calls | 20 | **1** |
| 20 different new questions at once: total time | 3.3 s | 3.3 s |
| A cached question while 60 slow questions run | 3.1 s | **0.11 s** |
| Those 60 slow questions | all answered, over about 6 s | 24 answered, 36 refused at once (503) |
| Entries lost or crashes, forced thread switching | 15 of 40 trials | 0 |

The 0.11 s misses my own 100 ms target; the embedder is CPU-bound under a burst. Sharing a
model call matches identical text, not paraphrases, and the service must run as one process.
The full account, including what is not fixed, is in
[`knowledge/concurrency.md`](knowledge/concurrency.md).

## Layout

| Path | |
|---|---|
| `src/hnsw.py` | The HNSW index, written from scratch |
| `src/linear_search.py` | Brute force, kept permanently as the ground truth HNSW is graded against |
| `src/cache_router.py` | The hit/miss decision the whole project builds toward |
| `src/sqlite_cache_store.py` | Durable storage; the index is rebuilt from it at startup |
| `src/llm.py` and backends | The pluggable LLM interface (`openai_compat_llm.py`, `fake_llm.py`) |
| `src/api.py` | HTTP entry point |
| `src/single_flight.py` | Lets identical concurrent misses share one model call |
| `src/limiter.py` | A concurrency cap that refuses instead of waiting, used to keep worker threads free for hits |
| `src/metrics.py`, `src/metered_llm.py` | Counters and latency percentiles; a wrapper that counts every model call |
| `scripts/loadtest.py` | Repeatable concurrency scenarios, so each fix can be judged by measurement |
| `src/static/index.html` | The demo page |
| `scripts/` | `try_queries.py` (the tour), `loadtest.py`, `reset_cache.py` |
| `tests/` | 154 tests; none need Ollama or the network |
| `Makefile` | `make setup`, `run`, `run-fake`, `test`, `tour`, `loadtest`, `reset` |
| `eval/` | Threshold sweep, recall/latency benchmarks, collision verification |
| `data/eval_pairs.json` | The 194-pair labeled evaluation set |
| `knowledge/learned.md` | **28 documented decisions, tradeoffs, and debugging stories** |
| `knowledge/concurrency.md` | The four concurrency fixes, with before and after numbers |
| `knowledge/resume.md` | Resume bullets and interview answers, every number traced to a measurement |

## The reasoning

`knowledge/concurrency.md` covers the concurrency work: four problems, the fix for each, and
before and after numbers, including the target I missed.

`knowledge/learned.md` is the part I would actually point at. It records why each decision was
made, what the alternatives were, and where measurements contradicted the hypothesis, including:

- Four embedding models compared, where the theoretically stronger one measurably lost
- Why the first fix for the recall problem was wrong, and how a flat curve proved it
- Why a cache changes what error handling means: a bad answer is stored and replayed, so a
  single failure becomes permanent
- Several bugs that shared one shape, a default that was correct in one phase and silently
  became a defect in the next

## History

| Stage | What it did |
|---|---|
| 1 | Linear search, the eval harness and the 194-pair set, the 0.80 threshold |
| 2 | HNSW from scratch, benchmarked against linear search; two real bugs found at scale |
| 3 | The cache router: embed, search, hit or miss, store |
| 4 | An AWS deployment (Lambda, then EC2 with DynamoDB and Bedrock), built and measured, later removed |
| 5 | A local base: SQLite store, pluggable LLM (Ollama by default), no cloud |
| 6 | The concurrency layer: lock, metrics, single-flight, load shedding |

Stages 1 to 4 are in `knowledge/learned.md`, which is a log of reasoning and so still describes
the removed AWS code; its status note and section 25 say what changed. Stage 6 is in
`knowledge/concurrency.md`. The old code is in git history.
