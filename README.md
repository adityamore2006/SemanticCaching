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

Needs [Ollama](https://ollama.com) with a model pulled for the miss path (any model works):

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
ollama pull qwen2.5:7b-instruct

cd src && ../.venv/bin/python -m uvicorn api:app --port 8000
```

Open http://127.0.0.1:8000 for the demo page, or http://127.0.0.1:8000/docs for the API. On
first start it seeds 67 curated support answers into `data/cache.db`, so a hit works
immediately. A question not in the cache goes to the LLM, and the answer is cached.

| Variable | Default | |
|---|---|---|
| `CACHE_DB_PATH` | `data/cache.db` | SQLite file |
| `LLM_BACKEND` | `openai_compat` | or `fake` (no model, for tests) |
| `LLM_MODEL` | `qwen2.5:7b-instruct` | any model the server has |
| `LLM_BASE_URL` | `http://localhost:11434/v1` | Ollama; any OpenAI-compatible server works |
| `LLM_API_KEY` | none | for hosted providers |
| `OPERATING_THRESHOLD` | `0.80` | |
| `RESET_TOKEN` | unset | enables `POST /admin/reset` |

```bash
.venv/bin/python -m pytest tests/ -q          # no Ollama needed
.venv/bin/python scripts/try_queries.py       # labelled behaviour tour of a running service
.venv/bin/python scripts/loadtest.py          # concurrency scenarios against a slow fake model
.venv/bin/python eval/threshold_sweep.py      # regenerate the threshold table
```

`GET /metrics` returns request counts, model calls, an in-flight gauge and recent latency
percentiles as JSON (`GET /stats` is the smaller view the demo page uses). Run one server
process: the cache lives in that process's memory, so extra `--workers` would each build their
own index and mint colliding ids against the same SQLite file.

If the LLM is unreachable, a miss returns 503 (or 504 on a timeout) and nothing is cached,
because a cache replays whatever it stores.

## Layout

| Path | |
|---|---|
| `src/hnsw.py` | The HNSW index, written from scratch |
| `src/linear_search.py` | Brute force, kept permanently as the ground truth HNSW is graded against |
| `src/cache_router.py` | The hit/miss decision the whole project builds toward |
| `src/sqlite_cache_store.py` | Durable storage; the index is rebuilt from it at startup |
| `src/llm.py` and backends | The pluggable LLM interface (`openai_compat_llm.py`, `fake_llm.py`) |
| `src/api.py` | HTTP entry point |
| `src/metrics.py`, `src/metered_llm.py` | Counters and latency percentiles; a wrapper that counts every model call |
| `scripts/loadtest.py` | Repeatable concurrency scenarios, so each fix can be judged by measurement |
| `eval/` | Threshold sweep, recall/latency benchmarks, collision verification |
| `data/eval_pairs.json` | The 194-pair labeled evaluation set |
| `TODO.md` | What is planned next, and what is deliberately deferred |
| `knowledge/learned.md` | **28 documented decisions, tradeoffs, and debugging stories** |

## The reasoning

`knowledge/learned.md` is the part I would actually point at. It records why each decision was
made, what the alternatives were, and where measurements contradicted the hypothesis, including:

- Four embedding models compared, where the theoretically stronger one measurably lost
- Why the first fix for the recall problem was wrong, and how a flat curve proved it
- Why a cache changes what error handling means: a bad answer is stored and replayed, so a
  single failure becomes permanent
- Several bugs that shared one shape, a default that was correct in one phase and silently
  became a defect in the next

## Status

Runs fully locally: embedding, search, threshold, hit/miss, an Ollama-backed miss path, SQLite
persistence and reset all work, and the tests pass. The router is safe to call from many
threads, and `/metrics` plus `scripts/loadtest.py` measure the service under load. Not built
yet: single-flight, timeouts, concurrency limits and load shedding (see `TODO.md`). Known
limitations until then: simultaneous identical misses each call the LLM, and a burst of slow
misses makes cached questions wait behind them.
