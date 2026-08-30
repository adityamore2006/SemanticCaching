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

One long-lived process holds everything: the embedding model, the HNSW graph, and the hit/miss
decision. Three environment variables are the whole difference between running it on a laptop
and running it deployed.

```
client -> FastAPI ---> embed (mpnet, in-process)
                  ---> search HNSW (in RAM)
                  ---> similarity >= 0.80 ?
                         yes -> return stored answer        (no model call)
                         no  -> Bedrock (Claude Haiku) -> store -> return
```

| Variable | Unset | Set |
|---|---|---|
| `CACHE_TABLE_NAME` | in-memory, nothing persists | DynamoDB, plus graph snapshot to disk |
| `LLM_MODEL_ID` | a miss returns 501 | Bedrock answers, capped per day |
| `RESET_TOKEN` | reset endpoint disabled | one-command reset to the seeded corpus |

Deployed on AWS via CloudFormation: EC2, DynamoDB on-demand, least-privilege IAM, an EventBridge
nightly auto-stop, and a budget action that stops the instance on overspend. Roughly $2-3/month
at demo usage, since the instance is stopped when idle.

**A serverless design was built first and then rejected**, which is documented rather than
buried. Lambda recycles idle containers on its own schedule, so an unpredictable multi-second
cold start lands on an arbitrary request, and that is the one thing a latency-focused cache
cannot absorb. Keeping a Lambda warm costs about $21.90/month against $2.30/month for an
instance that stops when idle.

## Running it

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

cd src && ../.venv/bin/uvicorn api:app --port 8000
```

Open http://127.0.0.1:8000 for the demo page, or http://127.0.0.1:8000/docs for the API. The
cache seeds itself with 67 curated support answers on first start, so it can demonstrate a hit
immediately.

```bash
.venv/bin/python -m pytest tests/ -q          # 82 tests
.venv/bin/python scripts/try_queries.py       # labelled behaviour tour
.venv/bin/python eval/threshold_sweep.py      # regenerate the threshold table
```

No AWS account is needed for any of the above. Without `LLM_MODEL_ID` a genuinely new question
returns 501 rather than a placeholder, which is deliberate: a cache writes down whatever it is
given and replays it forever, so a fabricated answer would become permanent.

## Layout

| Path | |
|---|---|
| `src/hnsw.py` | The HNSW index, written from scratch |
| `src/linear_search.py` | Brute force, kept permanently as the ground truth HNSW is graded against |
| `src/cache_router.py` | The hit/miss decision the whole project builds toward |
| `src/api.py` | HTTP entry point, the deployed system |
| `eval/` | Threshold sweep, recall/latency benchmarks, collision verification |
| `data/eval_pairs.json` | The 194-pair labeled evaluation set |
| `infra/` | CloudFormation stack and deployment guide |
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

Complete and running. The miss path calls Bedrock, which is pending an AWS quota grant on a new
account; until then a miss is refused rather than answered, and everything else (embedding,
search, threshold, hit/miss, storage, reset) works.
