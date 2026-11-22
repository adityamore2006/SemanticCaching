# TODO

State today: runs fully locally. Embedder, HNSW index, threshold, hit/miss, a pluggable LLM
(Ollama by default), SQLite persistence, seeding and reset all work. The AWS deployment code was
removed (it is in git history, and its reasoning is in `knowledge/learned.md`).

## Next: infrastructure layer

Real bugs reproduced in the audit. Done: the router lock (fix 1), `/metrics` plus
`scripts/loadtest.py` (fix 2) and single-flight (fix 3: N identical concurrent misses make
one LLM call) and the miss-path cap, model timeout and load shedding (fix 4: cached
questions during a flood of 60 slow ones went from ~3.1 s to ~0.1 s). Still open:

- Embedding runs on every request before the cap applies, so a burst of 60 simultaneous
  requests still contends for CPU (~110 ms for a hit instead of ~17 ms). Fixing it means
  bounding the embedder, not the miss path.
- No usage cap on LLM calls (the old DynamoDB one was removed; rewrite it atomically).

Stretch: circuit breaker, semantic single-flight, crash-consistent persistence.

## Later

- A verification stage for borderline matches (cross-encoder or LLM judge) to cut the 7% false
  accepts, and a second `CacheStore` backend (Supabase/Postgres) behind the same interface.
- Settle the API contract and error codes; keep `src/static/index.html` in step.
- Docs pass: `knowledge/resume.md`, `Progress.md` and `PLAN.md` still describe the AWS
  deployment, and the resume bullets need correcting (the 25% cost-saving claim is unmeasured).
