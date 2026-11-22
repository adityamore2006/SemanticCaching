# TODO

State today: runs fully locally. Embedder, HNSW index, threshold, hit/miss, a pluggable LLM
(Ollama by default), SQLite persistence, seeding and reset all work. The AWS deployment code was
removed (it is in git history, and its reasoning is in `knowledge/learned.md`).

## Next: infrastructure layer

Real bugs reproduced in the audit. Done: the router lock (fix 1) and `/metrics` plus
`scripts/loadtest.py` (fix 2). Still open:

- No single-flight: N identical concurrent misses make N LLM calls and N duplicate entries.
- No isolation: slow misses can starve hits (60 concurrent misses on a 3 s LLM raised hit
  latency from ~35 ms to ~2.9 s). Needs a bulkhead, timeouts and load shedding.
- No usage cap on LLM calls (the old DynamoDB one was removed; rewrite it atomically).

Stretch: circuit breaker, semantic single-flight, crash-consistent persistence.

## Later

- A verification stage for borderline matches (cross-encoder or LLM judge) to cut the 7% false
  accepts, and a second `CacheStore` backend (Supabase/Postgres) behind the same interface.
- Settle the API contract and error codes; keep `src/static/index.html` in step.
- Docs pass: `knowledge/resume.md`, `Progress.md` and `PLAN.md` still describe the AWS
  deployment, and the resume bullets need correcting (the 25% cost-saving claim is unmeasured).
