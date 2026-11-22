# Concurrency: four problems, four fixes, measured

The cache worked for one request at a time. It had never been tested with many at once. This
note records what broke under concurrent traffic, what was changed, and what the numbers were
before and after.

**What these numbers are.** Every measurement uses a fake model that waits 3 seconds, so slow
model calls are repeatable and free. The embedder, the HNSW index, SQLite and uvicorn are real.
It was run on one 10-core laptop, with synthetic traffic and no real users. Reproduce it with
`.venv/bin/python scripts/loadtest.py`.

## Results

| | Before | After |
|---|---|---|
| 20 identical new questions at once: model calls | 20 | **1** |
| 20 identical new questions at once: entries written | 20 | **1** |
| 20 different new questions at once: total time | 3.3 s | 3.3 s (unchanged, on purpose) |
| A cached question while 60 slow questions run | **3.1 s** (3.06 to 3.26 s over 3 runs) | **0.11 s** (110 to 111 ms over 3 runs) |
| What happens to those 60 slow questions | all 60 answered, over about 6 s | 24 answered, 36 refused at once with a 503 |
| A cached question with no load | 17 ms | 17 ms |
| Forced thread switching on a young cache, 40 trials | 15 trials lost entries or crashed (7 `KeyError`) | 0 |
| Tests | 95 | 154 |

The 0.11 s is not the 17 ms of an idle server. See "What is not fixed" below.

## How the server handles requests

This decided what the fixes had to be, so it was checked first rather than assumed.

- uvicorn runs **one process** with one event loop.
- Every plain `def` endpoint runs on a pool of **40 worker threads**. Verified: anyio 4.12.1
  reports `total_tokens = 40`.
- The pool is **shared** by every endpoint. It is not split by hit versus miss. A request
  waiting on a slow model holds its thread the whole time.
- All 40 threads share one router, one index and one set of counters.

Two consequences. Shared data needs locks, because 40 threads really do run against it. And a
limit on the pool would cap cache hits too, so protecting hits has to be done in our own code.

## Fix 1: shared state needed a lock

**Problem.** Two threads could read the same id from the counter and overwrite each other's
entry. A search could also run in the middle of an HNSW insert and read a half-updated graph.
The insert sets `max_level` one line before `entry_point`, so a search could pair the new level
with the old entry node, which has no neighbors at that level. That raised a `KeyError`.

**How rare it is.** At Python's default thread-switch interval (5 ms) the windows are
microseconds wide, and 200 trials were clean. With the interval forced to 1e-6 s, a young cache
(3 entries) went wrong in 15 of 40 trials. A young index changes its top layer often, so this is
a startup problem, not a steady-state one. It is a latent bug, not a frequent one.

**Fix.** One lock in `CacheRouter` guards the id counter and every index search and insert. The
store write stays outside it, because a SQLite commit is slow and has its own lock. Embedding
and the model call stay outside too.

**Tradeoff.** Searches take turns. Measured with 2,000 entries, a search takes 0.7 ms and an
insert 1.9 ms. The alternative, a readers-writer
lock, is more code for a gain that cannot be measured at this scale.

## Fix 2: measure first

The old counters (`state["hits"] += 1`) had no lock, counted no model calls and recorded no
timing. `/metrics` now reports counters, an in-flight gauge and p50/p95/p99 over the latest
2,000 samples per kind.

**One thing worth knowing.** A handler that times itself cannot see a request that is still
waiting for a worker thread. Under the flood it would have reported "hits take 15 ms" while
clients waited 3 seconds. So `queue_wait` is stamped by an async dependency that runs on the
event loop before the handler is queued. In the flood it read 3.4 s at p95, in line with the
3.1 s clients saw.

`/health` and `/metrics` are `async def`, so they run on the event loop and still answer when
all 40 threads are busy.

## Fix 3: identical misses share one model call

**Problem.** Twenty identical new questions made 20 model calls and wrote 20 duplicate entries.

**Fix.** `SingleFlight` keeps a table of the questions being worked on. The first request for a
question calls the model. Identical requests that arrive meanwhile wait and share its answer.

**Why not one big lock around misses.** It would also stop the duplicates, but it makes every
miss wait for every other miss. Twenty different 3-second questions would take 60 seconds
instead of 3. The rule that decided this: never hold a lock while waiting on something slow.
SingleFlight's lock covers only the table lookup, so different questions run in parallel and
the 3.3 s above did not change.

**Details that matter.**
- "Same question" means identical text, ignoring case and spacing. It does not mean similar
  text. Paraphrases are caught by the normal cache lookup once the first answer is stored.
- The leader looks in the cache again before calling the model. Without that, a request that
  searched just before an earlier flight finished would pay for an answer already stored.
- A model failure goes to every waiter once and is not remembered. The next request retries.
- Requests that shared an answer are counted as `coalesced`, not as hits. They did not find a
  cache entry.

## Fix 4: a cap on the miss path

**Problem.** With all 40 threads waiting on slow model calls, a cached question could not get a
thread. It waited about 3 seconds for a lookup that takes 17 ms.

**Fix.** `ConcurrencyLimit` admits up to `MAX_CONCURRENT_MISSES` (default 24) requests into the
miss path and refuses the rest at once with a 503 and `Retry-After`. Cache hits return before
the limit and are never refused. With 24, at least 16 of the 40 threads are always free for
hits.

**Decisions.**
- **Refuse, don't queue.** A request that waits for a slot still holds a thread, so queuing
  would use up the capacity the limit protects.
- **Waiters count toward the cap**, because they hold threads too. The cost: 30 identical
  questions at once would see 6 refused, although one model call would serve them all. I chose
  protecting hits over serving every identical request.
- **503, not 429.** This means "the server is full", not "you sent too much". It is counted as
  `shed`, separately from model failures.
- **The model timeout is now a setting** (`LLM_TIMEOUT_SECONDS`, default 30, was a fixed 60). It
  is a socket timeout: it bounds each wait for the server's reply, not a hard wall-clock
  deadline. For a normal non-streaming reply those are the same.
- New settings are parsed strictly: `0`, `-3`, `2.5`, `nan` and `inf` stop the boot. A timeout
  of `nan` or `inf` would otherwise silently mean "never time out".

## What is not fixed

- **A cached question during a flood takes 110 ms, not 17 ms.** I set a target of under 100 ms
  and missed it. Every request, hits included, must be embedded before the service knows
  whether it is a hit, so the miss-path cap cannot help here. The embedder is CPU-bound, and I
  measured it directly: 16 ms for one embedding alone, 219 ms each when 10 run together, 455 ms
  each at 40. A burst of 60 requests slows everything that has to embed. Capping embedding
  concurrency would not help, because the total work is the same. Real options are shedding
  earlier on total requests in flight, a smaller embedder, or batching. Not done.
- **Single-flight matches text, not meaning.** Two phrasings of one question arriving together
  make two model calls.
- **One process only.** Each process would build its own index and mint its own ids against the
  same SQLite file, so ids would collide. A lock cannot fix that.
- **`/admin/reset` swaps the router while requests may be running.** A request holding the old
  router can write to the shared store after the reset. Not addressed.
- **The fix 4 tests were not run red first.** The first run failed on a missing module rather
  than on the behavior. The load test is the behavioral evidence for that fix. Fixes 1 and 3
  were run red on the old code (for fix 3, `assert 20 == 1`).
- **The numbers use a fake model.** A real Ollama model adds its own concurrency limits and
  variable latency, which a single cap does not model.

## Talking point

"A cache that works one request at a time can fail in four different ways under load: corrupted
shared state, duplicate model calls, slow requests starving fast ones, and no way to see any of
it. I measured each first, then fixed them one at a time with a before and after number. The
result I'm least satisfied with is the one I missed my own target on: 110 ms instead of 100.
I measured why: the embedder takes 16 ms alone but 455 ms each when 40 run at once, so the
remaining bottleneck was CPU, not the miss path."
