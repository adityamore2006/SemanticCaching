# Semantic Cache: resume and interview reference

Everything here is copy-paste ready and every number is traceable to a measurement in
`knowledge/learned.md`. Nothing depends on Bedrock, so all of it is accurate today.

**Rule for using these:** always attach the scale to a recall number. "99.5% recall@1"
unqualified is a claim an interviewer can puncture in one question; "99.5% recall@1 at 1,000
vectors" is the same number, fully defensible, and closes the question before it is asked.

---

## Resume bullets (primary, 5 lines)

**Semantic Caching Layer for LLM APIs** — Python, AWS (EC2, DynamoDB, Bedrock, CloudFormation)

- Built an HNSW approximate-nearest-neighbor index from scratch, no vector library, reaching **99.5% recall@1 at 1,000 vectors** and **1.5x faster queries than brute force at 10,000**
- Derived the cache's similarity threshold from a **194-pair labeled evaluation set** using an explicit asymmetric cost model, holding false-positive matches to **7%** while keeping hit rate high enough to be useful
- Diagnosed two algorithmic bugs that small-scale tests could not surface: an **O(n²) insert (443s to 0.37s at n=50,000)** and unrecoverable upper-layer graph routing (**recall 54% to 65.5%**), each confirmed by measurement before and after the fix
- Benchmarked four embedding models on the target task and found a theoretically stronger model performed worse in practice; reported the measured result over the hypothesis
- Deployed via CloudFormation with least-privilege IAM, scheduled auto-stop, and budget-triggered shutdown; **rejected a serverless design after measuring cold-start cost** ($21.90/mo to keep Lambda warm vs $2.30/mo for a stop/start instance)

## Resume bullets (tight, 3 lines)

- Built an HNSW vector index from scratch, no library, **99.5% recall@1** and **1.5x faster than brute force at 10k vectors**; found and fixed an **O(n²) insert (443s to 0.37s)** and a graph-routing bug worth 11 points of recall
- Chose the cache's similarity threshold from a **194-pair evaluation set** with an explicit asymmetric cost model, holding wrong-answer rate to **7%**, and can name the exact scale where brute force still beats the custom index
- Deployed on AWS with CloudFormation, least-privilege IAM, and layered spend controls; **rejected a Lambda architecture on measured cost** after building it

## One-line version

Built a semantic cache for LLM APIs around a from-scratch HNSW index (99.5% recall@1 at 1k
vectors), with the similarity threshold derived from a 194-pair labeled eval, deployed on AWS
via CloudFormation.

---

## The numbers, and where each comes from

**Index performance** (`all-mpnet-base-v2`, `ef_upper=8`, learned.md §17):

| n | linear query | HNSW query | recall@1 |
|---|---|---|---|
| 1,000 | 0.055 ms | 0.371 ms | **99.5%** |
| 5,000 | 0.234 ms | 0.287 ms | 86.5% |
| 10,000 | 0.632 ms | 0.419 ms | 70.0% |

Read this honestly: brute force **wins outright** at 1,000 (faster *and* effectively exact).
The crossover lands in the low-to-mid thousands. Saying so is a stronger answer than a chart
showing only the favorable half.

**Threshold selection** (194 pairs: 67 paraphrase, 67 near-miss, 60 unrelated, learned.md §11):

| threshold | hit rate | wrong-match rate |
|---|---|---|
| 0.70 | 0.55 | 0.21 |
| 0.75 | 0.40 | 0.12 |
| **0.80 (chosen)** | **0.22** | **0.07** |
| 0.85 | 0.04 | 0.04 |

Chose 0.80 and explicitly *not* 0.85: at 0.85 the hit rate collapses to 4%, which is not
"safer", it is a cache that no longer functions.

**Model comparison** (learned.md §8): `all-mpnet-base-v2` beat `BAAI/bge-large-en-v1.5` on this
task, even though BGE is trained with hard-negative mining aimed at exactly this near-miss
problem. The hypothesis was wrong and the measurement stood.

**Scale:** ~2,000 lines of source, ~1,100 lines of tests, 82 tests passing, 28 documented
decisions in `learned.md`.

---

## Talking points, ranked by how well they land

**1. "I can name the exact scale where my own algorithm loses to brute force."**
At n=1,000, linear search is both faster (0.055 ms vs 0.371 ms) and effectively exact. HNSW's
reason to exist does not appear until the low thousands. Volunteering the unfavorable half is
what makes the favorable half believable.

**2. "The threshold is a cost decision, not a tuning knob."**
A missed hit costs one extra API call. A wrong hit serves a confidently incorrect answer with
no failure signal. Those costs are not symmetric, so the threshold weights false positives
more heavily rather than splitting the difference.

**3. "My first fix was wrong, and I know because I measured it."**
Recall was falling at scale. The obvious cause was too narrow a search beam, so I swept
`ef_search` 32x wider and recall stayed **flat**. That result killed the hypothesis and pointed
at the real cause: a zero-backtracking greedy walk through the upper layers, which no amount of
widening the final beam can recover from. Confirmed with a targeted diagnostic before changing
any code.

**4. "A cache changes what error handling means."**
Elsewhere a bad upstream response is one bad response. In a cache it is written down and
replayed to every future query that matches, so one failure becomes permanent. My miss path
validates before it writes, and what the tests assert is not that an error was returned but
that the cache was left byte-for-byte unchanged.

**5. "I built the serverless version, measured it, and threw it away."**
Lambda recycles idle containers on its own schedule, so an unpredictable multi-second cold
start lands on an arbitrary request, which is the one thing a latency-focused cache cannot
absorb. Keeping a Lambda warm costs more than simply running a small instance. I had already
rejected provisioned concurrency on cost without pricing the alternative it was losing to.

**6. "Most of my bugs were silent, and they rhymed."**
A default that is correct in one phase becomes a defect in the next, with nothing announcing
the transition: an embedding model default that outlived the threshold it matched, an `or` that
discarded an empty-but-valid cache store, a stub LLM whose placeholder got cached and served
back as a confident hit.

---

## Questions to expect, with answers

**"Why not just use FAISS or pgvector?"**
For production, I would. The point was to build the index, not to consume one. I kept
brute-force linear search permanently as the exact ground truth that makes the approximate
index's recall number mean anything.

**"Recall at 10,000 is only 70%. Isn't that bad?"**
At the scale this operates at it is 99.5%. Recall degrades as a fixed-width search covers less
of a growing graph, which is the tradeoff HNSW exists to make. 50,000 was a stress test, not a
target, and it earned its keep by surfacing two real bugs. The operational fix is scaling
`ef_search` with n, which is a knob a real deployment tunes.

**"How much did it actually save?"**
Every hit is an API call that provably did not happen, and the service counts them. I can give
the threshold tradeoff precisely from the eval; what I have not yet done is run realistic
traffic end-to-end to produce a single measured savings figure. That is the next thing I would
build. *(Honest gap. Better to name it than to be caught.)*

**"What would you do differently?"**
Build the benchmark earlier. Two of my worst bugs, an O(n²) insert and a graph-routing failure,
were invisible at the scale my correctness tests ran at. Small n cannot distinguish O(n) from
O(n²).

---

## Project description (for a portfolio page or a longer application)

A proxy between an application and an LLM API that recognizes when a new question *means* the
same as one already answered, and serves the stored response instead of paying for a new call.

The engineering problem is the similarity threshold. Too loose and the cache confidently serves
a wrong answer with no visible failure signal. Too strict and it barely saves anything. That is
a precision/recall tradeoff that has to be measured, so I built a 194-pair evaluation set of
true paraphrases and deliberately deceptive near-misses, swept the threshold across it, and
chose an operating point using an explicit cost model rather than intuition.

The vector index underneath is a hierarchical navigable small world graph written from scratch,
with brute-force search kept alongside it permanently as the ground truth its recall is graded
against.

---

## Links

- Repository: https://github.com/adityamore2006/SemanticCaching
- Decision log: `knowledge/learned.md`, 28 entries covering the reasoning, the tradeoffs, and
  the debugging stories behind every number above
