# Learned: talking points for interviews

Running log of the non-obvious decisions, tradeoffs, and debugging lessons from building this project. Organized by topic, not chronology. The point of this file is "what would I actually say if someone asked me to defend a decision or walk through a bug," not a feature log.

---

## 1. Architecture: the hot-swap seam (strategy pattern)

**Decision:** `VectorIndex` is an abstract base class with three methods (`insert`, `search`, `__len__`). `LinearIndex` and `HNSWIndex` both implement it. A `factory.py` (`create_index(kind, dim=...)`) is the *only* place that imports both concrete classes; everything else (eval harness, later the cache router) depends only on the abstract type.

**Why this is worth saying out loud in an interview:** this is the textbook strategy pattern, and the reason it's here isn't decoration — it's what makes Phase 4 possible. The eval harness that grades HNSW's recall against linear search's exact answers is *the same code* for both indexes; only the string passed to `create_index` changes. Without the interface, adding HNSW later would mean touching every call site that used `LinearIndex` directly.

**The concrete enforcement mechanism, worth knowing precisely:** Python's `ABC` + `@abstractmethod` isn't just documentation — `LinearIndex(dim=2)` would raise `TypeError` at construction time if any abstract method were missing. That's a real, tool-enforced guarantee, not a convention people can silently drift from.

**Talking point:** "I kept the brute-force index permanently instead of deleting it once the real algorithm existed, because it's the ground truth I grade the approximate algorithm against — recall@k means nothing without an exact answer key to compare to."

---

## 2. The numpy performance decisions in linear search

**Normalize once at insert time, not at every comparison.** Cosine similarity is `(a·b)/(|a||b|)`. If every stored vector and the query are pre-normalized to unit length, the denominator becomes `1`, so similarity search collapses to a single matrix multiply (`self.vectors @ query`) instead of a division per comparison. This is a real optimization, not just tidiness — it moves the expensive `norm()` division from "once per query per stored vector" to "once per insert."

**Two-stage top-k selection: `argpartition` then `argsort`.** Fully sorting `n` similarities to get the top `k` is `O(n log n)` and wastes effort sorting items you're about to discard. `np.argpartition` finds the split point between "top k" and "everything else" in `O(n)`, without ordering either side. Only the tiny `k`-sized winner set then gets sorted (`O(k log k)`). At small `k`, this is a real asymptotic win, and it matters specifically because Phase 4 compares linear search's *speed* against HNSW's as data scales — using the actually-correct primitive here keeps that comparison honest.

**Why ids and vectors are two separate structures, not one dict.** Numpy's speed comes from operating on one contiguous, uniformly-typed block of memory (`float32` here). String ids can't live in that same array without numpy falling back to slow, boxed `dtype=object` storage — which throws away the entire benefit. So: `self.vectors` (numpy, fast math) and `self.ids` (plain Python list) are kept as parallel arrays, linked only by matching index position. This is a classic tradeoff: speed at the cost of an implicit invariant (the two lists must always be appended to together — nothing enforces that structurally, just insert() discipline).

---

## 3. Embeddings: what `dim` actually means

`dim` (384 for `all-MiniLM-L6-v2`) is a property of the *model*, completely decoupled from the length of the input text. A two-word query and a three-paragraph query both produce exactly 384 numbers — the model compresses arbitrary-length text into a fixed-size numeric summary of meaning, that compression is the entire point of an embedding model. Worth being precise about this distinction in an interview: dimensionality is a design choice of the encoder, not a measurement of the input.

---

## 4. The eval design choice: shared index, not isolated pairwise checks

**Decision:** the threshold-sweep harness doesn't check each `(query_a, query_b)` pair in isolation. It inserts every unique `query_a` into *one shared index* (simulating "everything the cache has already answered"), then checks each `query_b` against the whole thing.

**Why this is the more rigorous design, and worth explaining:** a real cache doesn't ask "does this match its intended partner" — it asks "does this match *anything* currently cached." Checking pairs in isolation can't catch a false match against a *different* stored entry. The shared-index design can, and did (see the debugging section below) — it's what surfaced two real bugs in the eval data itself.

**Test set structure, and why the middle bucket carries the most weight:** 45 pairs split into paraphrase (should hit), near-miss (should miss, hard negative — shares vocabulary/topic but means something different), and unrelated (should miss, easy negative — sanity check). The near-miss bucket is where the actual engineering problem lives: too loose a threshold serves a *confidently wrong answer* to a question that only sounded like a cached one, with no visible failure signal. That's the precision/recall tradeoff the whole project is built to measure, not a guess.

---

## 5. Debugging story #1: telling a real bug apart from a platform quirk

**Symptom:** `RuntimeWarning: divide by zero / overflow / invalid value encountered in matmul` on a plain `self.vectors @ query` cosine-similarity computation, with clean, unit-normalized, non-zero float32 inputs.

**The wrong move (resisted):** wrap it in `warnings.simplefilter("ignore")` and move on. That silences the *symptom* whether or not the *cause* is real — you'd have no idea afterward whether your similarity scores were secretly `NaN`.

**The actual method — separate "is my data broken" from "is the tool broken":**
1. Checked the real inputs directly (`np.isnan`, `np.linalg.norm` on every stored row) — clean.
2. Reproduced the warning with **completely fresh random data** that had never touched the project (`np.random.randn(...)`, manually unit-normalized). It fired identically. This is the key move: if a warning fires on data you know is clean, the bug can't be in your data — it has to be underneath you, in the library or platform.
3. Checked `np.show_config()` — numpy was built against Apple's **Accelerate** BLAS backend, which has a documented history of spurious floating-point warnings on certain matmul shapes even when the math is correct.
4. Confirmed the *output*, not just the absence of a crash: `np.isnan(result).any()` → `False`, `np.isinf(result).any()` → `False`. Only after that did we suppress the warning, with `np.errstate(...)` scoped exactly to that line, plus a comment recording why — so a future reader doesn't have to redo the investigation.

**Talking point:** "A warning is a claim about what happened, not proof. I verify the claim against the actual output before deciding whether to trust it or suppress it."

---

## 6. Debugging story #2: the 1.00 / 0.00 metric as a failure signal, not a result

This is the most reusable lesson in the whole project, worth having a sharp, ready answer for.

**What happened:** the first threshold sweep showed `unrelated_wrong_rate = 1.00` at *every* threshold from 0.50 up through 0.95, flat across the whole range. After a first fix, the metric became properly graded (0.89 down to 0.00 as the threshold tightened) — behaving the way a real metric should. But "behaving correctly" and "correct" turned out to be different things: it still took a second pass, checking raw similarity magnitudes directly rather than the threshold-classified rate, to find the second bug.

**Why a flat, perfectly unmoving metric across a wide parameter sweep is itself the red flag (bug #1):**
Real semantic similarity scores are continuous and messy — a genuine model, on genuine varied data, produces a *graded* response that shifts as you move the threshold. A metric glued to exactly `1.00` across the *entire* sweep range means the outcome isn't actually a function of the threshold at all — which means the measurement isn't testing what it claims to test. Here it meant every "unrelated" pair was scoring a near-perfect similarity regardless of threshold, impossible for any threshold sweep to fix, because the underlying similarity itself was pinned at ~1.0.

**Why a well-behaved, properly graded metric still isn't proof of correctness (bug #2):**
Once the metric started decreasing smoothly with threshold, it *looked* healthy — no flat, impossible pattern to spot from the summary table alone. The problem was invisible at the aggregate level entirely; it only showed up by sorting the raw per-pair similarities and noticing a few sitting suspiciously high in absolute terms, regardless of where any threshold happened to fall.

**How the actual causes were found — never trust the aggregate, always drop to raw rows:**
- For the `1.00` case: printed every individual `unrelated`-category result and found similarity `1.0000001` — not "very similar," *identical*. Traced it to a verbatim string collision: a "should be unrelated" query was, character-for-character, the exact text already stored in the shared index under a different anchor.
- After the first fix, `unrelated_wrong_rate` dropped to `0.00` at the strictest threshold (0.95) — but that number only counts pairs that *crossed* the threshold, it says nothing about how close the ones just under it got. Sorting all 9 unrelated pairs by raw similarity (ignoring thresholds entirely) surfaced three sitting suspiciously high: `0.938`, `0.840`, `0.836`. Two "unrelated" texts scoring 93.8% similar is a problem regardless of where any threshold sits. Investigating the worst one: its `query_b` ("restore a file deleted by accident") turned out to be a near-exact semantic paraphrase of a *completely different* stored anchor ("recover a file I accidentally deleted") — not a verbatim string match this time (that was bug #1), but the same underlying mistake one layer more subtle: reusing another anchor's meaning while trying to invent fresh "unrelated" content.

**The general principle:** an aggregate metric can be lying to you in either direction — badly broken and obviously wrong, or suspiciously perfect and quietly wrong. The only way to tell the difference is to look at the individual data points the aggregate was computed from. This is also, notably, a direct mirror of the actual thesis of the semantic-cache project itself (from the project brief): *"too loose, the cache confidently serves a wrong answer... with no visible failure signal."* The same failure mode that threatens the system being built also threatens the measurement of that system, if you stop at the summary number.

**Root cause, once found:** ground-truth label contamination in self-authored eval data. When you write your own "should not match" examples by hand, it's easy to accidentally reuse or paraphrase content that already exists elsewhere in the set under a different label. The fix wasn't a code change at all — it was checking each new negative example against *all* existing anchor topics for semantic overlap, not just against its own paired query.

---

## 8. Model comparison: a "more sophisticated" model isn't automatically better

Tried four embedding configurations against the same 45-pair eval, all through the exact same harness:

| model | dim | hit@0.7 | near_miss_wrong@0.7 |
|---|---|---|---|
| all-MiniLM-L6-v2 (baseline) | 384 | 0.61 | 0.22 |
| all-mpnet-base-v2 | 768 | 0.67 | 0.22 |
| BAAI/bge-large-en-v1.5, no instruction prefix | 1024 | 1.00 | 0.83 |
| BAAI/bge-large-en-v1.5, with its recommended query prefix | 1024 | 0.78 | 0.72 |

**mpnet won, and BGE lost, even though BGE is trained with hard-negative mining specifically for the "similar but different" distinction this project cares about.** Two separate lessons here:

**Using a model against its documented calling convention silently breaks it.** BGE's model card specifies its instruction prefix ("Represent this sentence for searching relevant passages: ") is for the query side of an asymmetric retrieval pair. Run it unprefixed and everything, including pairs that should be obviously unrelated, gets compressed into a high, poorly-separated similarity range (`unrelated_wrong_rate` hit `1.00` at low thresholds, something that never happened with MiniLM or mpnet). Adding the prefix measurably helped, but only partially closed the gap.

**A model's training objective matching your problem in theory doesn't guarantee it wins in practice.** BGE's hard-negative training was reasoned to be a better theoretical fit than mpnet's general STS objective for exactly this "near-miss" problem. Empirically, it was worse across the board, likely because BGE was tuned on large-scale web retrieval data, a different domain than short, FAQ-style, near-duplicate query pairs. The talking point: reasoning about *why* a model should help is necessary but not sufficient, run the actual eval before trusting the theory.

**Talking point:** "I didn't just pick the model with the best-sounding training story, I measured four options on the actual task and let the numbers pick the winner, including a case where my own hypothesis about what should help turned out to be wrong."

---

## 9. Confirming the decoupling actually works, and its real limits

The `Embedder(model_name)` swap was tested for real, four times, without touching `factory.py`, `linear_search.py`, or `vector_index.py` at all, just a different string. That's the interface-based decoupling from section 1 paying off in an eval context, not just a code-review nicety.

**Two honest limits, worth stating precisely rather than overclaiming "fully decoupled":**
- **Free before data is stored, not after.** Swapping the embedding model is a zero-cost config change *today*, because nothing is persisted yet. Once Phase 5 caches real query vectors, those vectors are tied to whatever model produced them (different dimensionality, different vector space). Changing models later means re-embedding everything already stored, not just flipping a config value.
- **The threshold is tied to the model, not portable across a swap.** The whole point of section 8's table is that `0.70` means something different on every model tested. Swapping models later means rerunning the threshold sweep and re-deriving an operating threshold, the old number doesn't carry over.

---

## 10. Scaling the eval set: automated verification beats manual eyeballing

45 pairs (18 per bucket) was too small to trust: moving a single pair shifts a reported rate by ~5-6 points, and the earlier hit-rate numbers turned out to be optimistic partly because of that. Expanded to 194 pairs across 67 topic anchors (paraphrase, near-miss, unrelated for each), spanning ten sub-domains of the fictional product instead of one narrow slice, to get a sample where single-pair luck stops dominating the result.

**At this scale, manually re-reading the growing set for contamination (the process that worked, barely, at 45 pairs) stops being reliable.** Built `eval/verify_pairs.py` instead: it embeds every anchor and every `query_b` with the real model and checks each one against the *entire* anchor set, not just its own pair, catching:
- verbatim collisions (a `query_b` identical to some stored anchor's text)
- anchor-vs-anchor collisions (two "different" anchors that are actually the same real-world question)
- a paraphrase whose best match across the whole set isn't its own anchor (a sign the wording is ambiguous, not that the model is wrong)

**First automated pass on the 194-pair set found three real issues**, all authoring mistakes, not model failures: "reset password" and "change password" were worded closely enough to be functionally the same anchor (0.859 similarity); "I'd like to stop my plan" (meant as a paraphrase of "cancel subscription") was ambiguous enough to read as closer to "downgrade my plan"; "adjust what a collaborator is allowed to do" read as closer to "share a folder externally" than to "change permission level." Reworded each for precision, reran the checker, zero issues. This is the same debugging principle as section 6 (check the raw data, not the aggregate), just running proactively during construction instead of reactively after a bug ships.

**What the larger sample actually changed, and what it didn't:**
- **Hit rate at every threshold dropped meaningfully** (56%→40% at threshold 0.75), confirming the 45-pair number really was optimistic, not just "small but still valid." This is the concrete payoff of catching a small-sample problem before treating a number as final.
- **The model ranking held.** Reran `bge-large` (prefixed) against the same 194 pairs specifically to check whether more data would flip the earlier verdict. It didn't: at matched safety levels, mpnet still beats it (mpnet reaches `near_miss_wrong=0.12` at threshold `0.75` with `40%` hit rate; BGE needs threshold `0.80` to reach that same safety level and only delivers `30%` hit rate there). A conclusion that survives a 4x larger, independently-verified sample is a much stronger claim than the same conclusion at 45 pairs.

**Talking point:** "I didn't just build a bigger test set, I built a way to trust it, an automated check run against the same failure modes that already burned me once, and used the larger sample specifically to stress-test whether my earlier conclusion actually held up."

---

## 11. The final operating threshold: 0.80, chosen on an explicit asymmetric cost model

The last real decision Phase 2 required: pick one threshold, on `all-mpnet-base-v2`, over the 194-pair set.

**The cost model, stated explicitly rather than left implicit:** a missed cache hit costs one extra LLM call, no correctness risk. A wrong cache hit silently serves the wrong answer with full confidence, no visible failure signal. Those costs are not symmetric, so the threshold decision should weight false-positive risk (`near_miss_wrong_rate`) more heavily than raw hit rate, not split the difference evenly. This is not an assumption invented after the fact, it is the exact risk the project brief names as the reason the threshold problem exists at all.

**The numbers that framed the decision:**

| threshold | hit_rate | near_miss_wrong_rate |
|---|---|---|
| 0.70 | 0.55 | 0.21 |
| 0.75 | 0.40 | 0.12 |
| 0.80 | 0.22 | 0.07 |
| 0.85 | 0.04 | 0.04 |

**Chose 0.80, explicitly not 0.85.** Under a safety-weighted cost model, the instinct is to keep tightening, but that instinct has a floor: at `0.85`, hit rate collapses to `0.04`, the cache would almost never fire, which isn't "extra safe," it's "no longer functioning as a cache." The marginal safety gain from `0.80`→`0.85` (`0.07`→`0.04`) is small and not worth trading away the system's actual reason to exist. `0.80` is the point where safety is prioritized hard (93% of near-misses correctly rejected, `0.00` unrelated false positives) while the system still does its job often enough to matter.

**Talking point:** "I didn't pick the safest possible threshold, I picked the safest threshold that still leaves the system doing its job, and I can point to the exact number where that tradeoff stops being worth it."

---

## 12. HNSW from scratch: the design forks, and why each side was picked

**The core reframe worth leading with:** HNSW's insert and search aren't two independent features built one after another, they're two callers of the same routine. The paper calls it SEARCH-LAYER, a greedy beam search within a single layer: search uses it to find a query's actual nearest neighbors, insert uses the identical routine to find *candidate* neighbors for a new node, then decides which candidates become permanent edges. Building "just search first" isn't actually possible, there's nothing to search until insert has built a graph, and insert can't find neighbors without the same traversal search will later reuse. That shared-primitive structure is the first thing worth explaining before anything about layers or parameters.

**Layer assignment is probabilistic, not designed per-node:** `level = floor(-ln(u) * mL)` where `mL = 1/ln(M)`. With `M=16`, `mL ≈ 0.36`, which makes `P(level >= 1) = 1/16`, `P(level >= 2) = 1/256`, and so on, each layer up roughly `1/M` as populated as the one below. This produces a skip-list-like shape without anyone deciding it explicitly: layer 0 ends up holding essentially every node (dense, complete graph), each layer above is a sparser "highway." That's what makes search sublinear, a greedy walk at the sparse top layers covers large distances in very few hops, then the walk drops down to refine locally once it's close. Concrete micro-example: a random draw of `u=0.01` gives `level=1`, `u=0.001` gives `level=2` (the node is now a member of layers 0, 1, *and* 2, not just 2).

**Neighbor selection during insert: picked SELECT-NEIGHBORS-SIMPLE over the paper's diversity-aware heuristic, deliberately, not by default.** Once insert's beam search (`ef_construction=200`) finds a wide candidate pool for a new node, the paper offers two ways to pick which `M` (or `M_max0=2M` at layer 0) candidates become real edges: keep the closest `M` outright, or the more sophisticated variant that skips a candidate if it's already closer to an *already-selected* neighbor than it is to the query itself (favoring edges that spread across different directions instead of clustering). Chose the simple version: easier to trace, easier to defend precisely in an interview, and the diversity heuristic's extra recall benefit shows up mainly on larger, more clustered datasets than this project's eval set exercises. Worth being able to name the alternative and why it wasn't picked, that's a stronger answer than not knowing it exists.

**One detail that's not a design choice, it's the paper's fixed default, worth knowing precisely if asked:** layer 0 gets `M_max0 = 2*M` neighbors, every other layer gets `M`. Reason: layer 0 is the only layer guaranteed to contain every node in the graph, so it's carrying most of the graph's real connectivity and gets a bigger edge budget unconditionally.

**Validation, and being honest about what it does and doesn't prove:** ran the real 194-pair eval set (not synthetic data) through HNSW instead of `LinearIndex` via the same `threshold_sweep.py` functions, using the locked `all-mpnet-base-v2` model. Result: 194/194 top-1 matches identical to linear search, similarity scores agreeing to 6 decimal places, and the threshold table matching the locked baseline exactly at 0.70/0.75/0.80/0.85. The honest caveat, worth stating unprompted rather than waiting to be asked: with only 67 anchors and `ef_search=50`, the bottom-layer beam search is wide enough to see nearly the entire graph, so near-perfect agreement with brute force is the *expected* result at this scale, not yet proof the approximation is doing anything interesting. This run validates that the graph construction and greedy traversal are *correct*. Whether HNSW actually trades a small amount of recall for a real speed win only becomes visible at Phase 4, once the dataset is large enough that `ef_search` can no longer see everything.

**Talking point:** "I can point to the exact line where my implementation deviates from the paper's default (simple vs. heuristic neighbor selection), explain why, and I validated correctness before ever measuring speed, because a fast wrong answer isn't a result."

---

## 13. Debugging story #3: an O(n^2) bug that only 194 real pairs could never expose

Phase 4 needed far more data than 194 hand-authored pairs to actually see HNSW's approximation and speed tradeoffs, so the plan was: perturb the 117 real, verified anchors (67 from Phase 2 + 50 new cross-domain ones added specifically to widen embedding-space coverage, see section 14) with small calibrated Gaussian noise, scale to 1,000 / 10,000 / 50,000 synthetic vectors, and measure recall@1 and query latency for both indexes as `n` grows.

**Symptom:** `LinearIndex`'s insert time didn't scale like brute-force insert should. `n=1,000 -> 0.13s`, `n=10,000 -> 21.45s` (a 160x jump for a 10x data increase), and the full run's `n=50,000` tier finished at **443.8 seconds** for insert alone -- longer than HNSW's insert at the same scale, which makes no sense for an operation that should just be "append to a list."

**Root cause:** `LinearIndex.insert` did `self.vectors = np.vstack([self.vectors, normalized])` on every single call -- reallocating and copying the *entire* array from scratch each time, making one insert `O(n)` and `n` inserts `O(n^2)` overall. This bug existed since Phase 1 and was invisible there: Phase 1's correctness tests use a handful of hand-picked vectors, and Phase 2's eval only ever builds an index of 67 unique anchors. `O(n^2)` is indistinguishable from `O(n)` at `n=67`; it only becomes visible once `n` is large enough for the squared term to dominate, which nothing before Phase 4 ever exercised.

**The fix:** accumulate inserted vectors in a plain Python list (`_vectors`, true O(1) amortized append, same trick Python's own list uses internally), and only materialize the contiguous numpy matrix `search()` actually needs lazily, cached, and invalidated on the next insert. This doesn't change what gets computed, just when: `n` inserts now cost `O(n)` total instead of `O(n^2)`, and the one-time matrix rebuild before the next search is `O(n)`, same as it always should have been.

**Verified, not assumed fixed:** re-timed inserting the same three scales after the fix -- `n=1,000: 0.13s -> 0.006s`, `n=10,000: 21.45s -> 0.064s`, `n=50,000: 443.8s -> 0.37s`. Full test suite (23/23) still passes unchanged, confirming the fix altered performance, not behavior.

**Talking point:** "This is the same lesson as the eval-data contamination bugs, just on the systems side instead of the measurement side: a correctness test suite built at small scale can't catch a complexity bug, because small `n` can't distinguish `O(n)` from `O(n^2)`. The fix wasn't intuition, it was building a benchmark large enough that the bug became visible, then re-measuring to prove the fix actually worked instead of assuming it did."

---

## 14. Phase 4 results: the recall/speed tradeoff, measured, not asserted

Data: the 117 real, collision-verified anchors (see section 13's opening) perturbed with `sigma=0.018` Gaussian noise (empirically calibrated against a real embedding, landing parent-similarity at `0.85-0.95`, matching where genuine Phase 2 paraphrases sit) into synthetic datasets of 1,000 / 10,000 / 50,000 vectors. Both indexes built on the identical dataset at each scale; 200 held-out queries (fresh perturbations of the same 117 anchors, never inserted) measured against both.

| n | linear query | hnsw query | recall@1 | hnsw insert | linear insert |
|---|---|---|---|---|---|
| 1,000 | 0.112ms | 0.541ms | 98.0% | 3.56s | 0.01s |
| 10,000 | 1.070ms | 0.577ms | 64.5% | 23.04s | 0.09s |
| 50,000 | 4.617ms | 1.083ms | 54.0% | 113.60s | 0.32s |

**The query-latency crossover is real and empirically located, not assumed.** Before measuring, the honest expectation (see section 12/dataset-planning discussion) was that our from-scratch HNSW is pure Python (heapq, dict lookups) competing against `LinearIndex`'s single vectorized numpy matmul, so the scale where HNSW's better asymptotic complexity actually overcomes numpy's raw constant-factor speed was a genuine unknown. The data answers it: brute force wins below ~a few thousand vectors, HNSW wins above that, and the gap widens fast -- by 50,000, linear's query time grew 41x over the 50x data increase (consistent with real `O(n)`), while HNSW's grew only 2x (real sub-linear growth).

**Recall degrades exactly as predicted once `ef_search` stops covering most of the graph, and it's reported honestly, not softened.** With `ef_search` fixed at the paper's default `50`, recall@1 fell 98.0% -> 64.5% -> 54.0% as `n` grew. At 50,000 vectors, `ef=50` is a genuinely small slice (0.1%) of the graph, and HNSW gets the wrong top-1 answer nearly half the time under unmodified default parameters. This isn't a flaw to hide, it's the actual tradeoff the whole project exists to make legible: the fix, not yet built, would be growing `ef_search` alongside `n` to hold a target recall, which is exactly the kind of operational knob a real deployment would tune.

**One more asymmetry worth naming out loud: HNSW trades a much more expensive build for a much cheaper query.** `113.6s` to insert 50,000 vectors into the graph vs. `0.32s` for linear's (now-fixed) insert. That's the right trade specifically *for a cache*: queries vastly outnumber inserts (cache misses) once the system is warmed up, so paying more at insert time to make every subsequent query faster is the correct shape of tradeoff for this exact use case, not a universal win for HNSW over brute force in general.

**Talking point:** "I didn't just implement HNSW and claim it's faster, I built a benchmark large enough to find the actual crossover point, and I found it in both directions, the query-speed win and the recall cost, at the same time, with the same measurement."

---
