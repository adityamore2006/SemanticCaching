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

## 15. Debugging story #4: 54% recall wasn't a ceiling, it was an unturned dial pointed at the wrong problem

Reaction to the raw Phase 4 number (recall@1 falling to 54% at n=50,000): "that isn't good enough." Correct instinct, worth trusting instead of explaining away -- but the fix wasn't obvious, and the first hypothesis turned out wrong, which is the more useful part of this story.

**First hypothesis (reasonable, wrong): `ef_search` is too narrow, widen it.** `ef_search` is the standard operational knob real HNSW deployments scale up alongside data specifically to hold a target recall, and it was fixed at the paper's default (50) throughout Phase 4 on purpose, to isolate how recall decays as `n` grows relative to an unchanged beam. So the natural next move was sweeping it. Since `ef_search` is a pure query-time parameter (doesn't touch graph structure), the n=50,000 graph only needed to be built once and reused across the sweep.

**Result: flat.** `ef=50 -> recall 0.650`, `ef=1600` (32x wider) `-> recall 0.655`. Query latency roughly doubled for that 32x wider beam and recall barely moved. This ruled out "beam too narrow" as the cause -- a real result worth reporting exactly as measured, not massaged into agreeing with the hypothesis.

**Second hypothesis, built from the first result, not assumed: the problem is upstream of the beam entirely.** `ef_search` only controls the final wide search at layer 0. Getting there requires a single-path, zero-backtracking greedy walk through the upper layers (`ef=1`, exactly as the paper's own K-NN-SEARCH algorithm specifies -- not an implementation bug). If that walk commits to the wrong neighborhood on one bad hop, no amount of widening the *final* beam recovers it, because `ef_search` never touches that walk at all.

**Verified before fixing anything:** for each of the 87 misses (n=50,000, `ef_search=50`), checked whether HNSW's wrong answer shared the same parent anchor as the true match (harmless -- same underlying topic, different near-duplicate sibling) or a genuinely different one. **60.9% were a different anchor entirely** -- not close calls between similar siblings, real wrong answers. Combined with the flat `ef_search` curve, this confirmed the upper-layer routing, not beam width, was the actual bottleneck.

**The fix:** added `ef_upper` (default 8, not from the paper), replacing the strict `ef=1` upper-layer descent in both `insert()`'s phase A and `search()` with a small beam -- enough width to consider a few alternatives at each upper layer instead of committing to one path outright. Applied to both call sites, not just `search()`, since insert's phase A has the identical structural weakness and a badly-routed insert wires a new node into the wrong part of the graph in the first place, compounding the problem for every future query.

**Re-verified, not assumed fixed, at every scale:**

| n | recall@1 before | recall@1 after | hnsw query before | hnsw query after |
|---|---|---|---|---|
| 1,000 | 98.0% | 98.0% (already near ceiling) | 0.541ms | 0.340ms |
| 10,000 | 64.5% | 67.0% | 0.577ms | 0.382ms |
| 50,000 | 54.0% | **65.5%** | 1.083ms | 1.002ms |

Also re-ran the exact 194-real-pair Phase 3 correctness check (small n, where routing failures are rare regardless) to confirm the change didn't regress anything there: still 194/194 exact agreement with linear search. And re-checked the miss composition at n=50,000: the "genuinely wrong cluster" rate (the number that actually matters for cache correctness, not the stricter "exact same id" recall metric) dropped from 26.5% of all queries to 18.0%.

**Honest framing, not declared "solved":** the fix produced a real, evidence-backed improvement, largest exactly where it mattered most (the largest, most stressed graph), at no query-latency cost. It did not fully close the gap -- 65.5% recall@1 (18% still landing in a genuinely wrong cluster) at n=50,000 is better, not production-ready. Untried next levers: sweeping `ef_upper` itself (only tested at one value, 8), or combining with a higher `M` for denser upper-layer connectivity.

**Talking point:** "The first fix I reached for was wrong, and I know that because I measured it before assuming it worked, a flat recall curve across a 32x wider beam. That result is what pointed me at the actual bottleneck, upper-layer routing with no backtracking, which I confirmed with a targeted diagnostic before touching any code, then re-measured after the fix at every scale, not just the one that looked worst."

---

## 16. Right-sizing the benchmark, and the AWS architecture decision

Two decisions made together, because they're the same underlying constraint viewed from two angles.

**Capped the realistic benchmark scale at 1k-10k, deliberately walked back from n=50,000.** 50,000 distinct previously-cached topics isn't a realistic size for most real semantic caches (the same GPTCache-precedent scale cited in the project brief operates on the order of hundreds to low-thousands of distinct topics before saturating), and on real infrastructure, hosting cost scales directly with index size. This isn't discarding the 50k work, it's re-scoping what it's *for*: the 50k stress test already earned its keep, it's what surfaced both the O(n^2) insert bug (section 13) and the unrecoverable-routing recall bug (section 15). Neither of those would have shown up at 10k. But going forward, 1k-10k is the range worth optimizing and reporting as "the" performance story, since it's the range that's actually economical to run in production.

**AWS architecture: fully serverless, chosen specifically to minimize idle cost, not for novelty.** API Gateway (HTTP API, the cheaper tier) -> Lambda -> DynamoDB (on-demand billing, not provisioned) -> Bedrock (pay per token, called only on cache miss) -> CloudWatch. Every piece of this is pay-per-request; nothing runs, or costs anything, when nobody's calling it. The Bedrock-only-on-miss piece is the project's actual cost-savings claim made literal: every cache hit is a Bedrock call that provably didn't happen, and CloudWatch can turn that into a real dollar figure over time, not an estimate.

**Explicitly ruled out: OpenSearch Serverless**, despite the project brief listing "compare from-scratch HNSW against OpenSearch Serverless" as an optional stretch idea. It carries a real minimum OCU-hour billing floor even sitting idle, which directly contradicts the economical-to-host goal. If that comparison ever happens, it'd be a one-time, immediately-torn-down exercise, not something left running alongside the rest of the stack.

**The two decisions reinforce each other, worth stating explicitly:** a small, realistic cache (decision 1) is also what makes a Lambda-native architecture viable at all (decision 2) -- a small HNSW graph is cheap to rebuild from scratch on a cold start (n=1,000 built in ~2 seconds, section 15's benchmark), so there's no need for an always-on server just to keep a large graph warm in memory. Choosing realistic data size and choosing serverless infrastructure aren't two separate cost-cutting moves, they're the same insight applied twice.

**Sequencing principle for Phase 5/6, carried over from the linear-before-HNSW pattern:** build and validate the cache-routing logic (hit -> return stored response, miss -> call Bedrock, store result) locally first, fast and free to iterate on, then wrap it for Lambda deployment once it's proven -- not developing the routing logic directly against live, billed AWS resources.

**Talking point:** "The cost-consciousness wasn't an afterthought bolted onto the architecture, it came from the same realism check that shaped the benchmark itself: a cache doesn't need 50,000 entries to prove its value, and neither does the hosting bill."

---

## 17. Linear vs HNSW: head-to-head, hard metrics (1k-10k, post-fix)

The canonical comparison table, both indexes on the identical dataset, `all-mpnet-base-v2`, `ef_upper=8`:

| n | linear query | hnsw query | recall@1 | hnsw insert | linear insert |
|---|---|---|---|---|---|
| 1,000 | 0.055ms | 0.371ms | 99.5% | 2.07s | 0.01s |
| 5,000 | 0.234ms | 0.287ms | 86.5% | 9.16s | 0.02s |
| 10,000 | 0.632ms | 0.419ms | 70.0% | 16.46s | 0.04s |

**Honest counterpoint first, worth leading with rather than burying: at n=1,000, brute force wins outright.** Linear is faster per query (0.055ms vs 0.371ms) *and* effectively exact (99.5% recall). HNSW's whole reason to exist doesn't show up yet at this size -- a real deployment starting from an empty cache would spend real time in the regime where the from-scratch algorithm work isn't paying for itself yet.

**The crossover is real and lands in the low-to-mid thousands, not a single fixed n.** Linear still edges out HNSW at n=5,000 (0.234ms vs 0.287ms) but loses by n=10,000 (0.632ms vs 0.419ms, HNSW 1.5x faster). Consistent with the earlier 50k-scale run, where the same crossover showed up between 1k and 10k -- the exact n shifts slightly run to run (layer assignment is randomized, not seeded), but the *location* of the crossover, low thousands, is consistent across independent runs.

**The tradeoff being bought, stated as a trade, not a win:** HNSW's insert cost grows real and fast across this range (2.07s -> 16.46s, an 8x increase for a 10x data increase) against linear's now-fixed, near-instant insert (0.01s -> 0.04s). Recall correspondingly degrades (99.5% -> 70.0%) as the fixed-width search has more graph to cover. HNSW is not a strictly-better replacement for linear search in this system, it's a deliberate trade: slower, more expensive graph construction and a real (measured, not hidden) recall cost, in exchange for faster queries once the dataset is large enough for that to matter -- which is exactly the right trade for a cache, where queries vastly outnumber inserts once warmed up (section 14).

**Talking point:** "I can name the exact scale where my own from-scratch algorithm loses to brute force, and I can explain precisely why, not just where it wins. That's a more credible systems story than a chart that only shows the favorable side."

---

## 18. Phase 5: cache routing, and cleaning up an inconsistency before building on top of it

**Found dead code that contradicted an already-documented principle, and removed it before adding more.** Both `LinearIndex` and `HNSWIndex` had an unused `metadata` dict (`id -> payload`, added early with an "e.g. cached response" comment, never actually read or written to by anything). It would have been tempting to just repurpose it as the cache's storage layer, less new code to write. But `vector_index.py`'s own docstring and section 1 already establish, in writing, that response storage must stay independent of the index specifically so index kind and storage backend can each change without touching the other. Using the index's own dict would have quietly undone that. Removed the dead field from both classes instead of building on top of it.

**`CacheStore` is a new ABC (`src/cache_store.py`), same hot-swap shape as `VectorIndex`:** `put`/`get`/`__len__`, one `InMemoryCacheStore` implementation now, `DynamoDBCacheStore` (Phase 6) implementing the identical contract against real persistence later, swapped the same way linear and HNSW already swap.

**`CacheRouter` (`src/cache_router.py`) is the actual system the whole project has been building toward:** embed the query, search the index, and if similarity clears the locked `0.80` threshold (section 11), return the stored response with no LLM call; otherwise call the LLM, insert the new vector, store the response, return it fresh. Every prior phase is a component this class wires together, not a separate deliverable.

**`index_kind` has no default, on purpose.** Section 17's hard numbers showed HNSW only wins on query speed once the cache has warmed up past roughly 5,000-10,000 entries -- defaulting the router to HNSW would have baked in an assumption the project's own measurements don't unconditionally support. Forcing the caller to state a choice keeps that an honest, visible decision rather than a hidden default.

**The LLM call is a stub (`call_llm`), deliberately, matching the sequencing already decided in section 16.** `llm` is a constructor parameter specifically so swapping in real Bedrock later is passing a different callable, not touching the routing logic at all.

**Verified end-to-end with the real embedding model, not just unit tests.** The numbers originally recorded here (`0.8260` for the paraphrase, `0.0991` for the unrelated query) turned out to be measured on the *wrong model* and have been corrected in section 24 -- the demo was silently running `all-MiniLM-L6-v2` while the threshold it was being checked against came from `all-mpnet-base-v2`. The current, correct demo output is:

```
MISS  sim=n/a     'Can I merge two accounts into one?'
HIT   sim=0.9404  'Is it possible to combine my two separate accounts?'
MISS  sim=0.1019  'Does this integrate with Slack?'
``` Unit tests (`tests/test_cache_router.py`) use a small `FakeEmbedder` with hand-picked, exactly-computed cosine similarities (e.g. `[0.9, sqrt(1-0.9^2)]` gives exactly `0.9` similarity to `[1,0]`) instead of the real model, matching the same fast, dependency-free testing philosophy already established for the index contract tests -- real-model verification and unit-test correctness are two separate, deliberate checks, not one relied on to cover the other.

**Talking point:** "Before adding a new component, I checked whether it would contradict a design decision I'd already written down and defended, found one that did, quietly, and removed it rather than build the new piece on top of an inconsistency."

---

## 19. Phase 6 design, decided but not yet built: the stateless-Lambda-vs-stateful-graph problem

**The problem, and why it's not obvious from the code as it stands:** `CacheRouter`'s vector index and `CacheStore` both currently live in plain Python memory. That's correct for a long-running local process, but Lambda containers are stateless between invocations and can be recycled at any time. Deploying the current code as-is would silently lose the entire cache -- both the graph and every stored response -- on every cold start, making the system nonfunctional in production (every query would miss forever, since nothing would ever persist). `CacheStore` was already designed to be swappable specifically for this (`DynamoDBCacheStore` was always the planned Phase 6 piece), but the vector index itself has no equivalent persistence story yet.

**Three options considered:**
1. **Rebuild the graph from DynamoDB on every cold start**, keep it in memory for that container's lifetime. Reuses `insert()` exactly as it already exists, no new serialization code.
2. **Serialize the built graph to S3**, reload (deserialize, not rebuild) on cold start. Faster cold starts than rebuilding, but adds real complexity: something has to decide when to re-persist the graph after every insert-on-miss, and concurrent Lambda invocations updating it introduce a real consistency question that doesn't exist with option 1.
3. **Provisioned concurrency**, pay AWS to keep at least one container permanently warm so the in-memory graph effectively never gets discarded.

**Decided: option 1, rebuild-from-DynamoDB-on-cold-start.** Chosen for the same reason linear-before-HNSW and simple-before-heuristic-neighbor-selection were chosen earlier in this project: it's the simplest thing that actually works, reuses code that's already built and tested, and only pays for added complexity (option 2) if measurement later proves it's actually needed. It also directly cashes in the section 16 decision to cap realistic scale at 1k-10k -- that's exactly what makes a full rebuild cheap enough to do on every cold start in the first place.

**Explicitly rejected: provisioned concurrency (option 3).** It's a standing, 24/7 cost for reserved compute capacity, not pay-per-use -- roughly $10+/month just to hold a container open, before any real traffic. That directly contradicts the near-zero-idle-cost serverless architecture already committed to in section 16, for the same reason OpenSearch Serverless was ruled out there.

**Cost estimate for the chosen approach** (ballpark, not quoted AWS pricing -- worth checking the Pricing Calculator before trusting a real bill, but the order of magnitude holds), at a realistic demo/portfolio traffic level of 100 cold starts/month, each rebuilding a 10,000-entry graph:
- Lambda compute: 100 x 16.5s (the actual measured HNSW insert time at n=10,000, section 17) x 1GB = 1,650 GB-seconds -- inside AWS's perpetual 400,000 GB-second/month free tier, effectively $0.
- DynamoDB reads for the rebuild: 100 x 10,000 reads = 1,000,000 read-request-units, roughly $0.25 total at on-demand pricing.
- Realistic total: **$0-1/month** at this usage level.

**The honest cost that isn't in dollars: cold-start latency.** Rebuilding a 10,000-entry graph takes ~16.5 seconds, a real, one-time penalty per cold container, not something to hide behind the cheap dollar figure. Worth measuring once this is actually built, and worth treating graph-serialization-to-S3 (option 2) as the documented next optimization if that latency turns out to be a real problem in practice -- not something to build preemptively without that evidence.

**Reaffirmed, not a new decision: `LinearIndex` never gets deployed.** `CacheRouter`'s `index_kind` has no default specifically so this stays an explicit, visible choice (section 18) -- the Lambda handler will always construct it with `"hnsw"`. Linear search stays exactly what it's always been in this project: the local ground-truth tool HNSW gets graded against, never a production component.

**Status: design decided and documented, implementation deliberately deferred to its own session.** Local tooling is ready (`aws-cli` 2.36.29 and SAM CLI 1.165.0 already installed via Homebrew), but AWS credentials are not yet configured locally (`aws configure` still needs to be run, directly by the user in their own terminal, never through an assistant's tool calls, so access keys never pass through a transcript). Concrete next build artifacts, in order: `DynamoDBCacheStore` (implementing the existing `CacheStore` contract), a Lambda handler wrapping `CacheRouter` with the cold-start rebuild logic described above, then the SAM template (API Gateway + Lambda + DynamoDB) to actually deploy it.

**Talking point:** "I designed the persistence strategy around a constraint I'd already measured, not a general best practice, the rebuild is only cheap because I'd already capped the realistic scale at 1k-10k earlier for cost reasons. The two decisions weren't made independently, the second one only works because of the first."

---

## 20. Phase 6: the deployment constraint that invalidated a locked decision

**The constraint, discovered while sizing the Lambda package:** `all-mpnet-base-v2` plus torch is roughly 2GB. A zip-packaged Lambda allows 250MB unzipped. So the embedding model that every measured number in this project was derived on cannot be deployed the way the rest of the system can.

**Two real options, and why the cheaper-looking one wasn't free:**
1. **Container-image Lambda** (10GB limit): keeps mpnet, keeps the locked `0.80` threshold, keeps every Phase 2 number valid. Costs a Docker/ECR build step and a slower cold start, on top of a cold start that already rebuilds the graph.
2. **Embed via Bedrock instead** (Titan Text Embeddings V2): keeps the Lambda a small, fast zip. But it is a different model, therefore a different vector space, therefore **the `0.80` threshold does not transfer** and has to be re-derived.

**Chose option 2, with the re-derivation treated as required work rather than a detail to skip.** This is the section 9 limit ("the threshold is tied to the model, not portable across a swap") arriving as a live consequence rather than a hypothetical. Worth being precise in an interview: the honest cost of the lighter deployment wasn't infrastructure complexity, it was invalidating a measured result and having to re-measure. The eval harness re-runs unchanged against the new backend (`--embedder bedrock`), which is the interface work from section 1 paying off a third time.

**A framing that quietly stopped being true, and shouldn't be repeated unqualified.** Section 16 says Bedrock is "called only on cache miss." With a local embedding model that was literally true: a hit cost nothing external. Embedding via Bedrock means **every** request now makes a Bedrock call, hit or miss, because the query's vector is what the hit/miss decision is made *from*. The cost claim survives, since the avoided call is the far more expensive generative one, but the sentence needs the qualifier.

---

## 21. Debugging story #5: the falsy cache store, or why `or` is not a null check

**The bug, in one line:** `CacheRouter.__init__` did `self.cache_store = cache_store or InMemoryCacheStore()`.

**Why that is fine for four phases and catastrophic in the fifth.** `CacheStore` implements `__len__` (it's a container, so that's the right interface). Python falls back to `__len__` for truthiness when `__bool__` is absent, so **an empty store is falsy**. Passing in a real, correctly-constructed `DynamoDBCacheStore` that simply had no rows yet meant `or` discarded it and substituted a throwaway in-memory dict.

**The failure mode this produces is the worst kind: silent and total.** A freshly deployed cache table is empty *by definition*. So on first deploy, every response would have been written to a dict that dies with the container, nothing would ever persist, every cold start would restore zero entries, and every query would miss forever. The system would return correct answers the whole time, at full LLM cost, while reporting itself healthy. There is no error, no exception, no failed assertion, no log line.

**How it was actually caught, which is the transferable part:** not by reading the code, and not by unit tests. The in-memory store's own tests passed, the DynamoDB store's tests passed, and the router's tests passed, because every one of them either injected a non-empty store or didn't care which store it got. It surfaced only when running the *real handler* end to end against a mocked DynamoDB and simulating a container recycle: the cold-start log line read `"restored_entries": 0` when it should have read 2. The bug lived in the seam between two components that were each individually correct and individually tested.

**The general lesson, and it's the same one as sections 6 and 13 from a third angle:** `or`-as-default is a truthiness test, not a null test, and the two only agree for types that are never legitimately empty/zero/false. For anything implementing `__len__`, they diverge exactly in the case that matters. `x if x is not None else default` says what was meant. Audited the other injectable dependencies in the codebase for the same pattern and converted them too, since a test double with a `__len__` would reintroduce it.

**Talking point:** "The tests that would have caught it didn't exist because each component was tested in isolation and each one was correct. What found it was exercising the actual deployment path, including the failure event I was designing around, a cold start, and checking a number I'd instrumented rather than assuming the log meant what I wanted it to."

---

## 22. Two ids can collide across a restart, and one of them wins silently

Related to section 21 and found in the same pass, but a separate defect worth its own note because the mechanism is different.

**Setup:** cache entries get ids from a counter, `q_0`, `q_1`, ... The counter is an instance attribute, so it restarts at `0` in a fresh Lambda container. The cold-start rebuild restores the *vectors* into the graph, but nothing was restoring the *counter*.

**The consequence:** a rebuilt container holding restored entries `q_0..q_5` would mint `q_0` again on its very next miss. The index would then contain two nodes claiming id `q_0` with different vectors, while the store held one response under it, the newer one. A later query matching the *old* `q_0` vector would be served the *new*, unrelated answer, above threshold, with high confidence.

**Why that's worth naming precisely:** this is the project's central failure mode ("the cache confidently serves a wrong answer to a question that only sounded like a cached one, with no visible failure signal") arriving through a completely different door. Phases 1-5 spent their effort making sure the *similarity threshold* couldn't produce that outcome. This one produces the identical outcome with a perfect similarity score, via id bookkeeping, and no threshold value defends against it.

**The fix, and where it belongs:** `CacheRouter.restore()` replays the vectors and advances the counter past the highest restored id. The first draft had the Lambda handler do this, reaching into `router._next_id` and parsing the `"q_<n>"` format itself. That put knowledge of the id format in two places and made the deployment layer responsible for an invariant of the router. Moving it onto the router keeps the format in one place and makes the rule enforceable by the component that owns it.

**Talking point:** "The threshold work protects against semantically wrong matches. This was an exactly-wrong match, similarity 1.0, caused by bookkeeping rather than embeddings, and no threshold tuning would ever have caught it. Same user-visible failure, completely different cause, which is why I fixed it at the layer that owns id generation instead of at the call site that noticed it."

---

## 22b. Open lead: pruning orphans nodes on clustered data, which may be the rest of section 15's recall ceiling

**Not investigated yet, recorded so it isn't lost.** Came up from the question "doesn't HNSW already evict the least-connected entries?"

**What was measured** (2,000 vectors, dim 64, `M=16` so `M_max0=32`):

| data shape | nodes stored | nodes with zero in-edges |
|---|---|---|
| uniform random | 2,000 / 2,000 | **0** |
| 8 tight clusters (sigma 0.05) | 2,000 / 2,000 | **245** |

`_prune()` only rewrites adjacency lists, never `self.vectors`, so a node is never removed from storage. But under heavy clustering, near-identical vectors compete for the same 32 slots and some lose *every* incoming edge. A node nothing points at is unreachable by traversal no matter where a search starts.

**Why this is worth chasing:** `eval/scale_dataset.py` generates its benchmark by perturbing 117 anchors with `sigma=0.018`, which is exactly this clustered shape. Section 15 documented a recall ceiling that was improved but never closed (65.5% at n=50,000) and listed sweeping `ef_upper` and raising `M` as untried levers. Orphaning would cap recall directly and independently of both, and it fits section 15's most puzzling result: recall stayed **flat** across a 32x wider `ef_search`. A wider beam cannot reach a node with no in-edges.

**Explicitly a hypothesis, not a finding.** Section 15's `ef_upper` fix produced a real, measured improvement, so upper-layer routing was genuinely part of the problem. Orphaning would be an additional cause, not a replacement. The test: measure orphan rate on the real eval data, then check whether the queries that fail recall are the ones whose true match is orphaned. Overlap would confirm it.

---

## 23. Serverless was the wrong shape, and the rejected option said so first

**The decision:** replaced Lambda with a single small EC2 instance (`t4g.medium`) that gets started before a demo and stopped after.

**What made Lambda wrong here specifically, stated as a property of this workload rather than a general complaint:** Lambda's cold start is not a one-time boot cost, it is a *recurring, externally-scheduled* one. AWS recycles idle containers whenever it likes, so a pause in a conversation can put a multi-second stall on the very next request. Most workloads absorb that. A cache cannot: its entire pitch is that it answers faster than the thing it is caching, and an unpredictable multi-second stall is the one failure mode that invalidates the pitch. This system's cold start is also unusually expensive on both counts that matter, loading an embedding model *and* rebuilding an in-RAM HNSW graph.

**The number that settled it, and where it came from:** section 19 rejected provisioned concurrency as "a standing 24/7 cost, roughly $10+/month." That reasoning was right but the figure was low and, more importantly, it was never compared against the obvious alternative. Pricing it properly: keeping 2GB of Lambda permanently warm is about **$21.90/month**, while a `t4g.medium` (2 vCPU, 4GB) is **$24.53/month** run continuously and about **$0.67** if it is only on for twenty hours of demos. For roughly the cost of keeping Lambda warm, an instance removes cold starts entirely, and switching it off makes it an order of magnitude cheaper. The rejected option had been carrying the answer since section 19; nobody had costed the thing it was being rejected in favor of.

**The constraint that disappears, which is the larger win:** Lambda's 250MB package limit was the *only* reason the design needed a container image, ECR, and a Docker build step, and the only reason embedding was ever pushed to a managed service (section 20). Removing Lambda deletes all of it: Lambda, API Gateway, ECR, the image, and the SAM packaging. `uvicorn` serves HTTP directly and a virtualenv has no ceiling to design around. A migration that is mostly deletion is a good signal about how the code underneath was factored.

**It also cashes in a deferral.** Section 19 considered snapshotting the built graph to S3 and deferred it as too complex: something had to decide when to re-persist, and concurrent Lambdas made consistency a real question. On one instance with a disk that survives stop/start, both problems evaporate, so the snapshot is a local file written at shutdown and read at boot, with rebuild-from-DynamoDB as the fallback when it is missing or unusable. The complexity that justified deferring it was entirely a property of the platform, not of the idea.

**What is deliberately kept rather than deleted:** the Lambda implementation and its measurements stay in git history and stay documented. Same reasoning as keeping `LinearIndex` after HNSW existed (section 1): the rejected option is what makes the chosen one's numbers mean anything.

**Talking point:** "I built it serverless, measured the cold start, and concluded serverless was the wrong shape for a latency-sensitive cache. The tell was that I'd already rejected provisioned concurrency on cost without ever pricing the alternative it was losing to, and when I did, the alternative was cheaper *and* removed the constraint that had forced three other workarounds."

---

## 23b. Deployed, and what the real numbers turned out to be

Measured on the live stack (`m7i-flex.large`, us-east-1), not projected.

**The stop/start cycle, which is the whole reason for this architecture:**

| stage | measured |
|---|---|
| `start-instances` to API answering | **31s** (mostly EC2 boot, not the app) |
| app startup, first ever boot (rebuild from DynamoDB) | 0.92s at 0 entries |
| app startup, later boots (reload from EBS snapshot) | **1.6s at 73 entries** |
| warm request, over the internet | ~85ms |
| warm request, same box | ~15ms |

The snapshot path did what it was designed to do: `restored_from=dynamodb` on the first boot, then `restored_from=snapshot entries=73` after a stop/start, with the graph written on systemd shutdown. Section 19 predicted ~16.5s to rebuild 10,000 entries from DynamoDB; the snapshot sidesteps that entirely, and the honest caveat is that 73 entries is far too few to have stressed either path. The comparison worth running later is snapshot-vs-rebuild at a few thousand entries.

**Identical similarity scores local and deployed** (0.8838, 0.7266, 0.8211, 0.7683 across the whole test tour). Not a coincidence worth shrugging at: it confirms the embedding model, threshold, and index all behave the same in both places, which is the thing a "works on my machine" deployment usually gets wrong.

**A constraint discovered by hitting it: AWS's Free Tier plan restricts which instance types an account may launch at all.** The first deploy failed at the instance with `The specified instance type is not eligible for Free Tier` and rolled the whole stack back. `t4g.medium` is not eligible. This is worth knowing generally: on a new account, instance type availability is a *plan* restriction, not just a pricing question, and `aws ec2 describe-instance-types --filters Name=free-tier-eligible,Values=true` is the authoritative list. The eligible set turned out to include `m7i-flex.large` at 8GB, more headroom than the 4GB originally planned, for about 22 cents a month more at demo usage.

**The rollback behaved correctly and is worth noting as a reason to use IaC at all:** four of five resources had already been created when the instance failed, and CloudFormation removed all of them. No orphaned table, role, or security group left billing quietly. Clicking the same setup together in the console would have left every successful piece behind.

**Cost after the full build, deploy, seed, test, and stop/start cycle: $0.00.**

---

## 23c. Cost guardrails, and three ways they can quietly not work

The deployed instance is $0.096/hr, so the real risk was never the rate, it was leaving it running by accident: about $70 over a forgotten month. Worth writing up because the interesting part is not the config, it's the three separate ways this nearly ended up as protection that only *looked* like protection.

**Start from the honest constraint: AWS has no hard spending cap.** Nothing refuses to spend past a number. Budgets alert and can trigger actions, but billing data lags by hours. Anyone describing a budget as a spending limit is describing something AWS does not offer. That constraint is what forces a layered answer rather than one setting.

| layer | mechanism | reaction time |
|---|---|---|
| nightly auto-stop | EventBridge schedule, 3am ET | deterministic, caps a forgotten instance at <24h (~$2.30) |
| budget action | stops EC2 at $5 | hours (billing lag) |
| budget alerts | email at 50/80/100% + forecast | hours |

Only the first is deterministic. The others are backstops for spend that isn't the instance.

**Wall-clock over a CPU-idle alarm, and the reasoning generalizes.** The obvious design is "stop it when it looks idle." But one embedding request costs roughly a CPU-second, so even active demoing averages under 5% CPU across 2 vCPUs. An idle-detection threshold would either fire in the middle of a demo or never fire at all, and there's no value that reliably separates the two. A wall-clock schedule has no such ambiguity, and 3am is a time nobody demos, so it cannot interrupt a session. **The lesson: when a signal cannot distinguish the two states you care about, stop tuning the threshold and pick a different signal.**

**Failure mode 1: a CloudFormation output made the stack un-updatable.** The template output the instance's `PublicIp` via `!GetAtt`. A stopped instance has no public IP, so the attribute fails to resolve and rolls back *the entire update*. Every attempt to add the guards failed on this, with an error naming the output rather than anything to do with the change being made. The output was also wrong on its own terms: the IP changes on every start, so an output holding one is stale immediately. Removed both it and the `ApiUrl` derived from it. **Generalizes to: don't put ephemeral attributes in outputs. They are read at every update, including updates made while the resource is in a state where the attribute does not exist.**

**Failure mode 2: "No changes to deploy" while the change silently didn't happen.** After testing the schedule at a near-future time, redeploying *without* `StopHour` reported `No changes to deploy` and kept the test value. `aws cloudformation deploy` reuses previously-set parameter values for anything omitted rather than falling back to the template default. Trusting that message would have left the auto-stop firing at 3:55pm daily, in the middle of exactly when a demo happens. **A success message describing a no-op is indistinguishable from a success message describing the change you wanted, unless you check the resulting state.** Same shape as section 6's aggregate metric problem, one layer up.

**Failure mode 3, avoided: an untested safety mechanism.** A guard you have not fired is not a guard, it is a belief. Tested it properly: started the instance, set the schedule a few minutes out, and watched it stop itself at 15:55:17 against a 15:55 schedule with no intervention. That test is also what surfaced failure mode 2, since restoring the real time is when the parameter-reuse behaviour appeared. **Testing the guard found a bug in the guard.**

**Both IAM roles behind the guards are scoped to `ec2:StopInstances` only.** Neither can start anything, so the worst a bug in either can do is stop the instance early. A guard that can only ever reduce spend is a guard that needs much less scrutiny than one that could increase it.

**Talking point:** "The protection I shipped is a wall-clock auto-stop, not an idle detector, because on this workload CPU can't tell 'in use' from 'forgotten'. And I only trust it because I fired it deliberately, which is how I found that CloudFormation had silently kept my test value while reporting no changes."

---

## 23d. A cache makes every bad answer permanent, which changes what "error handling" means

Found while auditing the miss path before switching the LLM from a stub to real Bedrock. Nothing here was broken *yet*, because a stub cannot fail.

**The reframe worth leading with:** in an ordinary service, a bad response from an upstream API is one bad response. In a cache, it is written down and replayed to every future query that matches it. The blast radius of a single failure is unbounded in time. That makes validating the response before storing it a correctness requirement, not defensive politeness.

**The specific hole: an LLM that succeeds and returns nothing.** An exception was always safe, because `route()` calls the LLM before it writes anything, so a raise leaves the cache untouched. But `BedrockLLM` built its answer with `"".join(block.text for block in response.content if block.type == "text")`, which returns `""` when there are no text blocks. And `"" is not None`, so the empty string would be stored, pass the orphan check on every later query, and be served as a **confident HIT with an empty answer, forever**.

**Truncation is the same shape and easier to miss.** A `max_tokens` cut-off returns HTTP 200 with real, well-formed, useful-looking text that simply stops mid-sentence. Nothing downstream can distinguish it from a complete answer. Cached, it becomes a permanently half-finished response. `stop_reason` was never checked.

**Fixed at both ends, deliberately.** `BedrockLLM` raises on refusal, on `max_tokens`, and on empty text -- raising is the correct channel precisely because the router already handles exceptions safely. Then `CacheRouter` *independently* rejects a falsy or whitespace-only response, because `llm` is a caller-supplied callable and the component that owns the cache should not trust an argument it was handed. Same reasoning as section 21, where the router refused to serve a `None` it found in the store rather than assuming storage was consistent.

**The guarantee that is actually worth asserting**, and what the tests check: after any failure -- upstream throttle, refusal, truncation, empty answer, daily cap -- **the cache contains exactly what it contained before**. Not "an error was returned", which is easy and shallow, but "nothing was written", which is the property that keeps a transient failure from becoming permanent.

**A related trap the same audit surfaced: entries written during development are indistinguishable from real ones.** 73 rows held `[stub response for: ...]`, and switching on the real model would not have removed them. They would keep being served as cache hits, which look like success, so nothing would ever surface it. Hence `scripts/purge_cache.py`. **Anything a cache stores while the backend is fake outlives the fake backend.**

**Talking point:** "Error handling in a cache is a different problem than error handling in a normal service, because a bad answer doesn't fail once, it gets stored and replayed. So my miss path validates the response before it writes, and the thing my tests assert isn't that an error was returned, it's that the cache was left byte-for-byte unchanged."

---

## 23e. The stub was a fabricated answer, and a cache makes fabrications permanent

Caught by using the demo rather than reading the code. Asking something not in the cache
returned a MISS, and asking the *same thing again* returned a HIT at similarity 1.0 -- serving
back `[stub response for: ...]` as though it were an answer.

**The cause was a default, not a bug in the routing.** `CacheRouter.__init__` had
`llm: Callable = call_llm`, a Phase 5 stub returning placeholder text. That was correct while
nothing persisted: the stub let the routing logic be built and tested for free, which was the
whole point of the sequencing decision in section 16. It became wrong the moment storage
existed, and nothing announced the transition. A miss cached the placeholder, and the next
identical question found it and served it.

**The general shape, and it is the third instance in this project:** a default that is
harmless in one phase silently becomes a defect in the next. Section 24 was the embedding
model defaulting to MiniLM after the threshold moved to mpnet. Section 21 was `or` substituting
an in-memory store once a real one could legitimately be empty. Each was a sensible default
that outlived the conditions that made it sensible.

**Fixed by removing the default entirely.** `llm` is now required, joining `index_kind` and
`embedder` for the same reason: these are choices too load-bearing to be assumed. Constructing
a router without one raises `TypeError` immediately rather than fabricating quietly. When no
model is configured the deployed service passes a callable that raises `LLMNotConfigured`, and
the API answers `501` -- not `503`, because the service is healthy and cached questions still
work; what is missing is a capability, not availability.

**What this costs, stated honestly:** without a model the system can no longer answer anything
new. That is the correct behaviour rather than a limitation to work around. Every part that
makes this a cache still demonstrates: the seeded corpus serves real hits, rewordings still
match, and near-misses are still refused. The one operation that stops is the one that
genuinely requires a model.

**Talking point:** "A stub is a fabricated answer. In a normal service that is fine, because it
is thrown away. In a cache it is written down and replayed, so the placeholder becomes a
permanent answer to a question nobody ever answered. I removed the default rather than making
it smarter, because the failure was that a default existed at all for something that important."

---

## 24. Debugging story #6: the verification that was measuring the wrong model

**Found while re-running the end-to-end demo after the platform change**, not by looking for it.

**Symptom:** the demo's paraphrase pair, documented in section 18 as hitting at `0.826`, now scored `0.783` and missed. Nothing about the routing logic had changed.

**The instinct to resist:** treat a number that moved as a regression and go looking for what broke in the code. Nothing had. The measurement had been wrong the whole time, and fixing an unrelated bug is what exposed it.

**Cause:** `Embedder()` defaulted to `all-MiniLM-L6-v2`, while the `0.80` threshold was derived on `all-mpnet-base-v2` (sections 8 and 11). `CacheRouter` fell back to that bare default whenever no embedder was passed, which is exactly what the demo did. So section 18's "verified end-to-end with the real embedding model" was real, but the real model it verified was **not the one the threshold belonged to**. Confirmed directly by scoring the same pair on both:

| model | paraphrase | unrelated |
|---|---|---|
| all-MiniLM-L6-v2 | 0.8260 | 0.0991 |
| all-mpnet-base-v2 | 0.7827 | 0.0462 |

The first row is exactly what section 18 recorded. The demo had been passing on the wrong model's numbers.

**The part that makes this worth writing down rather than quietly fixing:** the corrected behavior is not a regression, it is the eval's own documented result finally showing up in the demo. Section 11's table says mpnet at threshold `0.80` has a hit rate of `0.22`, so **78% of genuine paraphrases are supposed to miss** at this deliberately safety-weighted threshold. Measured directly against the 194-pair set: 15 of 67 paraphrase pairs clear `0.80`, which is `22%`, matching the table exactly. A demo pair that hits was never typical, it was cherry-picked without anyone noticing it had been cherry-picked *by the wrong model*.

**Fix, in two parts.** The default was corrected when the embedder became an interface (`SentenceTransformerEmbedder`'s default is now the model the numbers were measured on, and `CacheRouter` requires an embedder outright so there is no bare default to fall back to). The demo pair was then re-chosen from the verified eval set, picking one that genuinely clears the threshold on the correct model (`0.9404`), so the demo demonstrates the behavior instead of accidentally contradicting the eval.

**The general lesson, and it is the sharpest version of a theme this project keeps hitting:** a passing end-to-end check is only evidence if you know what it actually ran. Sections 6 and 21 were aggregates hiding a bad component and a silently-substituted dependency; this is the same shape a third time, a *demo* silently substituting a dependency, and passing because the substitution happened to be favorable. The failure was not that the number was wrong, it was that a number was being compared against a threshold derived from a different system, and nothing in the output said so.

**Talking point:** "My end-to-end verification was passing on the wrong model for two phases. I found it because fixing an unrelated default made a documented number move, and I chased the moved number instead of assuming I'd broken something. The corrected result actually agrees with my own eval table, which is what convinced me the new number was right and the old one had never been."

---
