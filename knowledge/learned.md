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
