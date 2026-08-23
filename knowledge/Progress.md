# Progress

## Current status

- Phase 1, Phase 2, and Phase 3 are complete. Phase 4 (formal recall@k + speed comparison at scale) is next.
- The baseline implementation in src/linear_search.py is in place and validated with focused tests.
- Scaffolding is decoupled around a shared index contract so linear search and HNSW are hot-swappable (swap happens at one factory call, nothing downstream changes).
- The eval harness (eval/threshold_sweep.py) is built, verified against a 194-pair hand-authored test set, and has produced a documented operating threshold. Full reasoning lives in knowledge/learned.md.

## Completed

- Confirmed the project brief and execution roadmap in PLAN.md.
- Built the brute-force cosine-similarity search index with insert + search behavior.
- Added correctness tests covering self-similarity, ordering, extreme values, empty inputs, and dimension errors.
- Introduced the VectorIndex abstract contract (src/vector_index.py): the seam that lets the cache router and eval harness depend on an interface, never a concrete index.
- LinearIndex now implements VectorIndex; added src/hnsw.py as a contract-conforming Phase 3 stub and src/factory.py (create_index) as the single hot-swap point.
- Added an index-agnostic contract test suite (tests/test_index_contract.py) that HNSW will inherit for free once implemented. Full suite: 18 passing.
- Set up a local venv + .gitignore; deps (numpy, pytest, sentence-transformers) installed and green.
- Built src/embedding.py (Embedder) and eval/threshold_sweep.py, the Phase 2 harness: embeds a hand-authored test set, builds a shared index of previously-seen queries, sweeps similarity thresholds, reports hit rate and wrong-match rate.
- Compared four embedding configurations (MiniLM, mpnet, bge-large unprefixed, bge-large prefixed) on the same eval; mpnet won on the actual numbers, including a case where a theoretically-better model (BGE, hard-negative trained) lost in practice. See knowledge/learned.md section 8.
- Scaled the eval set from 45 to 194 pairs (67 topic anchors) and built eval/verify_pairs.py, an automated collision checker that embeds the whole set and checks every pair against every anchor, not just its own. Caught and fixed three real authoring mistakes before trusting the numbers. See knowledge/learned.md section 10.
- Chose the final operating threshold (0.80 on all-mpnet-base-v2) using an explicit asymmetric cost model, safety weighted over raw hit rate, with the reasoning for why 0.80 and not stricter documented in knowledge/learned.md section 11.
- Built HNSWIndex from scratch (src/hnsw.py): probabilistic layer assignment, greedy multi-layer descent for search, SELECT-NEIGHBORS-SIMPLE for insert-time neighbor selection. Conforms to VectorIndex, so it dropped straight into the existing contract test suite (tests/test_index_contract.py now runs all 6 invariant checks against both "linear" and "hnsw", 23/23 passing, stable across 10 repeated runs despite randomized layer assignment). Design reasoning in knowledge/learned.md section 12.
- Validated HNSW against LinearIndex on the real 194-pair eval set (not synthetic data): 194/194 top-1 matches identical, similarity scores agreeing to 6 decimal places, and the threshold_sweep table matching the locked linear baseline exactly at every reference threshold (0.70/0.75/0.80/0.85). Caveat documented, not overclaimed: at 67 anchors this is expected since ef_search=50 sees nearly the whole graph; the real approximation/speed tradeoff is Phase 4's job once the dataset is large enough to matter.

## Architectural decisions

- Single-responsibility index: the index maps vectors <-> ids only. Response storage (cache), text embedding, and hit/miss decisions are separate components, so each stays swappable on its own.
- Hot-swap via registry + factory: linear vs hnsw is one config value; the eval harness can run the identical code path against both to compute recall.
- Linear search is kept permanently as the exact ground-truth answer key for grading HNSW's approximate results (Phase 3/4).

## Next milestone

- Begin Phase 4: formal recall@k against LinearIndex as dataset size grows, query latency comparison (linear should degrade linearly, HNSW should not), and rerunning the Phase 2 threshold sweep specifically on the approximate index to confirm the operating threshold still holds once recall isn't 100%.
- Keep notes brief and evidence-backed, and avoid cloud or architecture detours until the local core is fully validated.

## Guardrail note

- No rabbit-hole exploration beyond the active phase unless it directly advances the approved build plan or is explicitly approved as a scope change.
