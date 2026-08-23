# Progress

## Current status

- Phase 1 through Phase 5 are complete. Phase 6 (AWS wiring) is next.
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
- Added 50 new cross-domain seed anchors (data/phase4_new_anchors.json: recipe app, fitness tracker, budgeting app) alongside the original 67, verified zero cross-anchor collisions across the combined 117 (max cross-domain similarity 0.638). Kept separate from data/eval_pairs.json so Phase 2's locked threshold decision stays untouched.
- Built eval/scale_dataset.py: generates synthetic datasets by perturbing the 117 real anchors with calibrated Gaussian noise (sigma=0.018, empirically tuned to land parent-similarity at 0.85-0.95, matching real paraphrase pairs) instead of using unrepresentative random vectors.
- Built eval/recall_latency.py and ran it at n=1,000/10,000/50,000: recall@1 degrades 98.0% -> 64.5% -> 54.0% as ef_search=50 becomes a shrinking fraction of the graph (expected, honestly reported); query latency crosses over between linear and HNSW between 1k and 10k, reaching HNSW 4.3x faster than linear at 50k (linear query time grew 41x over the 50x data increase, HNSW's only 2x). Full results and reasoning in knowledge/learned.md section 14.
- Found and fixed a real O(n^2) bug in LinearIndex.insert (np.vstack reallocating the whole array every insert) that Phase 4's scale testing exposed but Phase 1/2's small-n tests never could. Fixed via lazy-cached matrix rebuild; insert time at n=50,000 dropped from 443.8s to 0.37s, full test suite unaffected (23/23 still passing).
- Diagnosed and fixed a real HNSW recall problem at n=50,000: an ef_search sweep (50->1600) showed recall flat, ruling out "beam too narrow"; a targeted diagnostic then showed 60.9% of misses landed in a completely different anchor cluster, pointing at unrecoverable greedy routing through the upper layers (ef=1, per the paper's own algorithm) as the real bottleneck. Added ef_upper (default 8, a deliberate deviation from the paper) to both insert's phase A and search()'s upper-layer descent. Recall@1 at n=50,000 improved 54.0% -> 65.5% with no latency regression (still ~3.3x faster than linear); re-verified small-scale exact correctness still holds (194/194). Full investigation in knowledge/learned.md section 15.
- Re-scoped the benchmark's realistic operating range to 1k-10k (down from 50k), matching what a real cache and real hosting costs actually look like; n=50,000 stays documented as the stress test that found the O(n^2) and routing bugs, not the ongoing target scale. Chose a fully serverless AWS architecture for the eventual deployment (API Gateway + Lambda + DynamoDB on-demand + Bedrock + CloudWatch), explicitly avoiding OpenSearch Serverless due to its non-trivial idle billing floor. Full reasoning and the canonical linear-vs-HNSW comparison table in knowledge/learned.md sections 16-17.
- Removed the unused, contract-contradicting metadata dict from both LinearIndex and HNSWIndex, then built Phase 5 for real: src/cache_store.py (CacheStore ABC + InMemoryCacheStore) and src/cache_router.py (CacheRouter: embed -> search -> threshold check -> hit/miss -> stubbed LLM call -> insert). index_kind has no default deliberately; the LLM call is stubbed, real Bedrock wiring is Phase 6. 34/34 tests passing (7 new router tests against a fast FakeEmbedder, 4 new store tests), plus a real end-to-end run with the actual embedding model: paraphrase hit at similarity 0.826, unrelated query missed at 0.099. Full reasoning in knowledge/learned.md section 18. Debugging story in knowledge/learned.md section 13.

## Architectural decisions

- Single-responsibility index: the index maps vectors <-> ids only. Response storage (cache), text embedding, and hit/miss decisions are separate components, so each stays swappable on its own.
- Hot-swap via registry + factory: linear vs hnsw is one config value; the eval harness can run the identical code path against both to compute recall.
- Linear search is kept permanently as the exact ground-truth answer key for grading HNSW's approximate results (Phase 3/4).

## Next milestone

- Begin Phase 6: wrap CacheRouter for Lambda deployment (API Gateway + Lambda + DynamoDB on-demand + Bedrock + CloudWatch, see knowledge/learned.md section 16). Swap the stub call_llm for real Bedrock deliberately, and build DynamoDBCacheStore against the same CacheStore contract InMemoryCacheStore already implements.
- Keep notes brief and evidence-backed.

## Guardrail note

- No rabbit-hole exploration beyond the active phase unless it directly advances the approved build plan or is explicitly approved as a scope change.
