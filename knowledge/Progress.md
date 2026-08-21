# Progress

## Current status

- Phase 1 is active and the project is in the local brute-force vector search stage.
- The baseline implementation in src/linear_search.py is in place and validated with focused tests.
- Scaffolding is now decoupled around a shared index contract so linear search and HNSW are hot-swappable (swap happens at one factory call, nothing downstream changes).
- The immediate next step is to keep the project moving through the remaining Phase 1 verification and then transition into the eval harness and threshold work without expanding scope.

## Completed

- Confirmed the project brief and execution roadmap in PLAN.md.
- Built the brute-force cosine-similarity search index with insert + search behavior.
- Added correctness tests covering self-similarity, ordering, extreme values, empty inputs, and dimension errors.
- Introduced the VectorIndex abstract contract (src/vector_index.py): the seam that lets the cache router and eval harness depend on an interface, never a concrete index.
- LinearIndex now implements VectorIndex; added src/hnsw.py as a contract-conforming Phase 3 stub and src/factory.py (create_index) as the single hot-swap point.
- Added an index-agnostic contract test suite (tests/test_index_contract.py) that HNSW will inherit for free once implemented. Full suite: 18 passing.
- Set up a local venv + .gitignore; deps (numpy, pytest) installed and green.

## Architectural decisions

- Single-responsibility index: the index maps vectors <-> ids only. Response storage (cache), text embedding, and hit/miss decisions are separate components, so each stays swappable on its own.
- Hot-swap via registry + factory: linear vs hnsw is one config value; the eval harness can run the identical code path against both to compute recall.
- Linear search is kept permanently as the exact ground-truth answer key for grading HNSW's approximate results (Phase 3/4).

## Next milestone

- Finalize and verify the exact Phase 1 implementation before moving to the eval/data phase.
- Keep notes brief and evidence-backed, and avoid cloud or architecture detours until the local core is fully validated.

## Guardrail note

- No rabbit-hole exploration beyond the active phase unless it directly advances the approved build plan or is explicitly approved as a scope change.
