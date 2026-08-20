"""
HNSW (Hierarchical Navigable Small World) index -- Phase 3.

Placeholder that pins down the shape of the real thing. It conforms to the
same VectorIndex contract as LinearIndex, which is the whole point of the
scaffolding: when this is filled in, nothing above it (factory, cache
router, eval harness) has to change. Linear search stays around as the
answer key we grade this index's approximate results against.

Nothing here is implemented yet on purpose, building the graph is Phase 3
and is the core engineering deliverable of the project. Keeping the stub
in the tree means the factory can already list "hnsw" as a target and the
contract test can be pointed at it the moment insert/search exist.
"""

from typing import List, Sequence, Hashable

from vector_index import VectorIndex, SearchResult


class HNSWIndex(VectorIndex):
    """
    Approximate nearest-neighbor index via a layered navigable small-world
    graph. Construction params (M, ef_construction, ef_search) will land
    here in Phase 3; they are the knobs that trade recall against speed.
    """

    def __init__(self, dim: int, M: int = 16, ef_construction: int = 200, ef_search: int = 50):
        super().__init__(dim)
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        raise NotImplementedError(
            "HNSWIndex is a Phase 3 deliverable and is not implemented yet. "
            "Use create_index('linear', dim=...) for now."
        )

    def insert(self, id: Hashable, vector: Sequence[float]) -> None:
        raise NotImplementedError

    def search(self, query_vector: Sequence[float], k: int = 1) -> List[SearchResult]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError
