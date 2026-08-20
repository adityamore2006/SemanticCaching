"""
The vector index contract.

This is the seam that lets the rest of the system not care whether it's
talking to brute-force linear search or a from-scratch HNSW graph. Both
implement this interface, so the cache router and the eval harness depend
on `VectorIndex`, never on a concrete class. Swapping the index out is a
one-line change at the factory (see factory.py), not a rewrite.

Deliberately small. An index does exactly one thing: map vectors to ids
and answer "which stored vectors are nearest to this query". It does NOT
store cached responses, embed text, or make hit/miss decisions. Those are
separate components (Phase 2 embedding, Phase 5 cache store + router) so
that each piece stays swappable on its own.

Contract every implementation must honor, so linear search can serve as
the ground-truth answer key for HNSW's approximate results (Phase 3/4):
  - similarity is cosine similarity in [-1, 1], higher is nearer.
  - search returns at most k results, sorted by similarity descending.
  - search on an empty index returns [].
  - a vector whose dimensionality != self.dim is a ValueError, not a
    silent wrong answer.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Hashable, Sequence

# (id, cosine_similarity), similarity in [-1, 1].
SearchResult = Tuple[Hashable, float]


class VectorIndex(ABC):
    """Abstract nearest-neighbor index over fixed-dimensionality vectors."""

    def __init__(self, dim: int):
        # Fixed at construction so shape mistakes fail loudly and early
        # instead of surfacing as a subtly wrong similarity later.
        self.dim = dim

    @abstractmethod
    def insert(self, id: Hashable, vector: Sequence[float]) -> None:
        """Add one vector to the index under the given id."""

    @abstractmethod
    def search(self, query_vector: Sequence[float], k: int = 1) -> List[SearchResult]:
        """
        Return up to k nearest neighbors as (id, similarity) tuples, sorted
        by cosine similarity descending. Empty index -> []. k larger than
        the index size returns everything.
        """

    @abstractmethod
    def __len__(self) -> int:
        """Number of vectors currently stored."""
