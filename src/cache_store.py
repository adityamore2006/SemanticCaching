"""
Cache response storage -- Phase 5.

Deliberately separate from the vector index (see vector_index.py's
docstring and knowledge/learned.md section 1): the index answers "which
stored vectors are nearest", CacheStore answers "what response did we
actually give for this id". Keeping them separate means swapping index
kind (linear <-> hnsw) and swapping storage backend (dict -> DynamoDB,
Phase 6) are each independent, one-line changes -- neither ever forces
touching the other.

Contract: an id used here should always correspond 1:1 with an id
inserted into whatever VectorIndex is paired with this store. CacheStore
itself knows nothing about vectors or similarity; CacheRouter (the piece
that actually pairs an index with a store) is what keeps them in sync.
"""

from abc import ABC, abstractmethod
from typing import Hashable, Optional


class CacheStore(ABC):
    """Maps an id to the response stored under it. Nothing more."""

    @abstractmethod
    def put(self, id: Hashable, response: str) -> None:
        """Store response under id, overwriting any existing entry."""

    @abstractmethod
    def get(self, id: Hashable) -> Optional[str]:
        """Return the stored response for id, or None if not present."""

    @abstractmethod
    def __len__(self) -> int:
        """Number of stored responses."""


class InMemoryCacheStore(CacheStore):
    """Plain dict-backed store. Phase 5's default; DynamoDBCacheStore
    (Phase 6) implements the same contract against real persistence."""

    def __init__(self):
        self._data = {}

    def put(self, id, response):
        self._data[id] = response

    def get(self, id):
        return self._data.get(id)

    def __len__(self):
        return len(self._data)
