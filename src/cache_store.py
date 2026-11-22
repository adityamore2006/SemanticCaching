"""
Response storage: what answer did we give for this id?

Separate from the vector index on purpose. The index answers "which stored
vectors are nearest"; the store answers "what response belongs to this id".
CacheRouter is the piece that keeps the two in sync: every id in the index
should have an entry here.

Implementations: InMemoryCacheStore (below, for tests) and
SqliteCacheStore (sqlite_cache_store.py, durable).
"""

from abc import ABC, abstractmethod
from typing import Hashable, Iterator, Optional, Sequence, Tuple

import numpy as np


class CacheStore(ABC):
    @abstractmethod
    def put(
        self,
        id: Hashable,
        response: str,
        vector: Optional[Sequence[float]] = None,
        source: Optional[str] = None,
    ) -> None:
        """Store response under id, overwriting any existing entry.

        `vector` is the query embedding. It is stored so the index can be
        rebuilt from the store after a restart. `source` records where the
        answer came from ("seed" for the curated corpus, "llm" for a model).
        """

    @abstractmethod
    def get(self, id: Hashable) -> Optional[str]:
        """Return the stored response for id, or None. Never touches disk."""

    @abstractmethod
    def all_items(self) -> Iterator[Tuple[Hashable, np.ndarray, str]]:
        """Yield (id, vector, response) for every entry that has a vector,
        oldest first. Used to rebuild the index at startup."""

    @abstractmethod
    def __len__(self) -> int:
        """Number of stored responses."""

    @abstractmethod
    def clear(self) -> int:
        """Remove every entry, returning how many were removed."""


class InMemoryCacheStore(CacheStore):
    """Dict-backed store. Nothing survives the process."""

    def __init__(self):
        self._data = {}  # id -> (response, vector)

    def put(self, id, response, vector=None, source=None):
        vec = None if vector is None else np.asarray(vector, dtype=np.float32)
        self._data[id] = (response, vec)

    def get(self, id):
        entry = self._data.get(id)
        return None if entry is None else entry[0]

    def all_items(self):
        for id, (response, vec) in list(self._data.items()):
            if vec is not None:
                yield id, vec, response

    def __len__(self):
        return len(self._data)

    def clear(self):
        removed = len(self._data)
        self._data.clear()
        return removed
