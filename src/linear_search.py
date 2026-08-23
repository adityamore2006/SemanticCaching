"""
Linear (brute-force) vector search.

This is deliberately the simplest possible correct implementation: to find
the nearest neighbor of a query vector, compare it against every stored
vector and rank by similarity. O(n) per query, no approximation, no index
structure to reason about.

It exists for two reasons:
1. It's the working baseline. The cache can hit/miss against this before
   HNSW exists at all.
2. It's the answer key. Once HNSW (Phase 3) is built, we check its
   approximate results against what this brute-force search returns on the
   same data. If HNSW's top-k matches this index's top-k most of the time,
   HNSW is working. This index is the ground truth for that recall check.

Similarity metric: cosine similarity. We normalize vectors on insert so
similarity search reduces to a dot product at query time.
"""

import numpy as np

from vector_index import VectorIndex


class LinearIndex(VectorIndex):
    """Brute-force nearest neighbor index over vectors, ranked by cosine similarity.

    Implements the VectorIndex contract with zero approximation, which is
    what makes it the ground-truth answer key for grading HNSW later.
    """

    def __init__(self, dim):
        super().__init__(dim)
        self.ids = []              # ids[i] corresponds to _vectors[i]
        self._vectors = []         # list of normalized 1-D float32 arrays
        # Lazily rebuilt into one contiguous matrix on first search() after
        # an insert, then reused until the next insert invalidates it.
        # Rebuilding eagerly on every insert (np.vstack-ing the whole array
        # each time) is what used to make bulk-loading n vectors O(n^2)
        # instead of O(n) -- see knowledge/learned.md.
        self._matrix_cache = None
        self.metadata = {}         # id -> arbitrary payload (e.g. cached response)

    def _normalize(self, vector):
        vector = np.asarray(vector, dtype=np.float32)
        if vector.shape != (self.dim,):
            raise ValueError(f"expected vector of shape ({self.dim},), got {vector.shape}")
        norm = np.linalg.norm(vector)
        if norm == 0:
            raise ValueError("cannot normalize a zero vector")
        return vector / norm

    def insert(self, id, vector, metadata=None):
        """
        Add a vector to the index under the given id. If the id already
        exists, this appends a duplicate entry rather than overwriting,
        callers that need upsert semantics should check first.
        """
        normalized = self._normalize(vector)
        self._vectors.append(normalized)
        self.ids.append(id)
        self._matrix_cache = None
        if metadata is not None:
            self.metadata[id] = metadata

    def _matrix(self):
        if self._matrix_cache is None:
            self._matrix_cache = (
                np.stack(self._vectors) if self._vectors else np.empty((0, self.dim), dtype=np.float32)
            )
        return self._matrix_cache

    def search(self, query_vector, k=1):
        """
        Return the top-k nearest neighbors to query_vector as a list of
        (id, similarity) tuples, sorted by similarity descending.

        similarity is cosine similarity in [-1, 1] (in practice usually
        [0, 1] for text embeddings, which tend to point in similar
        directions). 1.0 means identical direction.
        """
        if len(self.ids) == 0:
            return []

        query = self._normalize(query_vector)

        # vectors are already unit-normalized at insert time, so cosine
        # similarity is just the dot product here.
        #
        # errstate suppresses spurious divide/overflow/invalid RuntimeWarnings
        # that Apple's Accelerate BLAS backend can raise on this matmul even
        # when every input is a clean, non-zero, finite unit vector and the
        # result contains no nan/inf, verified directly against this backend.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            similarities = self._matrix() @ query  # shape: (n,)

        k = min(k, len(self.ids))
        # argpartition for O(n) top-k selection instead of a full O(n log n)
        # sort, then sort just the k results for a clean ordered return.
        top_k_idx = np.argpartition(-similarities, k - 1)[:k]
        top_k_idx = top_k_idx[np.argsort(-similarities[top_k_idx])]

        return [(self.ids[i], float(similarities[i])) for i in top_k_idx]

    def __len__(self):
        return len(self.ids)


if __name__ == "__main__":
    # Small runnable demo with hand-checkable vectors, no embeddings needed.
    # Three 2D vectors: "east", "north", "almost-east" (close to "east").
    index = LinearIndex(dim=2)
    index.insert("east", [1.0, 0.0])
    index.insert("north", [0.0, 1.0])
    index.insert("almost_east", [0.98, 0.2])

    query = [1.0, 0.05]  # should be closest to "east", then "almost_east", then "north"
    results = index.search(query, k=3)

    print("query:", query)
    for id, similarity in results:
        print(f"  {id:12s} similarity={similarity:.4f}")
