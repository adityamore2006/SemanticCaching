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
        self.ids = []              # ids[i] corresponds to vectors[i]
        self.vectors = np.empty((0, dim), dtype=np.float32)
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
        self.vectors = np.vstack([self.vectors, normalized])
        self.ids.append(id)
        if metadata is not None:
            self.metadata[id] = metadata

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
        similarities = self.vectors @ query  # shape: (n,)

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
