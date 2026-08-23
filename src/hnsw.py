"""
HNSW (Hierarchical Navigable Small World) index -- Phase 3.

Built from the original Malkov & Yashunin paper, from scratch, no library.
Conforms to the same VectorIndex contract as LinearIndex (see vector_index.py),
which is the whole point of the scaffolding: nothing above this (factory,
eval harness, later the cache router) has to change now that this is real.
Linear search stays around as the exact answer key this index is graded
against.

Design decisions locked in for this build (see knowledge/learned.md for the
reasoning trail):
  - Layer assignment: probabilistic, level = floor(-ln(u) * mL) with
    mL = 1 / ln(M). Most nodes live only at layer 0; each layer up is
    roughly 1/M as populated as the one below it. This is what gives search
    its "sparse highway at the top, dense refinement at the bottom" shape.
  - Neighbor selection during insert: SELECT-NEIGHBORS-SIMPLE (keep the M
    closest candidates by similarity), not the paper's diversity-aware
    heuristic variant. Simpler to trace and defend; the diversity
    heuristic's extra recall mostly shows up at a scale beyond this eval.
  - Layer 0 gets 2*M neighbors (M_max0), every other layer gets M. This is
    the paper's Algorithm 1 default, unconditional -- layer 0 is the only
    layer guaranteed to hold every node, so it carries more of the graph's
    actual connectivity and gets a bigger budget for it.
  - Upper-layer descent uses a small beam (ef_upper), NOT the paper's
    strict ef=1. This is a deliberate deviation, added after Phase 4 scale
    testing (n=50,000) diagnosed a real problem the paper's default
    doesn't warn about: a single-path, no-backtracking greedy walk through
    the upper layers can commit to the wrong neighborhood before the wide
    ef_search beam at layer 0 ever runs, and no amount of widening that
    final beam recovers from a bad entry point -- confirmed empirically,
    an ef_search sweep from 50 to 1600 barely moved recall (65.0% ->
    65.5%), while checking where the misses actually landed showed 61% of
    them in a completely different anchor cluster than the true answer,
    not just a different near-duplicate of the right one. ef_upper gives
    the walk a few alternatives to consider at each upper layer instead of
    committing to one path outright. See knowledge/learned.md section 15.

Similarity metric: cosine similarity, same as LinearIndex. Vectors are
normalized to unit length on insert, so "distance" and "similarity" rank
identically here (see knowledge/learned.md) -- this file works entirely in
similarity terms (higher is closer) to stay consistent with LinearIndex.
"""

import heapq
import math
import random

from typing import Dict, Hashable, List, Sequence

import numpy as np

from vector_index import VectorIndex, SearchResult


class HNSWIndex(VectorIndex):
    """
    Approximate nearest-neighbor index via a layered navigable small-world
    graph. insert() builds the graph one node at a time; search() greedily
    descends it layer by layer. Both share one core routine, _search_layer,
    a greedy beam search within a single layer -- insert uses it (with
    ef_construction) to find candidate neighbors for a new node, search
    uses it (with ef_search) to find the query's actual nearest neighbors.
    """

    def __init__(self, dim: int, M: int = 16, ef_construction: int = 200, ef_search: int = 50, ef_upper: int = 8):
        super().__init__(dim)
        self.M = M
        self.M_max0 = 2 * M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        # Not from the paper -- see the module docstring's "Upper-layer
        # descent" note. Small on purpose: upper layers are sparse (see
        # the layer-assignment note above), so widening the walk there is
        # cheap even though ef_construction/ef_search are much larger.
        self.ef_upper = ef_upper
        self.mL = 1.0 / math.log(M)

        self.vectors: Dict[Hashable, np.ndarray] = {}
        # id -> {layer: [neighbor_id, ...]}. Only layers 0..levels[id] exist
        # for a given id -- looking up a layer a node doesn't live at is a
        # bug, not a valid empty-result case, so no .get() defaulting here.
        self.neighbors: Dict[Hashable, Dict[int, List[Hashable]]] = {}
        self.levels: Dict[Hashable, int] = {}

        self.entry_point: Hashable = None
        self.max_level: int = -1

    def _normalize(self, vector):
        vector = np.asarray(vector, dtype=np.float32)
        if vector.shape != (self.dim,):
            raise ValueError(f"expected vector of shape ({self.dim},), got {vector.shape}")
        norm = np.linalg.norm(vector)
        if norm == 0:
            raise ValueError("cannot normalize a zero vector")
        return vector / norm

    def _similarity(self, vec_a, vec_b) -> float:
        # Both sides are already unit-normalized at insert/search time, so
        # cosine similarity is just the dot product, same as LinearIndex.
        return float(np.dot(vec_a, vec_b))

    def _random_level(self) -> int:
        # 1 - random.random() maps to (0, 1], excluding 0 exactly, so log()
        # never sees a zero input.
        return int(-math.log(1.0 - random.random()) * self.mL)

    def _search_layer(self, query, entry_points, ef, layer):
        """
        Greedy beam search within one layer (the paper's SEARCH-LAYER).

        Explores outward from entry_points, following graph edges at this
        layer, keeping the `ef` best (highest-similarity) nodes seen.
        Returns those `ef` nodes as (id, similarity), sorted descending.

        Two small heaps do the bookkeeping:
          - candidates: what to explore next, best-first. Python's heapq is
            a min-heap, so similarities are pushed negated to pop the
            highest first.
          - found: the best `ef` results seen so far. Kept as a min-heap on
            plain similarity, so its root (heapq's smallest) is always the
            *worst* of the kept set -- exactly the one to evict when a
            better candidate shows up.
        """
        visited = set(entry_points)
        candidates = []
        found = []

        for ep in entry_points:
            sim = self._similarity(self.vectors[ep], query)
            heapq.heappush(candidates, (-sim, ep))
            heapq.heappush(found, (sim, ep))

        while candidates:
            neg_sim, current = heapq.heappop(candidates)
            current_sim = -neg_sim

            # Nothing left in `candidates` can beat the worst of `found`
            # (candidates only get explored in descending similarity
            # order), so once that's true and `found` is full, stop.
            if current_sim < found[0][0] and len(found) >= ef:
                break

            for neighbor_id in self.neighbors[current][layer]:
                if neighbor_id in visited:
                    continue
                visited.add(neighbor_id)
                sim = self._similarity(self.vectors[neighbor_id], query)

                if len(found) < ef or sim > found[0][0]:
                    heapq.heappush(candidates, (-sim, neighbor_id))
                    heapq.heappush(found, (sim, neighbor_id))
                    if len(found) > ef:
                        heapq.heappop(found)

        return sorted(((id_, sim) for sim, id_ in found), key=lambda pair: -pair[1])

    def _prune(self, node_id, layer, cap):
        """Keep only the `cap` closest neighbors of node_id at this layer."""
        node_vec = self.vectors[node_id]
        scored = [
            (self._similarity(node_vec, self.vectors[nid]), nid)
            for nid in self.neighbors[node_id][layer]
        ]
        scored.sort(key=lambda pair: -pair[0])
        self.neighbors[node_id][layer] = [nid for _, nid in scored[:cap]]

    def insert(self, id: Hashable, vector: Sequence[float]) -> None:
        normalized = self._normalize(vector)
        level = self._random_level()

        self.vectors[id] = normalized
        self.levels[id] = level
        self.neighbors[id] = {l: [] for l in range(level + 1)}

        if self.entry_point is None:
            self.entry_point = id
            self.max_level = level
            return

        entry = self.entry_point

        # Phase A: descend from the current top layer down to one above
        # where the new node lives, keeping a small beam (ef_upper) of
        # candidates at each layer instead of committing to a single path.
        # These upper layers are sparse highways -- all we need from them
        # is a good jumping-off point for phase B, not a thorough search,
        # but a pure ef=1 walk has zero backtracking and can commit to the
        # wrong neighborhood on one bad hop (see module docstring).
        for layer in range(self.max_level, level, -1):
            entry = self._search_layer(normalized, [entry], ef=self.ef_upper, layer=layer)[0][0]

        # Phase B: from min(max_level, level) down to layer 0, actually
        # find candidate neighbors and wire the new node into the graph.
        entry_points = [entry]
        for layer in range(min(self.max_level, level), -1, -1):
            found = self._search_layer(normalized, entry_points, ef=self.ef_construction, layer=layer)

            cap = self.M_max0 if layer == 0 else self.M
            selected = found[:cap]  # SELECT-NEIGHBORS-SIMPLE: closest `cap`
            self.neighbors[id][layer] = [nid for nid, _ in selected]

            # Edges are bidirectional: each selected neighbor also needs id
            # added to its own list, then re-pruned if that pushes it over
            # its own cap.
            for nid, _ in selected:
                self.neighbors[nid][layer].append(id)
                if len(self.neighbors[nid][layer]) > cap:
                    self._prune(nid, layer, cap)

            entry_points = [nid for nid, _ in found]

        if level > self.max_level:
            self.max_level = level
            self.entry_point = id

    def search(self, query_vector: Sequence[float], k: int = 1) -> List[SearchResult]:
        if self.entry_point is None:
            return []

        query = self._normalize(query_vector)
        entry = self.entry_point

        # Small-beam descent through the upper layers, same as insert's
        # phase A: find a good entry point for the real search without
        # paying for a wide beam until we're at the bottom, but with
        # enough width (ef_upper) to backtrack away from one bad hop
        # instead of committing to a single path (see module docstring).
        for layer in range(self.max_level, 0, -1):
            entry = self._search_layer(query, [entry], ef=self.ef_upper, layer=layer)[0][0]

        ef = max(self.ef_search, k)
        found = self._search_layer(query, [entry], ef=ef, layer=0)
        return found[:k]

    def __len__(self) -> int:
        return len(self.vectors)


if __name__ == "__main__":
    # Same hand-checkable demo as linear_search.py's __main__, so the two
    # indexes can be compared side by side on identical data.
    index = HNSWIndex(dim=2)
    index.insert("east", [1.0, 0.0])
    index.insert("north", [0.0, 1.0])
    index.insert("almost_east", [0.98, 0.2])

    query = [1.0, 0.05]  # should be closest to "east", then "almost_east", then "north"
    results = index.search(query, k=3)

    print("query:", query)
    for id, similarity in results:
        print(f"  {id:12s} similarity={similarity:.4f}")
