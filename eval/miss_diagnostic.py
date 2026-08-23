"""
Phase 4 follow-up: the ef_search sweep showed recall flat at ~65% from
ef=50 all the way to ef=1600, which rules out "beam too narrow" as the
cause. This checks the actual hypothesis: are HNSW's misses landing in a
totally different anchor cluster than the truth (a genuinely wrong
answer), or in a different synthetic sibling of the SAME anchor (harmless
for the real cache, since siblings represent near-identical queries)?

If misses are mostly same-anchor, the strict "exact id" recall@1 metric
used so far is pessimistic relative to what the cache actually needs. If
they're mostly different-anchor, that confirms the upper-layer greedy
routing (ef=1, no backtracking) is committing to the wrong neighborhood
before the wide layer-0 search even starts -- exactly what a flat
ef_search curve would predict.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from embedding import Embedder
from factory import create_index
from scale_dataset import load_seed_anchors, generate_synthetic_vectors

N = 50_000
NUM_QUERIES = 200


def main():
    anchors = load_seed_anchors()
    embedder = Embedder("all-mpnet-base-v2")
    raw = embedder.embed_batch([t for _, t in anchors])
    anchor_vectors = (raw / np.linalg.norm(raw, axis=1, keepdims=True)).astype(np.float32)

    synthetic, insert_parent_idx = generate_synthetic_vectors(anchor_vectors, n=N, seed=1)
    ids = [f"syn_{i:06d}" for i in range(N)]
    id_to_parent = {id_: int(p) for id_, p in zip(ids, insert_parent_idx)}

    print(f"building linear index (n={N})...")
    linear = create_index("linear", dim=embedder.dim)
    for id_, vec in zip(ids, synthetic):
        linear.insert(id_, vec)

    print(f"building hnsw graph (n={N})...")
    hnsw = create_index("hnsw", dim=embedder.dim)
    for id_, vec in zip(ids, synthetic):
        hnsw.insert(id_, vec)

    queries, query_parent_idx = generate_synthetic_vectors(anchor_vectors, n=NUM_QUERIES, seed=999)

    same_anchor_miss = 0
    diff_anchor_miss = 0
    total_miss = 0
    for q, true_parent in zip(queries, query_parent_idx):
        truth_id, _ = linear.search(q, k=1)[0]
        hnsw_id, _ = hnsw.search(q, k=1)[0]
        if hnsw_id == truth_id:
            continue
        total_miss += 1
        hnsw_answer_parent = id_to_parent[hnsw_id]
        if hnsw_answer_parent == true_parent:
            same_anchor_miss += 1
        else:
            diff_anchor_miss += 1

    print(f"\ntotal misses: {total_miss}/{NUM_QUERIES}")
    print(f"  same anchor as truth (harmless -- same underlying topic): {same_anchor_miss} ({same_anchor_miss/total_miss:.1%})")
    print(f"  DIFFERENT anchor than truth (genuinely wrong answer):     {diff_anchor_miss} ({diff_anchor_miss/total_miss:.1%})")


if __name__ == "__main__":
    main()
