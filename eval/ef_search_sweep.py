"""
Phase 4 follow-up: is 54% recall@1 at n=50,000 HNSW's ceiling, or just what
you get from never turning the ef_search dial?

ef_search is a pure query-time parameter -- it doesn't touch graph
structure at all, only how wide a beam search() uses at the bottom layer.
So this builds the n=50,000 HNSW graph ONCE (the expensive part, ~114s),
then reuses it across several ef_search values, just mutating the
attribute between rounds, to find the actual recall/latency curve instead
of reading off a single fixed-ef data point.

Uses the same 117 real seed anchors, same noise calibration, same 200
held-out queries, and the same LinearIndex ground truth as
recall_latency.py's n=50,000 tier, so results are directly comparable.
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from local_embedder import SentenceTransformerEmbedder
from factory import create_index
from scale_dataset import load_seed_anchors, generate_synthetic_vectors

N = 50_000
NUM_QUERIES = 200
EF_VALUES = [50, 100, 200, 400, 800, 1600]
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results", "phase4_ef_sweep.json")


def main():
    anchors = load_seed_anchors()
    embedder = SentenceTransformerEmbedder("all-mpnet-base-v2")
    raw = embedder.embed_batch([t for _, t in anchors])
    anchor_vectors = (raw / np.linalg.norm(raw, axis=1, keepdims=True)).astype(np.float32)

    synthetic, _ = generate_synthetic_vectors(anchor_vectors, n=N, seed=1)
    ids = [f"syn_{i:06d}" for i in range(N)]

    print(f"building linear index (n={N})...")
    linear = create_index("linear", dim=embedder.dim)
    for id_, vec in zip(ids, synthetic):
        linear.insert(id_, vec)

    print(f"building hnsw graph (n={N}), ef_search doesn't matter yet, built once...")
    t0 = time.perf_counter()
    hnsw = create_index("hnsw", dim=embedder.dim)
    for id_, vec in zip(ids, synthetic):
        hnsw.insert(id_, vec)
    print(f"hnsw insert: {time.perf_counter() - t0:.1f}s (paid once, reused for every ef below)")

    queries, _ = generate_synthetic_vectors(anchor_vectors, n=NUM_QUERIES, seed=999)
    ground_truth = [linear.search(q, k=1)[0][0] for q in queries]

    linear_query_ms = float(np.mean([
        _time_one(lambda q=q: linear.search(q, k=1)) for q in queries
    ])) * 1000

    results = []
    for ef in EF_VALUES:
        hnsw.ef_search = ef
        agree = 0
        times = []
        for q, truth_id in zip(queries, ground_truth):
            t0 = time.perf_counter()
            top_id, _ = hnsw.search(q, k=1)[0]
            times.append(time.perf_counter() - t0)
            if top_id == truth_id:
                agree += 1
        row = {
            "ef_search": ef,
            "recall_at_1": agree / NUM_QUERIES,
            "hnsw_query_ms": float(np.mean(times)) * 1000,
        }
        results.append(row)
        print(f"ef_search={ef:5d}  recall@1={row['recall_at_1']:.3f}  hnsw_query={row['hnsw_query_ms']:.3f}ms  (linear={linear_query_ms:.3f}ms)")

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump({"n": N, "num_queries": NUM_QUERIES, "linear_query_ms": linear_query_ms, "results": results}, f, indent=2)
    print(f"\nwrote {RESULTS_PATH}")


def _time_one(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


if __name__ == "__main__":
    main()
