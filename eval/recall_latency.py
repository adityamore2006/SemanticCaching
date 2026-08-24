"""
Phase 4: recall@1 and query-latency comparison, HNSW vs LinearIndex, as
index size grows.

Builds both indexes on the identical synthetic dataset (see
scale_dataset.py) at each scale in SCALES, then for a fixed set of
held-out queries (fresh noise-perturbations of the same 117 real anchors,
never inserted into either index -- simulating a new paraphrase of an
existing topic arriving at an already-populated cache):
  - recall@1: how often HNSW's top-1 match is the exact same id LinearIndex
    (the exact, ground-truth answer) returns.
  - latency: average per-query search time, both indexes, same queries.
  - miss_gap: when HNSW's top-1 differs from linear's, how much lower its
    similarity was than the true best -- distinguishes "picked a slightly
    worse neighbor" from "missed badly", which a bare recall number can't.

Run: python eval/recall_latency.py
Writes eval/results/phase4_scale.json.
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

SCALES = [1_000, 5_000, 10_000]
# n=50,000 was deliberately tested once (see knowledge/learned.md sections
# 13 and 15) as a stress test, not a realistic operating point -- a real
# semantic cache plausibly holds hundreds to low-thousands of distinct
# topics before saturating, and hosting cost scales with index size, so
# 1k-10k is the range actually worth optimizing and reporting going
# forward. The 50k run stays valuable: it's what surfaced both the O(n^2)
# insert bug and the unrecoverable-routing recall bug, documented, not
# discarded.
NUM_QUERIES = 200
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results", "phase4_scale.json")


def time_inserts(index, vectors, ids):
    t0 = time.perf_counter()
    for id_, vec in zip(ids, vectors):
        index.insert(id_, vec)
    return time.perf_counter() - t0


def run_scale(anchor_vectors, embedder_dim, n):
    print(f"\n=== n={n} ===")

    synthetic, _ = generate_synthetic_vectors(anchor_vectors, n=n, seed=1)
    ids = [f"syn_{i:06d}" for i in range(n)]

    linear = create_index("linear", dim=embedder_dim)
    linear_insert_s = time_inserts(linear, synthetic, ids)
    print(f"linear insert: {linear_insert_s:.2f}s")

    hnsw = create_index("hnsw", dim=embedder_dim)
    hnsw_insert_s = time_inserts(hnsw, synthetic, ids)
    print(f"hnsw insert:   {hnsw_insert_s:.2f}s")

    queries, _ = generate_synthetic_vectors(anchor_vectors, n=NUM_QUERIES, seed=999)

    linear_times, hnsw_times = [], []
    agree = 0
    miss_gaps = []
    for q in queries:
        t0 = time.perf_counter()
        lin_id, lin_sim = linear.search(q, k=1)[0]
        linear_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        hnsw_id, hnsw_sim = hnsw.search(q, k=1)[0]
        hnsw_times.append(time.perf_counter() - t0)

        if hnsw_id == lin_id:
            agree += 1
        else:
            miss_gaps.append(lin_sim - hnsw_sim)

    result = {
        "n": n,
        "linear_insert_s": linear_insert_s,
        "hnsw_insert_s": hnsw_insert_s,
        "linear_query_ms": float(np.mean(linear_times)) * 1000,
        "hnsw_query_ms": float(np.mean(hnsw_times)) * 1000,
        "recall_at_1": agree / NUM_QUERIES,
        "num_misses": NUM_QUERIES - agree,
        "avg_miss_gap": float(np.mean(miss_gaps)) if miss_gaps else 0.0,
        "max_miss_gap": float(np.max(miss_gaps)) if miss_gaps else 0.0,
    }
    print(
        f"recall@1={result['recall_at_1']:.3f}  "
        f"linear_query={result['linear_query_ms']:.3f}ms  "
        f"hnsw_query={result['hnsw_query_ms']:.3f}ms  "
        f"misses={result['num_misses']}/{NUM_QUERIES}  "
        f"avg_miss_gap={result['avg_miss_gap']:.4f}"
    )
    return result


def main():
    anchors = load_seed_anchors()
    embedder = SentenceTransformerEmbedder("all-mpnet-base-v2")
    raw = embedder.embed_batch([t for _, t in anchors])
    anchor_vectors = (raw / np.linalg.norm(raw, axis=1, keepdims=True)).astype(np.float32)
    print(f"{len(anchors)} seed anchors, dim={embedder.dim}")

    results = []
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    for n in SCALES:
        result = run_scale(anchor_vectors, embedder.dim, n)
        results.append(result)
        # Written after every scale, not just at the end, so a long run
        # that gets interrupted still leaves the completed tiers on disk.
        with open(RESULTS_PATH, "w") as f:
            json.dump({"noise_sigma": 0.018, "num_queries": NUM_QUERIES, "results": results}, f, indent=2)

    print(f"\nwrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
