"""
Phase 4: synthetic scale-dataset generator.

Recall@k and latency both need far more vectors than the 117 real,
hand-verified anchors (data/eval_pairs.json + data/phase4_new_anchors.json)
provide on their own -- hand-authoring thousands of realistic queries
isn't practical. Instead of pure random vectors (which don't reflect real
embedding geometry -- in 768 dimensions, random vectors are all nearly
equidistant from each other, giving HNSW's graph nothing meaningful to
exploit), this perturbs the real anchors: small calibrated Gaussian noise
added to each real, verified embedding and renormalized, simulating "many
different phrasings of the same underlying topic" -- a fair proxy for what
a real cache accumulates over time.

NOISE_SIGMA was picked empirically (not by formula alone) against a real
768-dim embedding: sigma=0.018 lands parent-to-child similarity at
mean 0.895, range roughly 0.88-0.91, matching where genuine paraphrase
pairs sit in the Phase 2 eval data (see knowledge/learned.md section 11).
"""

import json
import os

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
NOISE_SIGMA = 0.018


def load_seed_anchors():
    """
    Combines the 67 real Phase 2 anchors (loaded straight from
    eval_pairs.json, never copied) with the 50 new phase4_new_anchors.json
    anchors. Returns a list of (id, text), 117 entries.
    """
    with open(os.path.join(DATA_DIR, "eval_pairs.json")) as f:
        phase2 = json.load(f)
    phase2_texts = sorted(set(p["query_a"] for p in phase2["pairs"]))
    phase2_anchors = [(f"p2_{i:03d}", t) for i, t in enumerate(phase2_texts)]

    with open(os.path.join(DATA_DIR, "phase4_new_anchors.json")) as f:
        phase4_new = json.load(f)
    new_anchors = [(a["id"], a["text"]) for a in phase4_new["anchors"]]

    return phase2_anchors + new_anchors


def generate_synthetic_vectors(anchor_vectors, n, seed=0):
    """
    anchor_vectors: (num_anchors, dim) array, already unit-normalized.
    Returns (n, dim) synthetic vectors, distributed round-robin across
    anchors so every anchor gets an equal-sized noise cloud, plus a
    parallel list of which anchor id each synthetic vector was generated
    from (useful for sanity checks, not required for recall/latency
    measurement itself).
    """
    rng = np.random.default_rng(seed)
    num_anchors, dim = anchor_vectors.shape

    parent_idx = np.arange(n) % num_anchors
    noise = rng.normal(0, NOISE_SIGMA, size=(n, dim))
    synthetic = anchor_vectors[parent_idx] + noise
    synthetic = synthetic / np.linalg.norm(synthetic, axis=1, keepdims=True)

    return synthetic.astype(np.float32), parent_idx


if __name__ == "__main__":
    # Quick sanity check: generate a small batch, confirm the actual
    # resulting similarity to each parent lands where NOISE_SIGMA was
    # calibrated for.
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from local_embedder import SentenceTransformerEmbedder

    anchors = load_seed_anchors()
    embedder = SentenceTransformerEmbedder("all-mpnet-base-v2")
    raw = embedder.embed_batch([t for _, t in anchors])
    anchor_vectors = raw / np.linalg.norm(raw, axis=1, keepdims=True)

    synthetic, parent_idx = generate_synthetic_vectors(anchor_vectors, n=500, seed=0)
    sims = np.sum(synthetic * anchor_vectors[parent_idx], axis=1)
    print(f"{len(anchors)} seed anchors, generated {len(synthetic)} synthetic vectors")
    print(f"parent similarity: mean={sims.mean():.4f} min={sims.min():.4f} max={sims.max():.4f}")
