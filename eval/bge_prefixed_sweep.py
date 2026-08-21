"""
One-off experiment: does BGE's recommended instruction prefix fix the poor
separation seen when bge-large-en-v1.5 was run unprefixed (that run scored
everything, including unrelated pairs, suspiciously high)?

BGE's model card recommends prefixing only the QUERY side for retrieval:
"Represent this sentence for searching relevant passages: ". Our setup
already maps onto that asymmetric framing without forcing it: query_a is
the anchor already sitting in the cache (the "passage"), query_b is the
incoming query searching against it (the "query"). So only query_b gets
prefixed here, query_a does not.

Kept as a separate script on purpose, not folded into threshold_sweep.py:
the main harness has to stay a fair, prefix-agnostic comparison across
every model tried. This script tests one model-specific claim (does BGE
need its prefix to behave) without touching that main harness or risking
regressing the mpnet result we might still want to keep as the answer.
"""

import datetime
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from embedding import Embedder
from factory import create_index

from threshold_sweep import (
    RESULTS_DIR,
    append_log_entry,
    load_pairs,
    print_table,
    results_path_for,
    sweep_thresholds,
)

BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
MODEL_NAME = "BAAI/bge-large-en-v1.5"
LABEL = "BAAI/bge-large-en-v1.5 (query-prefixed)"


def build_anchor_index(pairs, embedder):
    """Anchors (query_a) are the "passage" side, no instruction prefix per BGE's docs."""
    index = create_index("linear", dim=embedder.dim)

    text_to_id = {}
    unique_texts = []
    for pair in pairs:
        text = pair["query_a"]
        if text not in text_to_id:
            anchor_id = f"anchor_{len(text_to_id) + 1:02d}"
            text_to_id[text] = anchor_id
            unique_texts.append(text)

    vectors = embedder.embed_batch(unique_texts)
    for text, vector in zip(unique_texts, vectors):
        index.insert(text_to_id[text], vector)

    return index, text_to_id


def compute_pair_results(pairs, index, text_to_id, embedder):
    """query_b is the "query" side searching the cache, gets the BGE instruction prefix."""
    prefixed_query_b = [BGE_QUERY_INSTRUCTION + pair["query_b"] for pair in pairs]
    vectors = embedder.embed_batch(prefixed_query_b)

    results = []
    for pair, vector in zip(pairs, vectors):
        matched_id, similarity = index.search(vector, k=1)[0]
        results.append({
            "id": pair["id"],
            "category": pair["category"],
            "expected_match": pair["expected_match"],
            "own_anchor_id": text_to_id[pair["query_a"]],
            "matched_id": matched_id,
            "matched_own_anchor": matched_id == text_to_id[pair["query_a"]],
            "similarity": similarity,
        })
    return results


def main():
    pairs = load_pairs()
    embedder = Embedder(MODEL_NAME)
    print(f"embedding model: {LABEL} (dim={embedder.dim})")
    print(f"loaded {len(pairs)} pairs")

    index, text_to_id = build_anchor_index(pairs, embedder)
    print(f"indexed {len(index)} unique anchor queries (unprefixed)")

    pair_results = compute_pair_results(pairs, index, text_to_id, embedder)
    threshold_rows = sweep_thresholds(pair_results)

    print()
    print_table(threshold_rows)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_path = results_path_for(LABEL)
    with open(results_path, "w") as f:
        json.dump({
            "model_name": LABEL,
            "model_dim": embedder.dim,
            "run_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "pairs": pair_results,
            "thresholds": threshold_rows,
        }, f, indent=2)

    embedder.model_name = LABEL  # only for the log row, matches the results file label
    append_log_entry(embedder, threshold_rows)

    print(f"\nwrote per-pair results and threshold sweep to {results_path}")
    print("appended summary row to eval/results/log.md")


if __name__ == "__main__":
    main()
