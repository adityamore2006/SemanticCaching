"""
Phase 2: threshold sweep eval harness.

Answers the project's actual question: where should the cache-hit
similarity threshold sit? Loads data/eval_pairs.json, embeds every query
with the same Embedder Phase 5's cache router will use, builds a single
shared index of every "anchor" query (query_a) the way a real cache would
accumulate previously-seen queries, then checks every query_b against that
shared index rather than against its own anchor in isolation.

That "shared index" choice is deliberate: a real cache doesn't ask "does
this match its intended partner", it asks "does this match anything
currently cached". A near-miss or unrelated query_b could in principle
match a completely different stored anchor, that is exactly the failure
mode the project cares about (a wrong answer served with high confidence).

Swapping this harness to grade HNSW instead of linear search (Phase 4) is
a one-line change: create_index("linear", ...) -> create_index("hnsw", ...).
Nothing else here should need to change.

Run against a specific embedding model to compare it against prior runs:
    python eval/threshold_sweep.py [model_name]
Defaults to Embedder's default (all-MiniLM-L6-v2) if no model is given.
Each run writes its own file under eval/results/<model_name>.json (so
multiple models' results sit on disk side by side, nothing gets
overwritten) and appends one summary row to eval/results/log.md.
"""

import datetime
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from embedding import DEFAULT_MODEL, Embedder
from factory import create_index

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "eval_pairs.json")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
LOG_PATH = os.path.join(RESULTS_DIR, "log.md")

# Swept independently of per-pair similarity computation, since similarity
# only needs to be computed once per pair, classification against a
# threshold is cheap to redo for every value in this range.
THRESHOLDS = [round(0.50 + 0.05 * i, 2) for i in range(10)]  # 0.50 .. 0.95

# Reference points logged in the summary table, so every model's run is
# comparable at the same fixed thresholds rather than whatever looked
# interesting for one particular model.
LOG_REFERENCE_THRESHOLDS = [0.60, 0.70, 0.80]


def load_pairs(path=DATA_PATH):
    with open(path) as f:
        data = json.load(f)
    return data["pairs"]


def build_anchor_index(pairs, embedder):
    """
    Insert one entry per unique query_a text (many pairs share the same
    anchor, e.g. the paraphrase/near-miss/unrelated trio built off "How do
    I reset my password?"). Returns the populated index plus a mapping
    from anchor text to the id it was stored under, so callers can label
    which anchor id "owns" a given pair.
    """
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
    """
    One search per pair against the shared index, independent of any
    threshold. Returns a list of dicts carrying everything the threshold
    sweep needs: category, expected_match, best similarity found, and
    which anchor id actually won (for spotting cross-anchor false matches).
    """
    query_b_texts = [pair["query_b"] for pair in pairs]
    query_b_vectors = embedder.embed_batch(query_b_texts)

    results = []
    for pair, vector in zip(pairs, query_b_vectors):
        matches = index.search(vector, k=1)
        matched_id, similarity = matches[0]
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


def sweep_thresholds(pair_results, thresholds=THRESHOLDS):
    """
    For each threshold, classify every pair as predicted-hit or
    predicted-miss and compare against its label. Reports:
      hit_rate: fraction of paraphrase pairs correctly hit (recall)
      near_miss_wrong_rate: fraction of near-miss pairs incorrectly hit
      unrelated_wrong_rate: fraction of unrelated pairs incorrectly hit
      wrong_match_rate: incorrect hits across all non-paraphrase pairs combined
    """
    paraphrase = [r for r in pair_results if r["category"] == "paraphrase"]
    near_miss = [r for r in pair_results if r["category"] == "near_miss"]
    unrelated = [r for r in pair_results if r["category"] == "unrelated"]
    negatives = near_miss + unrelated

    rows = []
    for t in thresholds:
        hits = sum(1 for r in paraphrase if r["similarity"] >= t)
        near_miss_wrong = sum(1 for r in near_miss if r["similarity"] >= t)
        unrelated_wrong = sum(1 for r in unrelated if r["similarity"] >= t)
        negatives_wrong = near_miss_wrong + unrelated_wrong

        rows.append({
            "threshold": t,
            "hit_rate": hits / len(paraphrase),
            "near_miss_wrong_rate": near_miss_wrong / len(near_miss),
            "unrelated_wrong_rate": unrelated_wrong / len(unrelated),
            "wrong_match_rate": negatives_wrong / len(negatives),
        })
    return rows


def print_table(rows):
    header = f"{'threshold':>9} {'hit_rate':>9} {'near_miss_wrong':>16} {'unrelated_wrong':>16} {'wrong_match_rate':>17}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['threshold']:>9.2f} "
            f"{row['hit_rate']:>9.2f} "
            f"{row['near_miss_wrong_rate']:>16.2f} "
            f"{row['unrelated_wrong_rate']:>16.2f} "
            f"{row['wrong_match_rate']:>17.2f}"
        )


def results_path_for(model_name):
    slug = model_name.replace("/", "__")
    return os.path.join(RESULTS_DIR, f"{slug}.json")


def append_log_entry(embedder, threshold_rows):
    """
    One compact row per run in eval/results/log.md, so every model tried
    is comparable at a glance without opening each full results file.
    Reference thresholds are fixed (see LOG_REFERENCE_THRESHOLDS) so rows
    line up across different models' runs.
    """
    by_threshold = {row["threshold"]: row for row in threshold_rows}
    is_new_file = not os.path.exists(LOG_PATH)

    with open(LOG_PATH, "a") as f:
        if is_new_file:
            cols = ["run_at (UTC)", "model", "dim"]
            for t in LOG_REFERENCE_THRESHOLDS:
                cols += [f"hit@{t}", f"near_miss_wrong@{t}"]
            cols.append("results_file")
            f.write("| " + " | ".join(cols) + " |\n")
            f.write("|" + "---|" * len(cols) + "\n")

        run_at = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M")
        cells = [run_at, embedder.model_name, str(embedder.dim)]
        for t in LOG_REFERENCE_THRESHOLDS:
            row = by_threshold[t]
            cells += [f"{row['hit_rate']:.2f}", f"{row['near_miss_wrong_rate']:.2f}"]
        cells.append(f"`{os.path.relpath(results_path_for(embedder.model_name), os.path.dirname(__file__))}`")
        f.write("| " + " | ".join(cells) + " |\n")


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL

    pairs = load_pairs()
    embedder = Embedder(model_name)
    print(f"embedding model: {embedder.model_name} (dim={embedder.dim})")
    print(f"loaded {len(pairs)} pairs from {DATA_PATH}")

    index, text_to_id = build_anchor_index(pairs, embedder)
    print(f"indexed {len(index)} unique anchor queries")

    pair_results = compute_pair_results(pairs, index, text_to_id, embedder)
    threshold_rows = sweep_thresholds(pair_results)

    print()
    print_table(threshold_rows)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_path = results_path_for(embedder.model_name)
    with open(results_path, "w") as f:
        json.dump({
            "model_name": embedder.model_name,
            "model_dim": embedder.dim,
            "run_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "pairs": pair_results,
            "thresholds": threshold_rows,
        }, f, indent=2)
    append_log_entry(embedder, threshold_rows)

    print(f"\nwrote per-pair results and threshold sweep to {results_path}")
    print(f"appended summary row to {LOG_PATH}")


if __name__ == "__main__":
    main()
