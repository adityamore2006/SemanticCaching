"""
Automated collision check for data/eval_pairs.json.

Manually eyeballing a growing eval set for accidental label contamination
does not scale, and it already failed us twice at 45 pairs (see
knowledge/learned.md section 7). This does the same check the debugging
session did by hand, exhaustively: embed every anchor and every query_b
with the real model, then check each pair against the ENTIRE anchor set,
not just its own pair, flagging anything that looks mislabeled.

Checks:
  - exact duplicate text anywhere in the set (verbatim collision, bug #1's class)
  - anchor-vs-anchor: are any two supposedly distinct anchors actually the
    same real-world question (would make every downstream pair suspect)?
  - unrelated pairs: does query_b score high against ANY anchor, not just
    its own (bug #2's class, the accidental-paraphrase-of-a-different-anchor case)?
  - paraphrase pairs: does query_b's best match across the WHOLE anchor set
    equal its own anchor? A paraphrase that top-matches a different anchor
    is a red flag worth a human look.

Run: python eval/verify_pairs.py [model_name]
Exits non-zero if anything is flagged, so it can gate before a real eval run.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from local_embedder import DEFAULT_MODEL, SentenceTransformerEmbedder
from factory import create_index
from threshold_sweep import DATA_PATH, load_pairs

# Above this, an "unrelated" pair scoring against some OTHER anchor is
# treated as a likely accidental collision, not a hard-but-fair negative.
# Chosen below where real near-misses tend to sit (see the eval's own
# near_miss score distribution), so it flags "this looks like a mistake",
# not "this is a hard case".
UNRELATED_COLLISION_THRESHOLD = 0.55

# Anchors this close to each other are probably the same real-world
# question asked two different ways, which would make every pair built
# off either of them suspect.
ANCHOR_DUPLICATE_THRESHOLD = 0.85


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    pairs = load_pairs()
    embedder = SentenceTransformerEmbedder(model_name)
    print(f"model: {embedder.model_name} (dim={embedder.dim})")
    print(f"checking {len(pairs)} pairs from {DATA_PATH}\n")

    issues = []

    # build the anchor set (unique query_a) exactly like threshold_sweep.py does
    text_to_id = {}
    anchor_texts = []
    for p in pairs:
        if p["query_a"] not in text_to_id:
            anchor_id = f"anchor_{len(text_to_id) + 1:03d}"
            text_to_id[p["query_a"]] = anchor_id
            anchor_texts.append(p["query_a"])
    id_to_text = {v: k for k, v in text_to_id.items()}
    anchor_text_set = set(anchor_texts)

    # exact-duplicate check: only flags text reuse that's NOT the expected
    # kind (many pairs sharing one anchor's query_a is by design). What
    # actually matters: a query_b that's verbatim identical to some
    # anchor's query_a (bug #1's exact class from the 45-pair set), or two
    # different anchors somehow sharing identical text.
    for p in pairs:
        if p["query_b"] in anchor_text_set:
            issues.append(f"VERBATIM COLLISION: {p['id']}.query_b is identical to an indexed anchor: {p['query_b']!r}")

    index = create_index("linear", dim=embedder.dim)
    anchor_vectors = embedder.embed_batch(anchor_texts)
    for text, vector in zip(anchor_texts, anchor_vectors):
        index.insert(text_to_id[text], vector)
    print(f"indexed {len(index)} unique anchors\n")

    # anchor-vs-anchor duplicate check: search each anchor against every
    # OTHER anchor (k=2, drop the trivial self-match at similarity 1.0)
    for text, vector in zip(anchor_texts, anchor_vectors):
        own_id = text_to_id[text]
        top2 = index.search(vector, k=2)
        other = next((m for m in top2 if m[0] != own_id), None)
        if other and other[1] >= ANCHOR_DUPLICATE_THRESHOLD:
            issues.append(
                f"ANCHOR COLLISION: {own_id} ({text!r}) is {other[1]:.3f} similar to "
                f"{other[0]} ({id_to_text[other[0]]!r})"
            )

    # per-pair checks against the WHOLE anchor set, not just the pair's own anchor
    query_b_texts = [p["query_b"] for p in pairs]
    query_b_vectors = embedder.embed_batch(query_b_texts)

    for p, vector in zip(pairs, query_b_vectors):
        own_anchor_id = text_to_id[p["query_a"]]
        best_id, best_sim = index.search(vector, k=1)[0]

        if p["category"] == "unrelated" and best_sim >= UNRELATED_COLLISION_THRESHOLD:
            flag = "own anchor" if best_id == own_anchor_id else f"a DIFFERENT anchor ({id_to_text[best_id]!r})"
            issues.append(
                f"UNRELATED COLLISION: {p['id']} query_b {p['query_b']!r} scores "
                f"{best_sim:.3f} against {flag}, expected a low score everywhere"
            )

        if p["category"] == "paraphrase" and best_id != own_anchor_id:
            issues.append(
                f"PARAPHRASE MISMATCH: {p['id']} query_b {p['query_b']!r} top-matches "
                f"{best_id} ({id_to_text[best_id]!r}) instead of its own anchor "
                f"{own_anchor_id} ({p['query_a']!r}), similarity {best_sim:.3f}"
            )

    print(f"{'ISSUES FOUND' if issues else 'NO ISSUES FOUND'} ({len(issues)})\n")
    for issue in issues:
        print(f"  - {issue}")

    sys.exit(1 if issues else 0)


if __name__ == "__main__":
    main()
