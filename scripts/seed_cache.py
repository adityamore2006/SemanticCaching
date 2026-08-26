"""
Warm the cache with the verified anchor questions, so a demo starts from a
populated cache instead of an empty one.

Seeds over HTTP against a running server rather than importing the app and
writing to storage directly. Two reasons that's the better shape: every
seeded entry travels the exact code path real traffic does (embed, search,
threshold, miss, store), so seeding can't drift from routing; and the same
script works unchanged against localhost or a deployed instance.

Source data is data/eval_pairs.json's anchors -- the 67 questions already
collision-verified in Phase 2 (knowledge/learned.md section 10), so the
seeded cache has no two entries that mean the same thing.

    python scripts/seed_cache.py
    python scripts/seed_cache.py --url http://<instance-ip>:8000

Against a deployed instance backed by DynamoDB this is a one-time cost:
the entries persist, and later boots reload them. Run locally without a
table and the cache lives only as long as the server, so re-run after a
restart.
"""

import argparse
import json
import os
import time
import urllib.error
import urllib.request

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "eval_pairs.json")


def load_anchors(path=DATA_PATH):
    """Unique query_a values, in first-seen order. Many pairs share an
    anchor (each has a paraphrase, a near-miss, and an unrelated sibling),
    so this deduplicates down to the distinct questions."""
    with open(path) as f:
        pairs = json.load(f)["pairs"]

    seen = {}
    for pair in pairs:
        seen.setdefault(pair["query_a"], None)
    return list(seen)


def post(url, query, timeout=120):
    request = urllib.request.Request(
        f"{url.rstrip('/')}/query",
        data=json.dumps({"query": query}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--limit", type=int, default=None,
                        help="seed only the first N anchors, for a quick check")
    parser.add_argument("--verbose", action="store_true",
                        help="print every anchor as it is seeded")
    args = parser.parse_args()

    anchors = load_anchors()
    if args.limit:
        anchors = anchors[:args.limit]

    print(f"seeding {len(anchors)} anchors into {args.url}")

    added = skipped = failed = 0
    started = time.time()
    for i, anchor in enumerate(anchors, 1):
        try:
            result = post(args.url, anchor)
        except (urllib.error.URLError, TimeoutError) as exc:
            failed += 1
            print(f"  [{i}/{len(anchors)}] FAILED {anchor[:52]!r}: {exc}")
            continue

        # A hit means something already cached is close enough to this
        # anchor, so it needs no entry of its own. On a fresh cache that
        # should never happen: the anchors were verified non-colliding.
        if result["hit"]:
            skipped += 1
            if args.verbose:
                print(f"  [{i}/{len(anchors)}] already cached "
                      f"(sim={result['similarity']:.3f}) {anchor[:48]!r}")
        else:
            added += 1
            if args.verbose:
                print(f"  [{i}/{len(anchors)}] added {anchor[:60]!r}")

    elapsed = time.time() - started
    print(f"\nadded={added}  already_cached={skipped}  failed={failed}  "
          f"in {elapsed:.1f}s")

    try:
        with urllib.request.urlopen(f"{args.url.rstrip('/')}/stats") as response:
            stats = json.load(response)
        print(f"cache now holds {stats['entries']} entries "
              f"at threshold {stats['threshold']}")
    except (urllib.error.URLError, TimeoutError):
        pass


if __name__ == "__main__":
    main()
