"""
Fire a set of queries at a running cache and show what it decided.

Meant for poking at the system by hand: the interesting behaviour is not
"does it return something", it's *where* the threshold falls and which
near-misses it refuses. Run against a seeded cache (scripts/seed_cache.py)
so queries have something to match against.

    python scripts/try_queries.py                       # the built-in tour
    python scripts/try_queries.py "your own question"   # one-off
    python scripts/try_queries.py --url http://<ip>:8000
"""

import argparse
import json
import urllib.request

# Grouped by what each case is meant to prove, because a list of queries
# with no expectation attached can't tell you when the cache is wrong.
TOUR = [
    ("should HIT: near-identical wording", [
        "Can I merge two accounts into one?",
        "Is it possible to combine my two separate accounts?",
        "I deleted something by mistake, can I get it back?",
    ]),
    ("should MISS: same topic, different question (the dangerous ones)", [
        "Can I delete one of my two accounts?",
        "How do I permanently erase a file so it can't be recovered?",
        "Why did my password stop working after I changed it?",
    ]),
    ("should MISS: nothing to do with the cached set", [
        "What's the weather in Chapel Hill?",
        "Write me a haiku about databases",
    ]),
    # The most informative group: two casual rephrasings that land either
    # side of 0.80. "my login isn't working" clears it at ~0.82 and is
    # served; "how do I join my accounts together" falls short at ~0.77
    # and pays for a fresh answer despite meaning exactly what a cached
    # entry means. That second one is the cost of a safety-weighted
    # threshold made concrete, not a bug (knowledge/learned.md section 11).
    ("right at the boundary: one clears 0.80, one doesn't", [
        "my login isn't working",
        "how do I join my accounts together",
    ]),
]


def post(url, query, timeout=120):
    request = urllib.request.Request(
        f"{url.rstrip('/')}/query",
        data=json.dumps({"query": query}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def run(url, queries, header=None):
    if header:
        print(f"\n{header}")
        print("-" * 78)
    for query in queries:
        result = post(url, query)
        sim = f"{result['similarity']:.4f}" if result["similarity"] is not None else "n/a"
        verdict = "HIT " if result["hit"] else "MISS"
        print(f"  {verdict}  sim={sim:>7}  {result['latency_ms']:>7.1f}ms   {query}")
        if result["hit"]:
            print(f"          served from {result['matched_id']}: {result['response'][:60]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("queries", nargs="*", help="ask your own instead of the tour")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    if args.queries:
        run(args.url, args.queries)
    else:
        for header, queries in TOUR:
            run(args.url, queries, header)

    with urllib.request.urlopen(f"{args.url.rstrip('/')}/stats") as response:
        stats = json.load(response)
    print(f"\nentries={stats['entries']}  requests={stats['requests']}  "
          f"hit_rate={stats['hit_rate']}  threshold={stats['threshold']}")


if __name__ == "__main__":
    main()
