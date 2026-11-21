"""
Restore the running cache to the curated corpus, in one command.

    python scripts/reset_cache.py                        # localhost:8000
    python scripts/reset_cache.py --url http://host:8000

Needs the service's RESET_TOKEN, via --token or $RESET_TOKEN, and the service
must be started with the same token or the endpoint stays disabled.

It talks to the running service rather than editing the database file,
because the in-memory index is also part of the cache and only the service
holds it. It verifies the result instead of trusting the response.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

CANONICAL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "seed_answers.json"
)

# A question the curated corpus should answer, to check the reset restored a
# working cache rather than merely a full one.
KNOWN_QUESTION = "Is it possible to combine my two separate accounts?"


def request(url, method="GET", body=None, headers=None, timeout=180):
    req = urllib.request.Request(
        url,
        method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.status, json.load(response)


def canonical_count():
    with open(CANONICAL_PATH) as f:
        return len(json.load(f)["answers"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--token", default=os.environ.get("RESET_TOKEN"))
    args = parser.parse_args()
    base = args.url.rstrip("/")

    if not args.token:
        parser.error(
            "no token. Pass --token or set RESET_TOKEN.\n"
            "The service must also be started with the same RESET_TOKEN, "
            "or the endpoint stays disabled."
        )

    expected = canonical_count()
    print(f"resetting {base} to the canonical {expected} entries")

    try:
        _, before = request(f"{base}/stats")
        print(f"  before: {before['entries']} entries")
    except (urllib.error.URLError, TimeoutError) as exc:
        sys.exit(f"could not reach {base}: {exc}")

    try:
        _, result = request(
            f"{base}/admin/reset", method="POST", headers={"X-Reset-Token": args.token}
        )
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            sys.exit("reset endpoint is disabled: start the service with RESET_TOKEN set")
        if exc.code == 401:
            sys.exit("token rejected: it must match the service's RESET_TOKEN")
        sys.exit(f"reset failed ({exc.code}): {exc.read().decode()[:200]}")

    print(f"  removed {result['removed']}, restored {result['restored']}")

    # Everything below is verification. The reset already reported success;
    # this is checking whether that report was true.
    failures = []

    _, after = request(f"{base}/stats")
    if after["entries"] != expected:
        failures.append(f"expected {expected} entries, found {after['entries']}")

    _, hit = request(f"{base}/query", method="POST", body={"query": KNOWN_QUESTION})
    if not hit["hit"]:
        failures.append(f"a seeded question missed (similarity {hit['similarity']})")

    if failures:
        print("\nreset reported success but verification failed:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)

    print(f"  verified: {expected} entries, seeded question hits")


if __name__ == "__main__":
    main()
