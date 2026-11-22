"""
Load test: measure how the service behaves under concurrent traffic.

Starts the real app (real embedder, HNSW index, SQLite file in a temp dir,
real uvicorn with its default 40-thread pool) with one substitution: the
model is a FakeLLM that sleeps `--latency` seconds. That makes a slow model
repeatable, free, and independent of Ollama.

Run from the repo root:
    .venv/bin/python scripts/loadtest.py
    .venv/bin/python scripts/loadtest.py --latency 3 --json results.json

Each scenario prints what it saw and what a healthy service would show.
Model-call counts come from the service's own /metrics, so this script also
checks that /metrics tells the truth. Latencies are measured by the client.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def serve(port, latency):
    """Child-process mode: run the real app with a slow fake model."""
    sys.path.insert(0, os.path.join(ROOT, "src"))
    import api
    import uvicorn
    from fake_llm import FakeLLM

    llm = FakeLLM(latency=latency)
    api._build_llm = lambda: llm
    uvicorn.run(api.app, host="127.0.0.1", port=port, log_level="warning")


class Client:
    def __init__(self, port):
        self.base = f"http://127.0.0.1:{port}"

    def call(self, path, body=None, timeout=120):
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(
            self.base + path, data=data, headers={"Content-Type": "application/json"}
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status, payload = response.status, json.load(response)
        except urllib.error.HTTPError as exc:
            status, payload = exc.code, json.load(exc)
        except Exception as exc:  # noqa: BLE001 - report, don't crash the run
            status, payload = 0, {"error": repr(exc)}
        return status, (time.perf_counter() - started) * 1000, payload

    def query(self, text):
        return self.call("/query", {"query": text})

    def metrics(self):
        return self.call("/metrics")[2]


def pct(values, p):
    values = sorted(values)
    return values[min(len(values) - 1, int(round(p / 100 * (len(values) - 1))))]


def line(label, values):
    print(
        f"  {label}: n={len(values)}  p50={pct(values, 50):.0f}ms  "
        f"p95={pct(values, 95):.0f}ms  max={max(values):.0f}ms"
    )


def run(client, latency):
    seed = list(json.load(open(os.path.join(ROOT, "data", "seed_answers.json")))["answers"])
    out = {"fake_model_latency_s": latency}

    client.query(seed[0])  # warm the embedder so no number includes a cold start

    print("\n1. Cache hits, one at a time (healthy baseline)")
    lat = []
    for q in (seed * 3)[:30]:
        status, ms, body = client.query(q)
        assert status == 200 and body["hit"], (status, body)
        lat.append(ms)
    line("hit", lat)
    out["hit_p50_ms"], out["hit_p95_ms"] = pct(lat, 50), pct(lat, 95)

    print("\n2. One new question, alone, then asked again")
    q = "What is your policy on refunds for annual plans?"
    _, first_ms, first = client.query(q)
    _, second_ms, second = client.query(q)
    print(f"  first : hit={first['hit']} {first_ms:.0f}ms   second: hit={second['hit']} {second_ms:.0f}ms")

    print("\n3. 20 IDENTICAL new questions at the same moment")
    before = client.metrics()
    q = "How do I migrate my data from the legacy dashboard?"
    t0 = time.perf_counter()
    with ThreadPoolExecutor(20) as pool:
        results = list(pool.map(lambda _: client.query(q), range(20)))
    wall = time.perf_counter() - t0
    after = client.metrics()
    calls = after["model"]["calls"] - before["model"]["calls"]
    added = after["entries"] - before["entries"]
    print(f"  wall time {wall:.1f}s   statuses {sorted({r[0] for r in results})}")
    print(f"  model calls: {calls}   (ideal 1)")
    print(f"  entries added: {added}   (ideal 1)")
    shared = after["requests"]["coalesced"] - before["requests"]["coalesced"]
    print(f"  requests that shared another's answer: {shared}   (ideal 19)")
    out["identical_model_calls"], out["identical_entries_added"] = calls, added
    out["identical_coalesced"] = shared

    print("\n4. 20 DIFFERENT new questions at the same moment")
    before = client.metrics()
    questions = [f"Unrelated topic {i}: how do I configure feature number {i * 7919}?" for i in range(20)]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(20) as pool:
        results = list(pool.map(client.query, questions))
    wall = time.perf_counter() - t0
    calls = client.metrics()["model"]["calls"] - before["model"]["calls"]
    print(f"  wall time {wall:.1f}s   (ideal about {latency:.0f}s: they should not wait for each other)")
    print(f"  model calls: {calls}   (ideal 20)   statuses {sorted({r[0] for r in results})}")
    out["distinct_wall_s"], out["distinct_model_calls"] = round(wall, 2), calls

    print("\n5. Cached questions while 60 different slow questions are in flight")
    before = client.metrics()
    flood_pool = ThreadPoolExecutor(60)
    flood_questions = [f"Flood question {i}: explain topic {i * 104729} in detail" for i in range(60)]
    flood = [flood_pool.submit(client.query, q) for q in flood_questions]
    time.sleep(0.7)  # let the flood fill the server

    health_ms = client.call("/health")[1]
    with ThreadPoolExecutor(10) as pool:
        lat = [r[1] for r in pool.map(client.query, (seed * 2)[:10])]
    line("cached question during flood", lat)
    print(f"  /health during flood: {health_ms:.0f}ms")

    statuses = sorted({f.result()[0] for f in flood})
    flood_pool.shutdown()
    after = client.metrics()
    print(f"  flood finished, statuses {statuses}")
    print(
        f"  server's own view: queue_wait p95 = {after['latency_ms']['queue_wait']['p95']}ms, "
        f"in_flight_peak = {after['in_flight_peak']}"
    )
    out["flood_hit_p50_ms"], out["flood_hit_max_ms"] = pct(lat, 50), max(lat)
    out["flood_health_ms"] = round(health_ms)
    out["flood_statuses"] = statuses
    out["flood_in_flight_peak"] = after["in_flight_peak"]

    print("\n6. Counter accuracy: 400 concurrent cached questions")
    before = client.metrics()["requests"]["total"]
    with ThreadPoolExecutor(50) as pool:
        list(pool.map(lambda i: client.query(seed[i % len(seed)]), range(400)))
    counted = client.metrics()["requests"]["total"] - before
    print(f"  sent 400, /metrics counted {counted}")
    out["counter_lost"] = 400 - counted

    print("\nFinal /metrics:")
    print(json.dumps(client.metrics(), indent=2))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--latency", type=float, default=3.0, help="fake model latency in seconds")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--json", metavar="PATH", help="also write the headline numbers here")
    parser.add_argument("--serve", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.serve:
        serve(args.port, args.latency)
        return

    env = dict(os.environ, CACHE_DB_PATH=os.path.join(tempfile.mkdtemp(prefix="loadtest-"), "cache.db"))
    env["SEED_ON_EMPTY"] = "true"
    child = subprocess.Popen(
        [sys.executable, os.path.abspath(__file__), "--serve",
         "--port", str(args.port), "--latency", str(args.latency)],
        env=env, cwd=ROOT,
    )
    client = Client(args.port)
    try:
        for _ in range(120):
            if client.call("/health", timeout=2)[0] == 200:
                break
            time.sleep(1)
        else:
            sys.exit("server did not come up")
        print(f"server up, fake model latency {args.latency}s, {client.metrics()['entries']} seeded entries")
        results = run(client, args.latency)
    finally:
        child.terminate()
        child.wait(timeout=10)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nheadline numbers written to {args.json}")


if __name__ == "__main__":
    main()
