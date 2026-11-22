"""
In-process metrics: counters, a gauge, and recent-latency percentiles.

Deliberately small. One lock guards everything, and every operation holds
it for microseconds, so it is safe to call from all 40 request threads.
Latencies keep only the most recent WINDOW samples per kind, so memory is
bounded and the percentiles describe what the service is doing now, not
its whole history.

The numbers are exposed as JSON at GET /metrics. They exist so each
concurrency fix can be judged by measurement ("model calls for 20 identical
questions: 20 -> 1") instead of by feel.
"""

import math
import threading
import time
from collections import Counter, deque
from contextlib import contextmanager

WINDOW = 2000

# Counter names, listed once so a typo in a call site is easy to spot.
REQUESTS = "requests"          # every /query that reached the handler
HITS = "hits"
MISSES = "misses"             # asked the model itself
COALESCED = "coalesced"       # shared the answer of an identical in-flight miss
ERRORS_503 = "errors_503"      # model failed or returned nothing usable
ERRORS_504 = "errors_504"      # model timed out
ERRORS_500 = "errors_500"      # anything unexpected
MODEL_CALLS = "model_calls"
MODEL_ERRORS = "model_errors"
MODEL_TIMEOUTS = "model_timeouts"

# Latency kinds. "queue_wait" is the time a request waited for a free
# worker thread before its handler started: the number that exposes a
# saturated thread pool, which handler timing alone cannot see.
LATENCY_KINDS = ("hit", "miss", "coalesced", "error", "model", "queue_wait")


def _percentile(sorted_values, p):
    """Nearest-rank percentile of an already-sorted list."""
    rank = math.ceil(p / 100 * len(sorted_values))
    return sorted_values[max(rank - 1, 0)]


class Metrics:
    def __init__(self, window=WINDOW):
        self._lock = threading.Lock()
        self._window = window
        self._reset_locked()

    def _reset_locked(self):
        self._counters = Counter()
        self._latency = {kind: deque(maxlen=self._window) for kind in LATENCY_KINDS}
        self._in_flight = 0
        self._in_flight_peak = 0
        self._started = time.time()

    def reset(self):
        with self._lock:
            self._reset_locked()

    def incr(self, name, amount=1):
        with self._lock:
            self._counters[name] += amount

    def get(self, name):
        with self._lock:
            return self._counters[name]

    def observe(self, kind, milliseconds):
        with self._lock:
            self._latency[kind].append(milliseconds)

    @contextmanager
    def in_flight(self):
        """Count a request as running while its handler is inside this block."""
        with self._lock:
            self._in_flight += 1
            self._in_flight_peak = max(self._in_flight_peak, self._in_flight)
        try:
            yield
        finally:
            with self._lock:
                self._in_flight -= 1

    def snapshot(self):
        # Copy under the lock, sort and summarize outside it.
        with self._lock:
            counters = Counter(self._counters)
            samples = {kind: list(values) for kind, values in self._latency.items()}
            in_flight, peak = self._in_flight, self._in_flight_peak
            uptime = time.time() - self._started

        hits, misses, coalesced = counters[HITS], counters[MISSES], counters[COALESCED]
        answered = hits + misses + coalesced
        return {
            "uptime_seconds": round(uptime, 1),
            "in_flight": in_flight,
            "in_flight_peak": peak,
            "requests": {
                "total": counters[REQUESTS],
                "hits": hits,
                "misses": misses,
                "coalesced": coalesced,
                "hit_rate": round(hits / answered, 4) if answered else None,
                "errors_503": counters[ERRORS_503],
                "errors_504": counters[ERRORS_504],
                "errors_500": counters[ERRORS_500],
            },
            "model": {
                "calls": counters[MODEL_CALLS],
                "errors": counters[MODEL_ERRORS],
                "timeouts": counters[MODEL_TIMEOUTS],
            },
            "latency_ms": {kind: self._summarize(values) for kind, values in samples.items()},
        }

    @staticmethod
    def _summarize(values):
        if not values:
            return {"count": 0, "p50": None, "p95": None, "p99": None, "max": None}
        values.sort()
        return {
            "count": len(values),
            "p50": round(_percentile(values, 50), 1),
            "p95": round(_percentile(values, 95), 1),
            "p99": round(_percentile(values, 99), 1),
            "max": round(values[-1], 1),
        }
