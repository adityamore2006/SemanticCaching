"""
Metrics and the metered LLM wrapper.
"""

import threading

import pytest

import metrics as m
from llm import LLMError, LLMTimeout
from metered_llm import MeteredLLM
from metrics import Metrics


def test_counters_start_at_zero_and_add_up():
    metrics = Metrics()
    assert metrics.get(m.HITS) == 0
    metrics.incr(m.HITS)
    metrics.incr(m.HITS, 4)
    assert metrics.get(m.HITS) == 5


def test_percentiles_use_nearest_rank():
    metrics = Metrics()
    for ms in range(1, 101):  # 1..100
        metrics.observe("hit", ms)

    summary = metrics.snapshot()["latency_ms"]["hit"]

    assert summary == {"count": 100, "p50": 50, "p95": 95, "p99": 99, "max": 100}


def test_empty_latency_is_none_not_zero():
    # Zero would read as "instant". No samples means no answer.
    summary = Metrics().snapshot()["latency_ms"]["miss"]
    assert summary["count"] == 0 and summary["p50"] is None


def test_latency_window_is_bounded_and_keeps_the_newest():
    metrics = Metrics(window=10)
    for ms in range(1000):
        metrics.observe("hit", ms)

    summary = metrics.snapshot()["latency_ms"]["hit"]

    assert summary["count"] == 10
    assert summary["max"] == 999


def test_hit_rate_counts_answered_requests_only():
    metrics = Metrics()
    metrics.incr(m.HITS, 3)
    metrics.incr(m.MISSES, 1)
    metrics.incr(m.ERRORS_503, 6)
    assert metrics.snapshot()["requests"]["hit_rate"] == 0.75
    assert Metrics().snapshot()["requests"]["hit_rate"] is None


def test_in_flight_gauge_tracks_peak_and_always_comes_back_down():
    metrics = Metrics()
    with metrics.in_flight():
        with metrics.in_flight():
            assert metrics.snapshot()["in_flight"] == 2
    with pytest.raises(RuntimeError):
        with metrics.in_flight():
            raise RuntimeError("handler blew up")

    snapshot = metrics.snapshot()
    assert snapshot["in_flight"] == 0
    assert snapshot["in_flight_peak"] == 2


def test_reset_zeroes_everything():
    metrics = Metrics()
    metrics.incr(m.HITS)
    metrics.observe("hit", 5)
    metrics.reset()
    snapshot = metrics.snapshot()
    assert snapshot["requests"]["hits"] == 0
    assert snapshot["latency_ms"]["hit"]["count"] == 0


def test_no_lost_updates_under_threads():
    metrics = Metrics()
    threads = [
        threading.Thread(target=lambda: [metrics.incr(m.HITS) for _ in range(2000)])
        for _ in range(20)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert metrics.get(m.HITS) == 40000


# --- MeteredLLM ---------------------------------------------------------


def test_metered_llm_counts_calls_and_passes_the_answer_through():
    metrics = Metrics()
    llm = MeteredLLM(lambda q: f"answer to {q}", metrics)

    assert llm("hi") == "answer to hi"
    llm("again")

    snapshot = metrics.snapshot()
    assert snapshot["model"] == {"calls": 2, "errors": 0, "timeouts": 0}
    assert snapshot["latency_ms"]["model"]["count"] == 2


def _raises(exc):
    def llm(query):
        raise exc
    return llm


def test_metered_llm_separates_timeouts_from_other_errors_and_reraises():
    metrics = Metrics()

    with pytest.raises(LLMTimeout):
        MeteredLLM(_raises(LLMTimeout("slow")), metrics)("q")
    with pytest.raises(LLMError):
        MeteredLLM(_raises(LLMError("down")), metrics)("q")

    snapshot = metrics.snapshot()
    # A timeout is a call that failed, but it is counted once, as a timeout.
    assert snapshot["model"] == {"calls": 2, "errors": 1, "timeouts": 1}
    assert snapshot["latency_ms"]["model"]["count"] == 2
