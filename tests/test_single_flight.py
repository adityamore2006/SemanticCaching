"""
SingleFlight: when several threads ask for the same key at once, one runs
the work and the rest wait for and share its result.
"""

import threading
import time

import pytest

from single_flight import SingleFlight


def run_together(n, target):
    """Start n threads at the same moment; return their results in order."""
    barrier = threading.Barrier(n)
    results = [None] * n

    def worker(i):
        barrier.wait()
        try:
            results[i] = ("ok", target(i))
        except Exception as exc:  # noqa: BLE001 - tests inspect what was raised
            results[i] = ("raised", exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def test_concurrent_callers_with_one_key_share_one_execution():
    flights = SingleFlight()
    executions = []

    def work():
        executions.append(1)
        time.sleep(0.2)
        return "answer"

    results = run_together(20, lambda i: flights.do("k", work))

    assert len(executions) == 1
    assert all(kind == "ok" and value[0] == "answer" for kind, value in results)
    assert sum(1 for _, (_, led) in results if led) == 1  # exactly one leader


def test_different_keys_run_in_parallel_not_in_turn():
    flights = SingleFlight()

    def work():
        time.sleep(0.3)
        return "done"

    started = time.perf_counter()
    run_together(10, lambda i: flights.do(f"key {i}", work))
    elapsed = time.perf_counter() - started

    # In turn this would take 10 x 0.3 = 3 s.
    assert elapsed < 1.0


def test_a_failure_reaches_every_waiter_and_is_not_remembered():
    flights = SingleFlight()
    executions = []

    def failing():
        executions.append(1)
        time.sleep(0.2)
        raise RuntimeError("model down")

    results = run_together(10, lambda i: flights.do("k", failing))

    assert len(executions) == 1
    assert all(kind == "raised" and "model down" in str(value) for kind, value in results)

    # The failure was not cached: the next caller starts a fresh attempt.
    assert flights.do("k", lambda: "recovered") == ("recovered", True)


def test_it_is_not_a_cache_each_finished_flight_is_forgotten():
    flights = SingleFlight()
    executions = []

    def work():
        executions.append(1)
        return len(executions)

    assert flights.do("k", work) == (1, True)
    assert flights.do("k", work) == (2, True)


def test_no_flight_is_left_registered_after_success_or_failure():
    flights = SingleFlight()
    flights.do("a", lambda: 1)
    with pytest.raises(ValueError):
        flights.do("b", _raise_value_error)
    assert flights.in_progress() == 0


def _raise_value_error():
    raise ValueError("boom")
