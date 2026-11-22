"""
ConcurrencyLimit: admit up to N at once, refuse the rest immediately.

Refusing instead of waiting is the point. A request that waits for a slot
is still holding one of the server's 40 worker threads, so waiting would
use up the very capacity the limit exists to protect.
"""

import threading

import pytest

from limiter import ConcurrencyLimit, Overloaded


def test_admits_up_to_the_limit_then_refuses_immediately():
    limit = ConcurrencyLimit(2)
    with limit.slot():
        with limit.slot():
            assert limit.in_use() == 2
            with pytest.raises(Overloaded):
                with limit.slot():
                    pytest.fail("a third caller must not get in")


def test_a_slot_is_released_on_normal_exit_and_on_error():
    limit = ConcurrencyLimit(1)
    with limit.slot():
        pass
    with pytest.raises(RuntimeError):
        with limit.slot():
            raise RuntimeError("work failed")

    assert limit.in_use() == 0
    with limit.slot():  # still usable
        pass


def test_none_means_unlimited():
    limit = ConcurrencyLimit(None)
    slots = [limit.slot() for _ in range(1000)]
    for s in slots:
        s.__enter__()
    assert limit.in_use() == 1000


def test_never_admits_more_than_the_limit_under_threads():
    limit = ConcurrencyLimit(10)
    barrier = threading.Barrier(100)
    release = threading.Event()
    lock = threading.Lock()
    admitted = refused = 0

    def worker():
        nonlocal admitted, refused
        barrier.wait()
        try:
            with limit.slot():
                with lock:
                    admitted += 1
                release.wait(timeout=2)
        except Overloaded:
            with lock:
                refused += 1

    threads = [threading.Thread(target=worker) for _ in range(100)]
    for t in threads:
        t.start()
    # Let every thread try before anyone leaves a slot.
    while True:
        with lock:
            if admitted + refused == 100:
                break
    release.set()
    for t in threads:
        t.join()

    assert admitted == 10 and refused == 90
    assert limit.in_use() == 0


def test_rejects_a_limit_below_one():
    with pytest.raises(ValueError):
        ConcurrencyLimit(0)
