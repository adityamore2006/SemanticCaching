"""
Concurrency tests for CacheRouter.

FastAPI runs the sync /query handler on a pool of 40 threads that all share
one router, so route() must be safe to call from many threads at once.

CPython switches threads every 5 ms by default, and the race windows here
are microseconds wide, so a plain stress test would pass by luck. The tests
shrink the switch interval to make an unlucky interleaving likely. That does
not make the code incorrect; it makes the latent bugs show up in seconds
instead of in production.
"""

import hashlib
import sys
import threading

import numpy as np
import pytest

from cache_router import CacheRouter
from cache_store import InMemoryCacheStore
from embedder import Embedder
from fake_llm import FakeLLM

DIM = 64
THREADS = 20
TRIALS = 40
YOUNG_CACHE = 3  # a young index changes its top layer often, which is the worst case


class HashEmbedder(Embedder):
    """Same text, same vector. Different texts are nearly orthogonal, so every
    distinct query is a miss and goes through the write path."""

    dim = DIM
    model_name = "hash"

    def embed(self, text):
        seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
        return np.random.default_rng(seed).standard_normal(DIM).astype(np.float32)

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]


@pytest.fixture
def aggressive_thread_switching():
    original = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    yield
    sys.setswitchinterval(original)


def make_router(prepopulate):
    store = InMemoryCacheStore()
    rng = np.random.default_rng(0)
    for i in range(prepopulate):
        store.put(f"q_{i}", "seed answer", rng.standard_normal(DIM).astype(np.float32))
    router = CacheRouter("hnsw", HashEmbedder(), FakeLLM(), cache_store=store)
    router.restore((id, vector) for id, vector, _ in store.all_items())
    return router, store


def run_threads(router, trial, queries_per_thread):
    """Start THREADS threads together; return the exceptions they raised."""
    barrier = threading.Barrier(THREADS)
    errors = []

    def worker(i):
        barrier.wait()
        try:
            for j in range(queries_per_thread):
                router.route(f"trial {trial} thread {i} question {j}")
        except Exception as exc:  # noqa: BLE001 - the test reports whatever escapes
            errors.append(repr(exc))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return errors


def test_concurrent_misses_lose_nothing_and_raise_nothing(aggressive_thread_switching):
    problems = []
    for trial in range(TRIALS):
        router, store = make_router(YOUNG_CACHE)
        errors = run_threads(router, trial, queries_per_thread=1)

        expected = YOUNG_CACHE + THREADS
        if errors:
            problems.append(f"trial {trial}: raised {errors}")
        elif len(store) != expected or len(router.index) != expected:
            problems.append(
                f"trial {trial}: expected {expected} entries, "
                f"store has {len(store)}, index has {len(router.index)}"
            )

    assert not problems, f"{len(problems)}/{TRIALS} trials went wrong:\n" + "\n".join(problems[:5])


def test_every_entry_gets_a_unique_id(aggressive_thread_switching):
    """Two threads reading the same counter value would write over each other."""
    collisions = []
    for trial in range(TRIALS):
        router, store = make_router(YOUNG_CACHE)
        run_threads(router, trial, queries_per_thread=1)
        ids = [id for id, _, _ in store.all_items()]
        if len(ids) != len(set(ids)):
            collisions.append(trial)
        # Every minted id must be accounted for: ids q_0..q_(n-1) with no gaps lost.
        if router._next_id != YOUNG_CACHE + THREADS:
            collisions.append(f"trial {trial}: counter is {router._next_id}")

    assert not collisions, f"id problems in trials: {collisions[:5]}"


def test_concurrent_hits_and_misses_do_not_interfere(aggressive_thread_switching):
    """Searches (hits) run while other threads insert (misses)."""
    router, store = make_router(YOUNG_CACHE)
    known = "seed question"
    router.seed([(known, "known answer")])
    errors = []
    barrier = threading.Barrier(THREADS)

    def worker(i):
        barrier.wait()
        try:
            for j in range(10):
                if i % 2:
                    result = router.route(known)
                    assert result.hit and result.response == "known answer"
                else:
                    router.route(f"thread {i} new question {j}")
        except Exception as exc:  # noqa: BLE001
            errors.append(repr(exc))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors[:3]
    writers = THREADS // 2
    assert len(store) == len(router.index) == YOUNG_CACHE + 1 + writers * 10
