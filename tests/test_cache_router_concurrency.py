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
import time

import numpy as np
import pytest

from cache_router import CacheRouter
from cache_store import InMemoryCacheStore
from embedder import Embedder
from fake_llm import FakeLLM
from llm import LLMError

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


# --- single-flight: identical concurrent misses share one model call ------

def run_identical(router, query, n=THREADS):
    """n threads ask the same question at the same moment."""
    barrier = threading.Barrier(n)
    outcomes = [None] * n

    def worker(i):
        barrier.wait()
        try:
            outcomes[i] = router.route(query)
        except Exception as exc:  # noqa: BLE001
            outcomes[i] = exc

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return outcomes


def router_with(llm, prepopulate=YOUNG_CACHE):
    router, store = make_router(prepopulate)
    router.llm = llm
    return router, store


def test_identical_concurrent_misses_make_one_model_call_and_one_entry():
    llm = FakeLLM(latency=0.3)
    router, store = router_with(llm)

    outcomes = run_identical(router, "how do I export my data?")

    assert llm.calls == 1
    assert len(store) == len(router.index) == YOUNG_CACHE + 1
    assert {o.response for o in outcomes} == {"fake answer to: how do I export my data?"}
    assert sum(o.coalesced for o in outcomes) == THREADS - 1
    assert not any(o.hit for o in outcomes)  # nobody found it in the cache


def test_different_concurrent_misses_do_not_wait_for_each_other():
    llm = FakeLLM(latency=0.3)
    router, store = router_with(llm)
    barrier = threading.Barrier(10)

    def worker(i):
        barrier.wait()
        router.route(f"unrelated question number {i}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    started = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - started

    assert llm.calls == 10
    assert elapsed < 1.0  # one after another would be 3 s


def test_a_model_failure_is_shared_not_cached_and_retried_by_the_next_caller():
    router, store = router_with(FakeLLM(latency=0.2, error=LLMError("model down")))

    outcomes = run_identical(router, "will this fail?")

    assert router.llm.calls == 1
    assert all(isinstance(o, LLMError) for o in outcomes)
    assert len(store) == YOUNG_CACHE  # nothing cached

    router.llm = FakeLLM()  # the model recovers
    assert router.route("will this fail?").response == "fake answer to: will this fail?"


def test_case_and_spacing_do_not_defeat_the_match():
    llm = FakeLLM(latency=0.3)
    router, store = router_with(llm)
    barrier = threading.Barrier(3)

    def worker(text):
        barrier.wait()
        router.route(text)

    variants = ["How do I reset my password?", "how do i  reset my password?", "  HOW DO I RESET MY PASSWORD?  "]
    threads = [threading.Thread(target=worker, args=(v,)) for v in variants]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert llm.calls == 1


def test_an_answer_stored_while_a_caller_was_getting_ready_is_found_not_regenerated():
    """The gap: a caller searches, misses, and only then becomes the leader.
    If a previous flight finished in between, its answer is already stored,
    so the new leader must look again before spending a model call."""
    llm = FakeLLM()
    router, store = router_with(llm)
    query = "a question someone just answered"
    vector = router.embedder.embed(query)
    router._add(query_vector=vector, response="stored a moment ago", source="llm")

    result = router._fill(query, vector)

    assert result.hit and result.response == "stored a moment ago"
    assert llm.calls == 0
