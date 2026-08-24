import numpy as np
import pytest

from cache_router import CacheRouter
from cache_store import InMemoryCacheStore


class FakeEmbedder:
    """
    Deterministic stand-in for the real sentence-transformers model, so
    these tests are fast and don't depend on downloading/loading it --
    same philosophy as test_index_contract.py's hand-picked vectors.
    """

    dim = 2

    def __init__(self, vectors):
        self.vectors = vectors  # text -> raw vector

    def embed(self, text):
        return np.array(self.vectors[text], dtype=np.float32)


def counting_llm(calls):
    def llm(query):
        calls.append(query)
        return f"real answer to: {query}"
    return llm


def test_first_query_is_always_a_miss():
    embedder = FakeEmbedder({"reset password": [1.0, 0.0]})
    router = CacheRouter("linear", embedder=embedder)

    result = router.route("reset password")

    assert result.hit is False
    assert result.similarity is None
    assert result.matched_id is None
    assert len(router.index) == 1
    assert len(router.cache_store) == 1


def test_miss_calls_llm_and_caches_the_response():
    calls = []
    embedder = FakeEmbedder({"reset password": [1.0, 0.0]})
    router = CacheRouter("linear", embedder=embedder, llm=counting_llm(calls))

    result = router.route("reset password")

    assert calls == ["reset password"]
    assert result.response == "real answer to: reset password"
    assert router.cache_store.get("q_0") == "real answer to: reset password"


def test_repeated_identical_query_is_a_hit_no_second_llm_call():
    calls = []
    embedder = FakeEmbedder({"reset password": [1.0, 0.0]})
    router = CacheRouter("linear", embedder=embedder, llm=counting_llm(calls))

    first = router.route("reset password")
    second = router.route("reset password")

    assert calls == ["reset password"]  # llm only ever called once
    assert second.hit is True
    assert second.response == first.response
    assert second.similarity == pytest.approx(1.0)
    assert len(router.index) == 1  # no new entry inserted on a hit


def test_paraphrase_above_threshold_is_a_hit():
    # cos_sim([1,0], [0.9, sqrt(1 - 0.9^2)]) == 0.9 exactly, both unit vectors.
    embedder = FakeEmbedder({
        "reset password": [1.0, 0.0],
        "i forgot my password": [0.9, 0.435889894354067],
    })
    router = CacheRouter("linear", embedder=embedder)
    router.route("reset password")

    result = router.route("i forgot my password")

    assert result.hit is True
    assert result.similarity == pytest.approx(0.9)
    assert result.response == "[stub response for: reset password]"


def test_near_miss_below_threshold_is_a_miss_and_gets_inserted():
    # cos_sim([1,0], [0.7, sqrt(1 - 0.7^2)]) == 0.7 exactly, below the 0.80 threshold.
    embedder = FakeEmbedder({
        "reset password": [1.0, 0.0],
        "change my email": [0.7, 0.7141428428542852],
    })
    router = CacheRouter("linear", embedder=embedder)
    router.route("reset password")

    result = router.route("change my email")

    assert result.hit is False
    assert result.similarity == pytest.approx(0.7)  # how close it came
    assert len(router.index) == 2  # inserted as its own, separate entry


def test_works_with_hnsw_index_too():
    embedder = FakeEmbedder({"reset password": [1.0, 0.0]})
    router = CacheRouter("hnsw", embedder=embedder)

    first = router.route("reset password")
    second = router.route("reset password")

    assert first.hit is False
    assert second.hit is True
    assert second.response == first.response


def test_an_empty_injected_cache_store_is_still_used():
    # Regression: __init__ used `cache_store or InMemoryCacheStore()`, and
    # CacheStore implements __len__, so an EMPTY store was falsy and got
    # silently swapped for an in-memory one. A freshly deployed persistent
    # store is empty by definition, so the deployed cache would have
    # written every entry to a throwaway dict and persisted nothing --
    # every query missing forever, with no failure signal.
    embedder = FakeEmbedder({"reset password": [1.0, 0.0]})
    store = InMemoryCacheStore()
    assert len(store) == 0

    router = CacheRouter("linear", embedder=embedder, cache_store=store)
    router.route("reset password")

    assert router.cache_store is store
    assert len(store) == 1


def test_restore_rebuilds_the_index_and_serves_hits_from_it():
    # Simulates a Lambda cold start: a fresh router with an empty index,
    # handed the vectors a previous container had persisted.
    embedder = FakeEmbedder({"reset password": [1.0, 0.0]})
    store = InMemoryCacheStore()
    store.put("q_0", "restored answer")
    router = CacheRouter("hnsw", embedder=embedder, cache_store=store)

    restored = router.restore([("q_0", np.array([1.0, 0.0], dtype=np.float32))])

    assert restored == 1
    result = router.route("reset password")
    assert result.hit is True
    assert result.response == "restored answer"


def test_restore_resumes_the_id_counter_instead_of_overwriting_entries():
    # The failure this guards: a restored router whose counter restarted
    # at 0 would mint "q_0" again on the next miss, overwriting a cached
    # entry that still exists in the index -- so a later query could match
    # the old vector and be served the new, unrelated response.
    calls = []
    embedder = FakeEmbedder({
        "reset password": [1.0, 0.0],
        "change my email": [0.7, 0.7141428428542852],
    })
    store = InMemoryCacheStore()
    store.put("q_0", "restored answer")
    router = CacheRouter(
        "linear", embedder=embedder, cache_store=store, llm=counting_llm(calls)
    )
    router.restore([("q_0", np.array([1.0, 0.0], dtype=np.float32))])

    # Below the 0.80 threshold, so this is a miss and mints a new id.
    router.route("change my email")

    assert store.get("q_0") == "restored answer"  # untouched
    assert store.get("q_1") == "real answer to: change my email"
    assert len(router.index) == 2


def test_restore_ignores_ids_that_do_not_follow_the_counter_format():
    # Ids that didn't come from the counter shouldn't corrupt it.
    embedder = FakeEmbedder({"reset password": [1.0, 0.0]})
    router = CacheRouter("linear", embedder=embedder)

    router.restore([("seeded-entry", np.array([0.0, 1.0], dtype=np.float32))])
    router.route("reset password")

    assert router.cache_store.get("q_0") == "[stub response for: reset password]"


def test_custom_threshold_is_respected():
    # similarity 0.9 would be a hit at the default 0.80 threshold, but not
    # at a stricter 0.95 -- confirms the threshold isn't hardcoded.
    embedder = FakeEmbedder({
        "reset password": [1.0, 0.0],
        "i forgot my password": [0.9, 0.435889894354067],
    })
    router = CacheRouter("linear", embedder=embedder, threshold=0.95)
    router.route("reset password")

    result = router.route("i forgot my password")

    assert result.hit is False
