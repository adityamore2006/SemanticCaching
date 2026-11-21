import numpy as np
import pytest

from cache_router import CacheRouter, EmptyLLMResponse
from llm import LLMError
from cache_store import InMemoryCacheStore


class FakeEmbedder:
    """
    Deterministic stand-in for the real sentence-transformers model, so
    these tests are fast and don't depend on downloading/loading it --
    same philosophy as test_index_contract.py's hand-picked vectors.
    """

    dim = 2
    model_name = "fake-embedder"

    def __init__(self, vectors):
        self.vectors = vectors  # text -> raw vector

    def embed(self, text):
        return np.array(self.vectors[text], dtype=np.float32)


def stub_llm(query):
    """Explicit test double. CacheRouter deliberately has no default llm:
    a default that fabricated answers is exactly what caused misses to be
    cached and replayed as hits."""
    return f"[stub response for: {query}]"


def counting_llm(calls):
    def llm(query):
        calls.append(query)
        return f"real answer to: {query}"
    return llm


def test_first_query_is_always_a_miss():
    embedder = FakeEmbedder({"reset password": [1.0, 0.0]})
    router = CacheRouter("linear", embedder=embedder, llm=stub_llm)

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
    router = CacheRouter("linear", embedder=embedder, llm=stub_llm)
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
    router = CacheRouter("linear", embedder=embedder, llm=stub_llm)
    router.route("reset password")

    result = router.route("change my email")

    assert result.hit is False
    assert result.similarity == pytest.approx(0.7)  # how close it came
    assert len(router.index) == 2  # inserted as its own, separate entry


def test_works_with_hnsw_index_too():
    embedder = FakeEmbedder({"reset password": [1.0, 0.0]})
    router = CacheRouter("hnsw", embedder=embedder, llm=stub_llm)

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

    router = CacheRouter("linear", embedder=embedder, cache_store=store, llm=stub_llm)
    router.route("reset password")

    assert router.cache_store is store
    assert len(store) == 1


def test_restore_rebuilds_the_index_and_serves_hits_from_it():
    # Simulates a restart: a fresh router with an empty index,
    # handed the vectors a previous container had persisted.
    embedder = FakeEmbedder({"reset password": [1.0, 0.0]})
    store = InMemoryCacheStore()
    store.put("q_0", "restored answer")
    router = CacheRouter("hnsw", embedder=embedder, cache_store=store, llm=stub_llm)

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
    router = CacheRouter("linear", embedder=embedder, llm=stub_llm)

    router.restore([("seeded-entry", np.array([0.0, 1.0], dtype=np.float32))])
    router.route("reset password")

    assert router.cache_store.get("q_0") == "[stub response for: reset password]"


def refusing_llm(query):
    raise LLMError("model unavailable")


def test_llm_is_required_and_has_no_fabricating_default():
    # The default used to be a stub returning placeholder text. Constructing
    # without an llm must now fail loudly rather than quietly fabricating.
    embedder = FakeEmbedder({"reset password": [1.0, 0.0]})
    with pytest.raises(TypeError):
        CacheRouter("linear", embedder=embedder)


def test_a_failed_model_call_caches_nothing():
    # A failed call must not leave a placeholder behind that a later
    # identical question would be served as a confident HIT.
    embedder = FakeEmbedder({"a new question": [1.0, 0.0]})
    store = InMemoryCacheStore()
    router = CacheRouter(
        "linear", embedder=embedder, cache_store=store, llm=refusing_llm
    )

    with pytest.raises(LLMError):
        router.route("a new question")

    assert len(store) == 0
    assert len(router.index) == 0

    # And critically: asking again still misses, rather than hitting on
    # something the first attempt left behind.
    with pytest.raises(LLMError):
        router.route("a new question")
    assert len(router.index) == 0


def test_seeded_answers_still_serve_hits_when_the_model_is_down():
    # A model outage must not take the cache offline: a hit never needs it.
    embedder = FakeEmbedder({
        "reset password": [1.0, 0.0],
        "i forgot my password": [0.9, 0.435889894354067],
        "something else entirely": [0.0, 1.0],
    })
    router = CacheRouter("linear", embedder=embedder, llm=refusing_llm)
    router.seed([("reset password", "Use the forgot password link.")])

    hit = router.route("i forgot my password")
    assert hit.hit is True
    assert hit.response == "Use the forgot password link."

    with pytest.raises(LLMError):
        router.route("something else entirely")


def test_seed_populates_without_calling_the_llm():
    calls = []
    embedder = FakeEmbedder({
        "reset password": [1.0, 0.0],
        "cancel plan": [0.0, 1.0],
    })
    store = InMemoryCacheStore()
    router = CacheRouter(
        "linear", embedder=embedder, cache_store=store, llm=counting_llm(calls)
    )

    added = router.seed([
        ("reset password", "Use the forgot password link."),
        ("cancel plan", "Settings then Billing then Cancel."),
    ])

    assert added == 2
    assert calls == []  # the whole point: seeding costs no model calls
    assert len(router.index) == 2
    assert store.get("q_0") == "Use the forgot password link."


def test_a_seeded_answer_is_served_on_a_paraphrase():
    embedder = FakeEmbedder({
        "reset password": [1.0, 0.0],
        "i forgot my password": [0.9, 0.435889894354067],
    })
    router = CacheRouter("linear", embedder=embedder, llm=stub_llm)
    router.seed([("reset password", "Use the forgot password link.")])

    result = router.route("i forgot my password")

    assert result.hit is True
    assert result.response == "Use the forgot password link."


def test_seeding_twice_does_not_duplicate_entries():
    # Re-running the seed must be safe, or a restart would pile up
    # near-identical entries competing for the same neighborhood.
    embedder = FakeEmbedder({"reset password": [1.0, 0.0]})
    router = CacheRouter("linear", embedder=embedder, llm=stub_llm)

    assert router.seed([("reset password", "an answer")]) == 1
    assert router.seed([("reset password", "an answer")]) == 0
    assert len(router.index) == 1


def test_seed_refuses_an_empty_answer():
    # Same guarantee as the LLM path: nothing empty ever reaches the store,
    # whatever produced it.
    embedder = FakeEmbedder({"reset password": [1.0, 0.0]})
    store = InMemoryCacheStore()
    router = CacheRouter("linear", embedder=embedder, cache_store=store, llm=stub_llm)

    with pytest.raises(EmptyLLMResponse):
        router.seed([("reset password", "   ")])
    assert len(store) == 0


def test_an_empty_llm_response_is_never_cached():
    # The failure this prevents: "" is not None, so a cached empty answer
    # passes the hit check and is served as a confident HIT to every
    # future paraphrase, permanently.
    embedder = FakeEmbedder({"reset password": [1.0, 0.0]})
    store = InMemoryCacheStore()
    router = CacheRouter(
        "linear", embedder=embedder, cache_store=store, llm=lambda q: ""
    )

    with pytest.raises(EmptyLLMResponse):
        router.route("reset password")

    # Nothing persisted, so the next attempt gets a clean shot at it.
    assert len(store) == 0
    assert len(router.index) == 0


def test_a_whitespace_only_llm_response_is_never_cached():
    embedder = FakeEmbedder({"reset password": [1.0, 0.0]})
    store = InMemoryCacheStore()
    router = CacheRouter(
        "linear", embedder=embedder, cache_store=store, llm=lambda q: "   \n  "
    )

    with pytest.raises(EmptyLLMResponse):
        router.route("reset password")
    assert len(store) == 0


def test_an_llm_that_raises_leaves_the_cache_untouched():
    # Already true before the empty-response guard, asserted so it stays
    # true: a failed miss must not half-write an entry.
    embedder = FakeEmbedder({"reset password": [1.0, 0.0]})
    store = InMemoryCacheStore()

    def broken(query):
        raise RuntimeError("model is throttling")

    router = CacheRouter("linear", embedder=embedder, cache_store=store, llm=broken)

    with pytest.raises(RuntimeError):
        router.route("reset password")
    assert len(store) == 0
    assert len(router.index) == 0


def test_a_miss_writes_the_store_and_index_together():
    # The invariant the orphan check used to paper over: every miss adds
    # the same entry to both the store and the index, so they stay in step.
    embedder = FakeEmbedder({"reset password": [1.0, 0.0]})
    store = InMemoryCacheStore()
    router = CacheRouter("linear", embedder=embedder, cache_store=store, llm=stub_llm)

    first = router.route("reset password")

    assert first.hit is False
    assert len(store) == 1
    assert len(router.index) == 1
    assert store.get("q_0") == "[stub response for: reset password]"

    second = router.route("reset password")
    assert second.hit is True
    assert second.matched_id == "q_0"


def test_an_orphan_fails_loudly_and_never_serves_none():
    # An id in the index with nothing in the store must not be answered
    # with None at similarity 1.0, and must not be silently regenerated
    # either: that would hide an inconsistent cache. It raises instead,
    # before the LLM is called.
    calls = []
    embedder = FakeEmbedder({"reset password": [1.0, 0.0]})
    store = InMemoryCacheStore()
    router = CacheRouter(
        "linear", embedder=embedder, cache_store=store, llm=counting_llm(calls)
    )
    # An orphan: in the index, absent from the store.
    router.restore([("q_0", np.array([1.0, 0.0], dtype=np.float32))])

    with pytest.raises(RuntimeError, match="q_0"):
        router.route("reset password")

    assert calls == []  # no LLM call
    assert len(store) == 0  # nothing written
    assert len(router.index) == 1  # the inconsistency is left visible


def test_custom_threshold_is_respected():
    # similarity 0.9 would be a hit at the default 0.80 threshold, but not
    # at a stricter 0.95 -- confirms the threshold isn't hardcoded.
    embedder = FakeEmbedder({
        "reset password": [1.0, 0.0],
        "i forgot my password": [0.9, 0.435889894354067],
    })
    router = CacheRouter("linear", embedder=embedder, threshold=0.95, llm=stub_llm)
    router.route("reset password")

    result = router.route("i forgot my password")

    assert result.hit is False
