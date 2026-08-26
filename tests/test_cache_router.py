import numpy as np
import pytest

from cache_router import CacheRouter, EmptyLLMResponse
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
    router = CacheRouter("linear", embedder=embedder)
    router.seed([("reset password", "Use the forgot password link.")])

    result = router.route("i forgot my password")

    assert result.hit is True
    assert result.response == "Use the forgot password link."


def test_seeding_twice_does_not_duplicate_entries():
    # Re-running the seed must be safe, or a restart would pile up
    # near-identical entries competing for the same neighborhood.
    embedder = FakeEmbedder({"reset password": [1.0, 0.0]})
    router = CacheRouter("linear", embedder=embedder)

    assert router.seed([("reset password", "an answer")]) == 1
    assert router.seed([("reset password", "an answer")]) == 0
    assert len(router.index) == 1


def test_seed_refuses_an_empty_answer():
    # Same guarantee as the LLM path: nothing empty ever reaches the store,
    # whatever produced it.
    embedder = FakeEmbedder({"reset password": [1.0, 0.0]})
    store = InMemoryCacheStore()
    router = CacheRouter("linear", embedder=embedder, cache_store=store)

    with pytest.raises(EmptyLLMResponse):
        router.seed([("reset password", "   ")])
    assert len(store) == 0


def test_an_empty_llm_response_is_never_cached():
    # The failure this prevents: "" is not None, so a cached empty answer
    # passes the orphan check and is served as a confident HIT to every
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
        raise RuntimeError("bedrock is throttling")

    router = CacheRouter("linear", embedder=embedder, cache_store=store, llm=broken)

    with pytest.raises(RuntimeError):
        router.route("reset password")
    assert len(store) == 0
    assert len(router.index) == 0


def test_healing_an_orphan_with_an_empty_answer_writes_nothing():
    # The orphan path writes under an id already in the index, so an empty
    # answer there would put "" where a lookup will later find it and treat
    # it as valid. The guard has to fire before that write, not after.
    #
    # Note the setup: the store is deliberately left empty so q_0 really is
    # an orphan. Seeding it with a response would make this an ordinary hit
    # that never reaches the LLM at all.
    embedder = FakeEmbedder({"reset password": [1.0, 0.0]})
    store = InMemoryCacheStore()
    router = CacheRouter(
        "linear", embedder=embedder, cache_store=store, llm=lambda q: ""
    )
    router.restore([("q_0", np.array([1.0, 0.0], dtype=np.float32))])

    with pytest.raises(EmptyLLMResponse):
        router.route("reset password")

    assert store.get("q_0") is None  # no empty entry written
    assert len(store) == 0


def test_a_hit_with_no_stored_response_never_serves_none():
    # The index and store can drift apart: the graph is snapshotted to
    # disk while responses live elsewhere, so a restored snapshot beside a
    # store missing those rows leaves ids in the index with nothing behind
    # them. Returning the lookup regardless answers the user with None at
    # similarity 1.0.
    calls = []
    embedder = FakeEmbedder({"reset password": [1.0, 0.0]})
    store = InMemoryCacheStore()
    router = CacheRouter(
        "linear", embedder=embedder, cache_store=store, llm=counting_llm(calls)
    )
    # An orphan: in the index, absent from the store.
    router.restore([("q_0", np.array([1.0, 0.0], dtype=np.float32))])

    result = router.route("reset password")

    assert result.response == "real answer to: reset password"
    assert result.response is not None
    assert result.hit is False  # honest: this cost an LLM call
    assert calls == ["reset password"]


def test_an_orphaned_entry_is_healed_in_place_not_duplicated():
    # Healing under the existing id matters: minting a new one would leave
    # the orphan in the index to fail identically on every future query.
    embedder = FakeEmbedder({"reset password": [1.0, 0.0]})
    store = InMemoryCacheStore()
    router = CacheRouter("linear", embedder=embedder, cache_store=store)
    router.restore([("q_0", np.array([1.0, 0.0], dtype=np.float32))])

    router.route("reset password")

    assert len(router.index) == 1  # no duplicate vector added
    assert store.get("q_0") == "[stub response for: reset password]"

    # The next identical query is now a clean hit with no LLM call.
    second = router.route("reset password")
    assert second.hit is True
    assert second.matched_id == "q_0"


def test_snapshot_roundtrips_the_index_and_serves_hits(tmp_path):
    embedder = FakeEmbedder({"reset password": [1.0, 0.0]})
    store = InMemoryCacheStore()
    router = CacheRouter("hnsw", embedder=embedder, cache_store=store)
    router.route("reset password")
    path = tmp_path / "graph.pkl"
    router.save_snapshot(path)

    # A fresh process: same store, empty index until the snapshot loads.
    revived = CacheRouter("hnsw", embedder=embedder, cache_store=store)
    assert len(revived.index) == 0
    assert revived.load_snapshot(path) is True

    result = revived.route("reset password")
    assert result.hit is True
    assert len(revived.index) == 1


def test_snapshot_carries_the_id_counter(tmp_path):
    # The section 22 failure, now via the snapshot path: restoring the
    # graph without its counter would reissue q_0 over a live entry.
    embedder = FakeEmbedder({
        "reset password": [1.0, 0.0],
        "change my email": [0.7, 0.7141428428542852],
    })
    store = InMemoryCacheStore()
    router = CacheRouter("linear", embedder=embedder, cache_store=store)
    router.route("reset password")  # mints q_0
    path = tmp_path / "graph.pkl"
    router.save_snapshot(path)

    revived = CacheRouter("linear", embedder=embedder, cache_store=store)
    revived.load_snapshot(path)
    revived.route("change my email")  # a miss, mints the next id

    assert store.get("q_0") == "[stub response for: reset password]"
    assert store.get("q_1") == "[stub response for: change my email]"


def test_snapshot_from_a_different_embedding_model_is_rejected(tmp_path):
    # Vectors only mean something in the space that produced them, so a
    # snapshot built by another model is not stale, it is nonsense. Loading
    # it would compare this model's queries against that model's vectors
    # and serve confident garbage.
    embedder = FakeEmbedder({"reset password": [1.0, 0.0]})
    router = CacheRouter("hnsw", embedder=embedder)
    router.route("reset password")
    path = tmp_path / "graph.pkl"
    router.save_snapshot(path)

    other = FakeEmbedder({"reset password": [1.0, 0.0]})
    other.model_name = "some-other-model"
    revived = CacheRouter("hnsw", embedder=other)

    assert revived.load_snapshot(path) is False
    assert len(revived.index) == 0  # left empty, caller falls back to the store


def test_load_snapshot_returns_false_when_there_is_no_file(tmp_path):
    embedder = FakeEmbedder({"reset password": [1.0, 0.0]})
    router = CacheRouter("hnsw", embedder=embedder)
    assert router.load_snapshot(tmp_path / "does-not-exist.pkl") is False


def test_load_snapshot_returns_false_on_a_corrupt_file(tmp_path):
    # A truncated or garbage snapshot must not crash startup: DynamoDB is
    # the source of truth and rebuilding from it is always available.
    path = tmp_path / "graph.pkl"
    path.write_bytes(b"not a pickle")
    embedder = FakeEmbedder({"reset password": [1.0, 0.0]})
    router = CacheRouter("hnsw", embedder=embedder)
    assert router.load_snapshot(path) is False


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
