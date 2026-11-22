import numpy as np

from cache_store import InMemoryCacheStore

def test_put_then_get_roundtrips():
    store = InMemoryCacheStore()
    store.put("a", "response for a")
    assert store.get("a") == "response for a"


def test_get_missing_id_returns_none():
    store = InMemoryCacheStore()
    assert store.get("nope") is None


def test_put_overwrites_existing_entry():
    store = InMemoryCacheStore()
    store.put("a", "first")
    store.put("a", "second")
    assert store.get("a") == "second"


def test_len_tracks_distinct_ids():
    store = InMemoryCacheStore()
    assert len(store) == 0
    store.put("a", "r1")
    store.put("b", "r2")
    assert len(store) == 2
    store.put("a", "overwritten")
    assert len(store) == 2


def test_in_memory_store_accepts_and_ignores_a_vector():
    # The vector argument exists for persistent stores; the in-memory one
    # must still accept it so both satisfy one contract.
    store = InMemoryCacheStore()
    store.put("a", "response for a", np.array([0.1, 0.2], dtype=np.float32))
    assert store.get("a") == "response for a"


# --- clear(): the reset primitive ---


def test_in_memory_clear_empties_and_reports_the_count():
    store = InMemoryCacheStore()
    store.put("a", "r1")
    store.put("b", "r2")

    assert store.clear() == 2
    assert len(store) == 0
    assert store.get("a") is None
