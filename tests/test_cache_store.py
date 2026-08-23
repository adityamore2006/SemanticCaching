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
