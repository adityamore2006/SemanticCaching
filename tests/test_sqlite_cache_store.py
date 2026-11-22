import threading

import numpy as np
import pytest

from cache_router import CacheRouter
from fake_llm import FakeLLM
from sqlite_cache_store import EmbedderMismatch, SqliteCacheStore


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "cache.db")


def test_put_then_get_roundtrips(db):
    store = SqliteCacheStore(db, "m")
    store.put("a", "response a", np.array([1.0, 0.0], dtype=np.float32), source="llm")
    assert store.get("a") == "response a"
    assert store.get("nope") is None
    assert len(store) == 1


def test_put_overwrites(db):
    store = SqliteCacheStore(db, "m")
    v = np.array([1.0, 0.0], dtype=np.float32)
    store.put("a", "first", v)
    store.put("a", "second", v)
    assert store.get("a") == "second"
    assert len(store) == 1


def test_entries_and_exact_vectors_survive_a_reopen(db):
    vector = np.random.default_rng(0).random(768).astype(np.float32)
    first = SqliteCacheStore(db, "m")
    first.put("q_0", "answer", vector)
    first.close()

    second = SqliteCacheStore(db, "m")
    assert second.get("q_0") == "answer"
    [(id, loaded, response)] = list(second.all_items())
    assert (id, response) == ("q_0", "answer")
    assert np.array_equal(loaded, vector)  # bit-for-bit, not approximately


def test_put_refuses_an_entry_without_a_vector(db):
    # A vectorless entry would be served by get() but never restored into
    # the index, so it must not be written at all.
    store = SqliteCacheStore(db, "m")
    with pytest.raises(ValueError, match="no_vector"):
        store.put("no_vector", "r", None)
    assert len(store) == 0
    assert store.get("no_vector") is None


def test_all_items_returns_every_entry_in_insert_order(db):
    store = SqliteCacheStore(db, "m")
    store.put("q_0", "r0", np.array([1.0, 0.0], dtype=np.float32))
    store.put("q_1", "r1", np.array([0.0, 1.0], dtype=np.float32))
    assert [i for i, _, _ in store.all_items()] == ["q_0", "q_1"]


def test_clear_removes_everything_in_memory_and_on_disk(db):
    store = SqliteCacheStore(db, "m")
    for i in range(5):
        store.put(f"q_{i}", "r", np.array([1.0, 0.0], dtype=np.float32))
    assert store.clear() == 5
    assert len(store) == 0 and store.get("q_0") is None
    store.close()
    assert len(SqliteCacheStore(db, "m")) == 0


def test_opening_with_a_different_embedder_is_refused(db):
    SqliteCacheStore(db, "model-a").close()
    with pytest.raises(EmbedderMismatch):
        SqliteCacheStore(db, "model-b")


def test_concurrent_puts_from_many_threads_are_all_stored(db):
    store = SqliteCacheStore(db, "m")

    def worker(n):
        for i in range(50):
            store.put(f"t{n}_{i}", "r", np.array([1.0, 0.0], dtype=np.float32))

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(store) == 400
    assert len(list(store.all_items())) == 400


class FakeEmbedder:
    dim = 2
    model_name = "m"
    vectors = {"reset password": [1.0, 0.0], "cancel plan": [0.0, 1.0]}

    def embed(self, text):
        return np.array(self.vectors[text], dtype=np.float32)


def test_a_restart_restores_the_cache_and_serves_a_hit_without_the_llm(db):
    # The point of the store: ask, "restart", ask again.
    llm = FakeLLM()
    first = CacheRouter("hnsw", FakeEmbedder(), llm, SqliteCacheStore(db, "m"))
    assert first.route("reset password").hit is False
    assert llm.calls == 1
    first.cache_store.close()

    llm2 = FakeLLM()
    store = SqliteCacheStore(db, "m")
    second = CacheRouter("hnsw", FakeEmbedder(), llm2, store)
    second.restore((id, v) for id, v, _ in store.all_items())

    result = second.route("reset password")
    assert result.hit is True
    assert result.response == "fake answer to: reset password"
    assert llm2.calls == 0

    # A new miss must not reuse the restored entry's id.
    second.route("cancel plan")
    assert store.get("q_0") == "fake answer to: reset password"
    assert store.get("q_1") == "fake answer to: cancel plan"
