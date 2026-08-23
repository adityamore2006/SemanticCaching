import numpy as np
import pytest

from cache_router import CacheRouter


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
