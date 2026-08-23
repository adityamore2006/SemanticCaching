"""
Contract tests: behavior every VectorIndex implementation must satisfy.

These are index-agnostic. They run through the factory so any index added
to the registry gets held to the same guarantees the cache router and eval
harness rely on. Both LinearIndex and HNSWIndex are exercised here;
exact-vs-approximate recall differences (Phase 4) are tested separately,
but these invariants hold for both.
"""

import math

import pytest

from factory import create_index, available_indexes

IMPLEMENTED = ["linear", "hnsw"]


def test_all_registered_indexes_are_implemented():
    # Guardrail: the registry shouldn't grow a name this suite silently
    # ignores.
    assert set(available_indexes()) == set(IMPLEMENTED)


@pytest.fixture(params=IMPLEMENTED)
def index_factory(request):
    """Returns a callable that builds a fresh index of the parametrized kind."""
    kind = request.param
    return lambda dim: create_index(kind, dim=dim)


def test_self_similarity_is_one(index_factory):
    index = index_factory(3)
    index.insert("a", [1.0, 2.0, 3.0])
    results = index.search([1.0, 2.0, 3.0], k=1)
    assert results[0][0] == "a"
    assert math.isclose(results[0][1], 1.0, abs_tol=1e-6)


def test_results_sorted_descending(index_factory):
    index = index_factory(2)
    index.insert("east", [1.0, 0.0])
    index.insert("north", [0.0, 1.0])
    index.insert("almost_east", [0.98, 0.2])

    results = index.search([1.0, 0.05], k=3)
    sims = [s for _, s in results]
    assert sims == sorted(sims, reverse=True)
    assert results[0][0] == "east"


def test_empty_index_returns_empty(index_factory):
    assert index_factory(2).search([1.0, 0.0], k=3) == []


def test_k_larger_than_size_returns_all(index_factory):
    index = index_factory(2)
    index.insert("only", [1.0, 0.0])
    assert len(index.search([1.0, 0.0], k=5)) == 1


def test_wrong_dimension_raises(index_factory):
    index = index_factory(3)
    with pytest.raises(ValueError):
        index.insert("bad", [1.0, 0.0])


def test_len_tracks_inserts(index_factory):
    index = index_factory(2)
    assert len(index) == 0
    index.insert("a", [1.0, 0.0])
    index.insert("b", [0.0, 1.0])
    assert len(index) == 2


def test_factory_rejects_unknown_kind():
    with pytest.raises(ValueError):
        create_index("does_not_exist", dim=2)
