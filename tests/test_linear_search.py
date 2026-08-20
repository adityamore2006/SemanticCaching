"""
Correctness tests for LinearIndex.

These are deliberately hand-checkable: every expected value here can be
verified with a calculator, not just "the code agrees with itself." That
matters because this index later becomes the ground truth for grading
HNSW's approximate results, if this is wrong, everything built on top of
it is wrong too.
"""

import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from linear_search import LinearIndex


def test_self_similarity_is_one():
    index = LinearIndex(dim=3)
    index.insert("a", [1.0, 2.0, 3.0])
    results = index.search([1.0, 2.0, 3.0], k=1)
    assert results[0][0] == "a"
    assert math.isclose(results[0][1], 1.0, abs_tol=1e-6)


def test_orthogonal_vectors_have_zero_similarity():
    index = LinearIndex(dim=2)
    index.insert("x_axis", [1.0, 0.0])
    results = index.search([0.0, 1.0], k=1)
    assert math.isclose(results[0][1], 0.0, abs_tol=1e-6)


def test_opposite_vectors_have_similarity_negative_one():
    index = LinearIndex(dim=2)
    index.insert("east", [1.0, 0.0])
    results = index.search([-1.0, 0.0], k=1)
    assert math.isclose(results[0][1], -1.0, abs_tol=1e-6)


def test_results_sorted_descending_by_similarity():
    index = LinearIndex(dim=2)
    index.insert("east", [1.0, 0.0])
    index.insert("north", [0.0, 1.0])
    index.insert("almost_east", [0.98, 0.2])

    results = index.search([1.0, 0.05], k=3)
    similarities = [s for _, s in results]

    assert similarities == sorted(similarities, reverse=True)
    assert results[0][0] == "east"
    assert results[-1][0] == "north"


def test_top_k_matches_hand_computed_ranking():
    # cos(query, a) = (3*1 + 4*0) / (5 * 1) = 0.6
    # cos(query, b) = (3*0 + 4*1) / (5 * 1) = 0.8
    index = LinearIndex(dim=2)
    index.insert("a", [1.0, 0.0])
    index.insert("b", [0.0, 1.0])

    results = index.search([3.0, 4.0], k=2)
    result_dict = dict(results)

    assert math.isclose(result_dict["a"], 0.6, abs_tol=1e-6)
    assert math.isclose(result_dict["b"], 0.8, abs_tol=1e-6)
    assert results[0][0] == "b"  # higher similarity ranked first


def test_k_larger_than_index_size_returns_all():
    index = LinearIndex(dim=2)
    index.insert("only_one", [1.0, 0.0])
    results = index.search([1.0, 0.0], k=5)
    assert len(results) == 1


def test_empty_index_returns_empty_list():
    index = LinearIndex(dim=2)
    results = index.search([1.0, 0.0], k=3)
    assert results == []


def test_wrong_dimension_raises():
    index = LinearIndex(dim=3)
    try:
        index.insert("bad", [1.0, 0.0])
        assert False, "expected ValueError for wrong dimension"
    except ValueError:
        pass


def test_len_reflects_number_of_inserts():
    index = LinearIndex(dim=2)
    assert len(index) == 0
    index.insert("a", [1.0, 0.0])
    index.insert("b", [0.0, 1.0])
    assert len(index) == 2
