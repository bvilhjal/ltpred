"""Selected relationships must preserve the full pedigree, including inbreeding."""
import tracemalloc

import numpy as np
import pytest

from ltpred._selected_kinship import _SelectedKinship, _covariance_reduction_is_safe
from ltpred.covariance import kinship_from_pedigree, construct_covmat_from_kinship
from ltpred.pedigree import build_parent_graph


IDS = ["gm", "gf", "a", "b", "mate", "inbred", "half"]
FATHER = [None, None, "gf", "gf", None, "a", "a"]
MOTHER = [None, None, "gm", "gm", None, "b", "mate"]


@pytest.mark.parametrize("cache_size", [0, 1, 3, 100_000])
def test_selected_inbred_halfsiblings_and_eviction_match_full_matrix(cache_size):
    graph = build_parent_graph(IDS, FATHER, MOTHER)
    selected = _SelectedKinship(graph, cache_size)
    _, full = kinship_from_pedigree(IDS, FATHER, MOTHER)
    requested = np.array([5, 6, 2, 3])
    expected = full[np.ix_(requested, requested)]
    np.testing.assert_array_equal(selected.matrix(requested), expected)
    assert expected[0, 0] == 1.25
    assert expected[0, 1] == .375  # shared father and related mothers
    selected.matrix(np.arange(len(IDS)))
    assert len(selected._cache) <= cache_size
    np.testing.assert_array_equal(selected.matrix(requested), expected)


def test_selected_random_pedigree_and_record_permutation():
    rng = np.random.default_rng(114)
    n = 60
    father = [None] * 6
    mother = [None] * 6
    for i in range(6, n):
        parents = rng.choice(i, size=2, replace=False).tolist()
        father.append(parents[0])
        mother.append(parents[1])
    ids = list(range(n))
    _, full = kinship_from_pedigree(ids, father, mother)
    requested = np.array([59, 31, 47, 22, 5])
    expected = full[np.ix_(requested, requested)]
    for order in [np.arange(n), rng.permutation(n)]:
        graph = build_parent_graph([ids[i] for i in order],
                                   [father[i] for i in order],
                                   [mother[i] for i in order])
        selected = _SelectedKinship(graph, 1_000)
        indices = np.array([graph.index[i] for i in requested])
        np.testing.assert_array_equal(selected.matrix(indices), expected)


def test_deep_pedigree_uses_bounded_cache_and_no_dense_population_matrix():
    n = 10_000
    graph = build_parent_graph(list(range(n)), [None] + list(range(n - 1)),
                               [None] * n)
    # Include a founder query, so the evaluator must traverse the entire
    # chain rather than merely compare the last two relatives.
    tracemalloc.start()
    try:
        selected = _SelectedKinship(graph, 32)
        result = selected.matrix(np.array([n - 1, n - 2, 0]))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    np.testing.assert_array_equal(result, [[1., .5, 0.], [.5, 1., 0.], [0., 0., 1.]])
    assert len(selected._cache) <= 32
    # A full 10,000-square float64 matrix alone needs 800 MB. The generous
    # ceiling covers rank, topological worklists and the explicit depth stack.
    assert peak < 8_000_000


def test_selected_rejects_cycles_anywhere_and_invalid_cache_sizes():
    graph = build_parent_graph(["o", "a", "b"], [None, "b", "a"],
                               [None] * 3)
    with pytest.raises(ValueError, match="cycle"):
        _SelectedKinship(graph)
    good = build_parent_graph(["o"], [None], [None])
    for value in [True, np.bool_(False), 1.5]:
        with pytest.raises(TypeError, match="integer"):
            _SelectedKinship(good, value)
    with pytest.raises(ValueError, match="nonnegative"):
        _SelectedKinship(good, -1)


@pytest.mark.parametrize("h2", [.000001, .2, .5, .95, .999999])
def test_reduction_lower_bound_is_conservative_for_inbred_covariances(h2):
    _, relationship = kinship_from_pedigree(IDS, FATHER, MOTHER)
    residual = (1.0 - h2) / (1.0 + h2)
    bound = 1.0 / (1.0 / h2 + 3.0 * len(IDS) / residual)
    for target in range(len(IDS)):
        cov = construct_covmat_from_kinship(relationship, h2=h2, target=target).matrix
        assert np.linalg.eigvalsh(cov)[0] >= bound
    assert _covariance_reduction_is_safe(h2, len(IDS)) == (bound > 1e-6)


def test_singular_boundary_keeps_legacy_covariance_path():
    assert not _covariance_reduction_is_safe(1., len(IDS))
    assert not _covariance_reduction_is_safe(1. - 1e-10, len(IDS))
    assert not _covariance_reduction_is_safe(1e-10, len(IDS))
    assert _covariance_reduction_is_safe(.5, len(IDS))
