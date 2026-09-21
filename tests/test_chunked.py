"""Chunked and streaming array drivers: agreement, gates, and empty inputs."""

import numpy as np
import pytest
from scipy import stats

from ltpred.estimate import (estimate_liability_gibbs_arrays,
                             estimate_liability_pa_arrays)
from ltpred.chunked import (estimate_liability_gibbs_batches,
                            estimate_liability_gibbs_chunked,
                            estimate_liability_pa_batches,
                            estimate_liability_pa_chunked)


def _nuclear_bounds(n=50, seed=1, p_case=0.3):
    t = float(stats.norm.isf(0.05))
    roles = ["o", "m", "f", "s1"]
    rng = np.random.default_rng(seed)
    case = rng.random((n, 4)) < p_case
    lower = np.where(case, t, -np.inf)
    upper = np.where(case, np.inf, t)
    return roles, lower, upper, t


def _mixture_bounds(n_control=20, n_case=10):
    t = float(stats.norm.isf(0.10))
    roles = ["o", "m"]
    n = n_control + n_case
    lower = np.empty((n, 2))
    upper = np.empty((n, 2))
    K_i = np.full((n, 2), np.nan)
    K_pop = np.full((n, 2), np.nan)
    lower[:n_control] = -np.inf
    upper[:n_control] = t
    K_i[:n_control] = 0.05
    K_pop[:n_control] = 0.10
    lower[n_control:] = t
    upper[n_control:] = np.inf
    return roles, lower, upper, K_i, K_pop


@pytest.mark.parametrize("chunk_size", [1, 7, 50, 10 ** 9])
def test_pa_chunked_matches_pa_arrays(chunk_size):
    roles, lower, upper, _t = _nuclear_bounds(50)
    est_a, var_a = estimate_liability_pa_arrays(roles, lower, upper, h2=0.5)
    est_c, var_c = estimate_liability_pa_chunked(
        roles, lower, upper, h2=0.5, chunk_size=chunk_size)
    np.testing.assert_array_equal(est_c, est_a)
    np.testing.assert_array_equal(var_c, var_a)


@pytest.mark.jit_required
@pytest.mark.parametrize("chunk_size", [1, 7, 10 ** 9])
def test_gibbs_chunked_matches_gibbs_arrays(chunk_size):
    roles, lower, upper, _t = _nuclear_bounds(12, seed=2)
    kwargs = dict(h2=0.5, n_sim=20_000, burn_in=400, seed=3, tol=0.05)
    est_a, se_a = estimate_liability_gibbs_arrays(roles, lower, upper, **kwargs)
    est_c, se_c = estimate_liability_gibbs_chunked(
        roles, lower, upper, chunk_size=chunk_size, **kwargs)
    np.testing.assert_array_equal(est_c, est_a)
    np.testing.assert_array_equal(se_c, se_a)
    est_a, se_a, var_a = estimate_liability_gibbs_arrays(
        roles, lower, upper, return_var=True, **kwargs)
    est_c, se_c, var_c = estimate_liability_gibbs_chunked(
        roles, lower, upper, chunk_size=chunk_size, return_var=True, **kwargs)
    np.testing.assert_array_equal(est_c, est_a)
    np.testing.assert_array_equal(se_c, se_a)
    np.testing.assert_array_equal(var_c, var_a)


def test_chunk_size_one_and_larger_than_F():
    roles, lower, upper, _t = _nuclear_bounds(8)
    est, var = estimate_liability_pa_arrays(roles, lower, upper, h2=0.5)
    for chunk_size in (1, 8, 9, 10 ** 9):
        got = estimate_liability_pa_chunked(
            roles, lower, upper, chunk_size=chunk_size, h2=0.5)
        np.testing.assert_array_equal(got[0], est)
        np.testing.assert_array_equal(got[1], var)


def test_empty_families_and_empty_batches():
    roles = ["o", "m", "f"]
    empty = np.empty((0, 3))
    est, var = estimate_liability_pa_chunked(roles, empty, empty, h2=0.5)
    assert est.shape == (0,) and var.shape == (0,)
    assert est.dtype == float and var.dtype == float
    est, var = estimate_liability_pa_batches(roles, [], h2=0.5)
    assert est.shape == (0,) and var.shape == (0,)
    est, se = estimate_liability_gibbs_chunked(
        roles, empty, empty, n_sim=20, burn_in=0, max_rounds=1, tol=1e9, h2=0.5)
    assert est.shape == (0,) and se.shape == (0,)
    est, se = estimate_liability_gibbs_batches(
        roles, [], n_sim=20, burn_in=0, max_rounds=1, tol=1e9, h2=0.5)
    assert est.shape == (0,) and se.shape == (0,)
    est, se, var = estimate_liability_gibbs_batches(
        roles, [], n_sim=20, burn_in=0, max_rounds=1, tol=1e9, return_var=True, h2=0.5)
    assert var.shape == (0,)


@pytest.mark.parametrize("chunk_size,error", [
    (0, ValueError), (-1, ValueError), (True, TypeError),
])
def test_invalid_chunk_size_rejected(chunk_size, error):
    roles, lower, upper, _t = _nuclear_bounds(4)
    with pytest.raises(error, match="chunk_size"):
        estimate_liability_pa_chunked(
            roles, lower, upper, chunk_size=chunk_size, h2=0.5)
    with pytest.raises(error, match="chunk_size"):
        estimate_liability_gibbs_chunked(
            roles, lower, upper, chunk_size=chunk_size,
            n_sim=20, burn_in=0, max_rounds=1, tol=1e9, h2=0.5)


def test_float32_chunked_matches_unchunked():
    roles, lower, upper, _t = _nuclear_bounds(40)
    lo32, hi32 = lower.astype(np.float32), upper.astype(np.float32)
    est_a, var_a = estimate_liability_pa_arrays(roles, lo32, hi32, h2=0.5)
    est_c, var_c = estimate_liability_pa_chunked(
        roles, lo32, hi32, chunk_size=7, h2=0.5)
    np.testing.assert_allclose(est_c, est_a, atol=1e-5)
    np.testing.assert_allclose(var_c, var_a, atol=1e-5)
    est64, var64 = estimate_liability_pa_arrays(roles, lower, upper, h2=0.5)
    np.testing.assert_allclose(est_c, est64, atol=1e-5)
    np.testing.assert_allclose(var_c, var64, atol=1e-5)


def test_pa_chunked_mixture_uses_global_gate():
    roles, lower, upper, K_i, K_pop = _mixture_bounds(n_control=6, n_case=8)
    est_a, var_a = estimate_liability_pa_arrays(
        roles, lower, upper, h2=0.5, use_mixture=True, K_i=K_i, K_pop=K_pop)
    est_c, var_c = estimate_liability_pa_chunked(
        roles, lower, upper, h2=0.5, use_mixture=True, K_i=K_i, K_pop=K_pop,
        chunk_size=1)
    np.testing.assert_array_equal(est_c, est_a)
    np.testing.assert_array_equal(var_c, var_a)
    with pytest.raises(ValueError, match="at least one active censored-control"):
        estimate_liability_pa_chunked(
            roles, lower[6:], upper[6:], h2=0.5, use_mixture=True,
            K_i=K_i[6:], K_pop=K_pop[6:], chunk_size=1)


def test_pa_batches_mixture_is_per_batch():
    roles, lower, upper, K_i, K_pop = _mixture_bounds(n_control=6, n_case=8)
    controls = (lower[:6], upper[:6], K_i[:6], K_pop[:6])
    cases = (lower[6:], upper[6:], K_i[6:], K_pop[6:])
    mixed = [
        {"lower": lower[:4], "upper": upper[:4],
         "K_i": K_i[:4], "K_pop": K_pop[:4]},
        (lower[4:6], upper[4:6], K_i[4:6], K_pop[4:6]),
    ]
    est_a, var_a = estimate_liability_pa_arrays(
        roles, lower[:6], upper[:6], h2=0.5, use_mixture=True,
        K_i=K_i[:6], K_pop=K_pop[:6])
    est_b, var_b = estimate_liability_pa_batches(
        roles, mixed, h2=0.5, use_mixture=True)
    np.testing.assert_array_equal(est_b, est_a)
    np.testing.assert_array_equal(var_b, var_a)
    with pytest.raises(ValueError, match="at least one active censored-control"):
        estimate_liability_pa_batches(
            roles, [controls, cases], h2=0.5, use_mixture=True)


def test_pa_batches_concat_matches_arrays():
    roles, lower, upper, _t = _nuclear_bounds(21)
    est_a, var_a = estimate_liability_pa_arrays(roles, lower, upper, h2=0.5)
    batches = [
        (lower[:5], upper[:5]),
        {"lower": lower[5:13], "upper": upper[5:13]},
        (lower[13:], upper[13:]),
    ]
    est_b, var_b = estimate_liability_pa_batches(roles, batches, h2=0.5)
    np.testing.assert_array_equal(est_b, est_a)
    np.testing.assert_array_equal(var_b, var_a)

    def gen():
        yield lower[:8], upper[:8]
        yield lower[8:], upper[8:]

    est_g, var_g = estimate_liability_pa_batches(roles, gen(), h2=0.5)
    np.testing.assert_array_equal(est_g, est_a)
    np.testing.assert_array_equal(var_g, var_a)


@pytest.mark.jit_required
def test_gibbs_batches_concat_matches_arrays_same_seed():
    roles, lower, upper, _t = _nuclear_bounds(9, seed=4)
    kwargs = dict(h2=0.5, n_sim=20_000, burn_in=400, seed=11, tol=0.05)
    est_a, se_a, var_a = estimate_liability_gibbs_arrays(
        roles, lower, upper, return_var=True, **kwargs)
    batches = [
        (lower[:3], upper[:3]),
        {"lower": lower[3:7], "upper": upper[3:7]},
        (lower[7:], upper[7:]),
    ]
    est_b, se_b, var_b = estimate_liability_gibbs_batches(
        roles, batches, return_var=True, **kwargs)
    np.testing.assert_array_equal(est_b, est_a)
    np.testing.assert_array_equal(se_b, se_a)
    np.testing.assert_array_equal(var_b, var_a)


@pytest.mark.parametrize("name", ["pa", "gibbs"])
def test_chunked_and_batches_reject_duplicate_and_g_roles(name):
    chunked = (estimate_liability_pa_chunked if name == "pa"
               else estimate_liability_gibbs_chunked)
    batches = (estimate_liability_pa_batches if name == "pa"
               else estimate_liability_gibbs_batches)
    extra = (dict(h2=0.5) if name == "pa"
             else dict(h2=0.5, n_sim=20, burn_in=0, max_rounds=1, tol=1e9))
    lower = np.full((2, 3), -np.inf)
    upper = np.full((2, 3), np.inf)
    with pytest.raises(ValueError, match="duplicate role"):
        chunked(["o", "m", "m"], lower, upper, **extra)
    with pytest.raises(ValueError, match="duplicate role"):
        batches(["o", "m", "m"], [(lower, upper)], **extra)
    with pytest.raises(ValueError, match="must not contain 'g'"):
        chunked(["g", "o", "m"], lower, upper, **extra)
    with pytest.raises(ValueError, match="must not contain 'g'"):
        batches(["g", "o", "m"], [(lower, upper)], **extra)


@pytest.mark.parametrize("name", ["pa", "gibbs"])
def test_chunked_rejects_wrong_column_count(name):
    chunked = (estimate_liability_pa_chunked if name == "pa"
               else estimate_liability_gibbs_chunked)
    extra = (dict(h2=0.5) if name == "pa"
             else dict(h2=0.5, n_sim=20, burn_in=0, max_rounds=1, tol=1e9))
    roles = ["o", "m"]
    lower = np.full((3, 4), -np.inf)
    upper = np.full((3, 4), np.inf)
    with pytest.raises(ValueError, match=r"\(n_families, 2\)"):
        chunked(roles, lower, upper, **extra)
    lower = np.full((3, 2), -np.inf)
    upper = np.full((3, 1), np.inf)
    with pytest.raises(ValueError):
        chunked(roles, lower, upper, **extra)
