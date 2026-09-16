"""The truncated-MVN Gibbs sampler against closed-form truncated-normal results."""

from concurrent.futures import ThreadPoolExecutor
import importlib
import threading

import numpy as np
import pytest
from scipy import stats

from ltpred._numba import HAVE_NUMBA
from ltpred.gibbs import (rtmvnorm_gibbs, gibbs_params, gibbs_advance,
                          _seed_rng)

from _helpers import imr

gibbs_mod = importlib.import_module("ltpred.gibbs")


def _advance_from_zero(seed=None, n_families=12, d=3, n_sweeps=7):
    if seed is not None:
        _seed_rng(seed)
    x = np.zeros((n_families, d))
    gibbs_advance(np.zeros((d, d)), np.ones(d),
                  np.full_like(x, -np.inf), np.full_like(x, np.inf),
                  np.zeros_like(x, dtype=bool), x, n_sweeps)
    return x


def test_gibbs_params_precision_matches_regression():
    # precision-matrix P/sd must equal the conditional-regression form
    rng = np.random.default_rng(0)
    for d in (2, 4, 6):
        A = rng.normal(size=(d, d))
        cov = A @ A.T + d * np.eye(d)          # random PD
        P, sd = gibbs_params(cov)
        idx = np.arange(d)
        for j in range(d):
            rest = idx[idx != j]
            pj = np.linalg.solve(cov[np.ix_(rest, rest)], cov[rest, j])
            assert np.allclose(P[rest, j], pj)
            assert P[j, j] == 0.0
            assert sd[j] == pytest.approx(np.sqrt(cov[j, j] - pj @ cov[rest, j]))


def test_univariate_truncated_mean_and_var():
    # N(0,1) truncated to (0.5, 2.0): compare to scipy.truncnorm
    lo, hi = 0.5, 2.0
    s = rtmvnorm_gibbs(np.array([[1.0]]), lower=[lo], upper=[hi], out=(0,),
                       n_sim=200_000, burn_in=1000, seed=3)[:, 0]
    ref = stats.truncnorm(lo, hi)
    assert s.mean() == pytest.approx(ref.mean(), abs=0.01)
    assert s.std() == pytest.approx(ref.std(), abs=0.01)


@pytest.mark.parametrize("lower,upper", [
    (8.0, np.inf),
    (9.0, 10.0),
    (-np.inf, -8.0),
    (-10.0, -9.0),
])
def test_univariate_extreme_tail_draws_are_stable_and_in_bounds(lower, upper):
    # Direct CDF interpolation collapses Phi(9) and Phi(10) to the same float.
    # The sampler must use the survival scale in the right tail (and symmetry in
    # the left tail), not clip the resulting draw back near 7.94 sigma.
    draws = rtmvnorm_gibbs(np.eye(1), lower=[lower], upper=[upper], out=(0,),
                            n_sim=30_000, burn_in=100, seed=19)[:, 0]
    ref = stats.truncnorm(lower, upper)

    assert np.isfinite(draws).all()
    assert np.all(draws >= lower)
    assert np.all(draws <= upper)
    assert draws.mean() == pytest.approx(ref.mean(), abs=0.005)
    assert draws.std() == pytest.approx(ref.std(), abs=0.005)


def test_parallel_advance_keeps_extreme_tail_draws_in_bounds():
    n_families = 16
    x = np.full((n_families, 2), 9.1)
    lower = np.tile([9.0, 8.0], (n_families, 1))
    upper = np.tile([np.inf, 10.0], (n_families, 1))
    fixed = np.zeros_like(x, dtype=bool)

    _seed_rng(23)
    gibbs_advance(np.zeros((2, 2)), np.ones(2), lower, upper, fixed, x, 20)

    assert np.isfinite(x).all()
    assert np.all(x >= lower)
    assert np.all(x <= upper)


def test_bivariate_case_posterior_genetic_liability():
    # (g, o) with cov [[h2,h2],[h2,1]]; proband is a case: o in (T, inf).
    # E[o | o>T] = IMR(T); E[g | .] = h2 * E[o | .]  since E[g|o] = h2 * o.
    h2, prev = 0.5, 0.05
    t = stats.norm.isf(prev)
    cov = np.array([[h2, h2], [h2, 1.0]])
    s = rtmvnorm_gibbs(cov, lower=[-np.inf, t], upper=[np.inf, np.inf],
                       out=(0, 1), n_sim=200_000, burn_in=1000, seed=1)
    gen, full = s.mean(axis=0)
    assert full == pytest.approx(imr(t), abs=0.02)
    assert gen == pytest.approx(h2 * imr(t), abs=0.02)


def test_bivariate_control_posterior_is_negative():
    # control: o in (-inf, T). E[o | o<T] = -phi(T)/Phi(T) < 0.
    h2, prev = 0.5, 0.20
    t = stats.norm.isf(prev)
    cov = np.array([[h2, h2], [h2, 1.0]])
    s = rtmvnorm_gibbs(cov, lower=[-np.inf, -np.inf], upper=[np.inf, t],
                       out=(0, 1), n_sim=200_000, burn_in=1000, seed=2)
    gen, full = s.mean(axis=0)
    ref_full = -stats.norm.pdf(t) / stats.norm.cdf(t)
    assert full == pytest.approx(ref_full, abs=0.02)
    assert gen == pytest.approx(h2 * ref_full, abs=0.02)
    assert gen < 0


def test_fixed_coordinate_is_held_constant():
    # o pinned at a point (lower==upper) -> that coordinate never moves.
    h2 = 0.5
    pin = 1.3
    cov = np.array([[h2, h2], [h2, 1.0]])
    s = rtmvnorm_gibbs(cov, lower=[-np.inf, pin], upper=[np.inf, pin],
                       out=(0, 1), n_sim=50_000, burn_in=500, seed=5)
    assert np.allclose(s[:, 1], pin)
    # with o fixed at pin, E[g] = h2 * pin exactly
    assert s[:, 0].mean() == pytest.approx(h2 * pin, abs=0.01)


def test_out_selection_and_ordering():
    cov = np.array([[0.5, 0.5], [0.5, 1.0]])
    one = rtmvnorm_gibbs(cov, lower=[-np.inf, 1.0], upper=[np.inf, np.inf],
                         out=(1,), n_sim=10_000, burn_in=200, seed=7)
    assert one.shape == (10_000, 1)
    both = rtmvnorm_gibbs(cov, lower=[-np.inf, 1.0], upper=[np.inf, np.inf],
                          out=(1, 0), n_sim=10_000, burn_in=200, seed=7)
    assert both.shape == (10_000, 2)  # out is sorted -> columns are (g, o)


def test_reversed_bounds_raise_instead_of_changing_the_model():
    with pytest.raises(ValueError, match=r"reversed bounds at indices \[1\]"):
        rtmvnorm_gibbs(np.eye(2), lower=[-np.inf, 1.0], upper=[np.inf, 0.0],
                       n_sim=1, burn_in=0)


@pytest.mark.parametrize("lower,upper", [([np.nan], [np.inf]),
                                          ([-np.inf], [np.nan])])
def test_nan_bounds_raise_instead_of_reaching_the_sampler(lower, upper):
    with pytest.raises(ValueError, match="must not contain NaN"):
        rtmvnorm_gibbs(np.eye(1), lower=lower, upper=upper,
                       n_sim=1, burn_in=0)


@pytest.mark.parametrize("out,exc,match", [
    ((), ValueError, "at least one"),
    ((-1,), ValueError, "valid range"),
    ((2,), ValueError, "valid range"),
    ((1.0,), TypeError, "must be an integer"),
    (("1",), TypeError, "must be an integer"),
    ((True,), TypeError, "not bool"),
])
def test_out_rejects_empty_noninteger_and_out_of_range_indices(out, exc, match):
    with pytest.raises(exc, match=match):
        rtmvnorm_gibbs(np.eye(2), out=out, n_sim=1, burn_in=0)


def test_params_reuse_matches_fresh():
    cov = np.array([[0.5, 0.5, 0.25], [0.5, 1.0, 0.25], [0.25, 0.25, 1.0]])
    lo = [-np.inf, 1.0, -np.inf]
    hi = [np.inf, np.inf, 1.2]
    pre = gibbs_params(cov)
    a = rtmvnorm_gibbs(cov, lower=lo, upper=hi, out=(0,), n_sim=20_000,
                       burn_in=300, seed=11, params=pre)
    b = rtmvnorm_gibbs(cov, lower=lo, upper=hi, out=(0,), n_sim=20_000,
                       burn_in=300, seed=11)
    assert np.array_equal(a, b)


@pytest.mark.skipif(not HAVE_NUMBA, reason="requires Numba worker threads")
def test_parallel_gibbs_advance_seed_is_scheduler_independent():
    from numba import config, get_num_threads, set_num_threads

    if config.NUMBA_NUM_THREADS < 2:
        pytest.skip("requires at least two Numba worker threads")

    n_families, d = 32, 3
    P = np.zeros((d, d))
    sd = np.ones(d)
    lower = np.full((n_families, d), -np.inf)
    upper = np.full((n_families, d), np.inf)
    fixed = np.zeros((n_families, d), dtype=bool)

    def run(seed, n_threads):
        set_num_threads(n_threads)
        x = np.zeros((n_families, d))
        _seed_rng(seed)
        gibbs_advance(P, sd, lower, upper, fixed, x, 7)
        return x

    previous = get_num_threads()
    workers = min(4, config.NUMBA_NUM_THREADS)
    try:
        # Compile before the comparison; compilation itself must not be part of
        # the experiment.
        run(0, workers)
        serial = run(123, 1)
        parallel_a = run(123, workers)
        parallel_b = run(123, workers)
    finally:
        set_num_threads(previous)

    assert np.array_equal(serial, parallel_a)
    assert np.array_equal(parallel_a, parallel_b)
    # A tempting but broken fix is np.random.seed(seed) inside every iteration.
    # Identical families must not therefore receive identical draws.
    assert np.unique(parallel_a, axis=0).shape[0] == n_families


def test_concurrent_gibbs_advance_seed_streams_are_isolated():
    barrier = threading.Barrier(2)

    def run(seed, synchronize=False):
        if HAVE_NUMBA:
            from numba import set_num_threads
            set_num_threads(1)
        _seed_rng(seed)
        x = np.zeros((12, 3))
        P = np.zeros((3, 3))
        sd = np.ones(3)
        lower = np.full_like(x, -np.inf)
        upper = np.full_like(x, np.inf)
        fixed = np.zeros_like(x, dtype=bool)
        for _ in range(4):
            if synchronize:
                barrier.wait(timeout=10)
            gibbs_advance(P, sd, lower, upper, fixed, x, 2)
        return x

    expected = {seed: run(seed) for seed in (101, 202)}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {seed: pool.submit(run, seed, True) for seed in expected}
        actual = {seed: future.result(timeout=30)
                  for seed, future in futures.items()}

    for seed in expected:
        assert np.array_equal(actual[seed], expected[seed])


@pytest.mark.parametrize("seed", [True, np.bool_(False), 1.0, np.float64(2), "3"])
def test_seed_rejects_non_integer_types(seed):
    with pytest.raises(TypeError, match="seed must be an integer"):
        _seed_rng(seed)


@pytest.mark.parametrize("seed", [-1, np.int64(-1), 1 << 32])
def test_seed_rejects_values_outside_uint32(seed):
    with pytest.raises(ValueError, match="seed must be in"):
        _seed_rng(seed)


def test_invalid_seed_does_not_mutate_parallel_stream():
    _seed_rng(29)
    with pytest.raises(ValueError):
        _seed_rng(-1)
    actual = _advance_from_zero()
    expected = _advance_from_zero(seed=29)
    assert np.array_equal(actual, expected)


def test_public_sampler_rejects_float_seed():
    with pytest.raises(TypeError, match="seed must be an integer"):
        rtmvnorm_gibbs(np.eye(1), n_sim=1, burn_in=0, seed=1.0)


def test_gibbs_advance_chunks_uniform_storage(monkeypatch):
    class RecordingGenerator:
        def __init__(self):
            self.shapes = []

        def random(self, shape):
            self.shapes.append(shape)
            return np.full(shape, 0.5)

    generator = RecordingGenerator()
    monkeypatch.setattr(gibbs_mod._advance_rng_state, "generator", generator,
                        raising=False)
    monkeypatch.setattr(gibbs_mod, "_MAX_ADVANCE_UNIFORMS", 20)
    _advance_from_zero(n_families=9, d=3, n_sweeps=5)

    assert len(generator.shapes) > 1
    assert max(np.prod(shape) for shape in generator.shapes) <= 20


def test_gibbs_advance_python_fallback_is_random_free(monkeypatch):
    expected = _advance_from_zero(seed=77)
    _seed_rng(77)

    kernel = gibbs_mod._gibbs_advance
    monkeypatch.setattr(gibbs_mod, "_gibbs_advance",
                        getattr(kernel, "py_func", kernel))

    def unexpected_reseed(*_args, **_kwargs):
        raise AssertionError("advance kernel must consume supplied uniforms")

    monkeypatch.setattr(np.random, "seed", unexpected_reseed)
    actual = _advance_from_zero()
    assert np.array_equal(actual, expected)


def test_gibbs_params_rejects_non_symmetric_and_indefinite_covmat():
    # a merely invertible but indefinite covmat used to silently give NaN sd
    nonsym = np.array([[1.0, 0.9], [0.1, 1.0]])
    with pytest.raises(ValueError, match="symmetric"):
        gibbs_params(nonsym)
    tiny_nonsym = np.array([[1e-13, 5e-14], [0.0, 1e-13]])
    with pytest.raises(ValueError, match="symmetric"):
        gibbs_params(tiny_nonsym)
    indefinite = np.array([[1.0, 2.0], [2.0, 1.0]])  # invertible, eigvals 3, -1
    assert np.linalg.det(indefinite) != 0.0
    with pytest.raises(ValueError, match="positive-definite"):
        gibbs_params(indefinite)
    with pytest.raises(ValueError, match="finite"):
        gibbs_params(np.array([[np.nan, 0.0], [0.0, 1.0]]))
    # rtmvnorm_gibbs goes through gibbs_params, so it rejects them too
    with pytest.raises(ValueError, match="symmetric"):
        rtmvnorm_gibbs(nonsym, n_sim=10, burn_in=0)
    with pytest.raises(ValueError, match="positive-definite"):
        rtmvnorm_gibbs(indefinite, n_sim=10, burn_in=0)
    singular = np.ones((2, 2))
    with pytest.raises(ValueError, match="positive-definite"):
        gibbs_params(singular)
    with pytest.raises(ValueError, match="positive-definite"):
        rtmvnorm_gibbs(singular, n_sim=10, burn_in=0)
    # Precomputed parameters are an optimisation, not a covariance-validation
    # bypass. The supplied covariance still defines the marginal initialisation.
    params = gibbs_params(np.eye(2))
    with pytest.raises(ValueError, match="positive-definite"):
        rtmvnorm_gibbs(singular, params=params, n_sim=10, burn_in=0)
    with pytest.raises(ValueError, match="positive-definite"):
        rtmvnorm_gibbs(indefinite, params=params, n_sim=10, burn_in=0)
    # Legitimate PD matrices still work, including a small but well-conditioned
    # general covariance (the numerical cutoff is scale-relative).
    gibbs_params(np.array([[1.0, 0.5], [0.5, 1.0]]))
    gibbs_params(1e-13 * np.eye(2))


@pytest.mark.parametrize("bound", [38.0, 40.0, 50.0, 100.0])
def test_far_tail_bounds_do_not_produce_nan(bound):
    # Beyond |z| ~ 38.5 the survival scale underflows to 0, the inverse returned
    # +-inf, and NaN then propagated through every later conditional mean.
    from ltpred.pearson_aitken import _std_tnorm_moments
    right = rtmvnorm_gibbs([[1.0]], lower=bound, out=(0,), n_sim=40_000,
                           burn_in=200, seed=1).mean()
    assert np.isfinite(right)
    np.testing.assert_allclose(right, _std_tnorm_moments(bound, np.inf)[0],
                               rtol=1e-3)
    left = rtmvnorm_gibbs([[1.0]], upper=-bound, out=(0,), n_sim=40_000,
                          burn_in=200, seed=1).mean()
    assert np.isfinite(left)
    np.testing.assert_allclose(left, _std_tnorm_moments(-np.inf, -bound)[0],
                               rtol=1e-3)


def test_far_tail_honours_a_finite_upper_bound():
    from ltpred.pearson_aitken import _std_tnorm_moments
    draws = rtmvnorm_gibbs([[1.0]], lower=40.0, upper=40.02, out=(0,),
                           n_sim=40_000, burn_in=200, seed=1)
    assert np.all((draws >= 40.0) & (draws <= 40.02))
    np.testing.assert_allclose(draws.mean(),
                               _std_tnorm_moments(40.0, 40.02)[0], rtol=1e-3)


def test_far_tail_member_does_not_poison_its_relatives():
    cov = np.array([[1.0, 0.5], [0.5, 1.0]])
    draws = rtmvnorm_gibbs(cov, lower=[40.0, -np.inf], upper=[np.inf, 0.0],
                           out=(0, 1), n_sim=20_000, burn_in=500, seed=2)
    assert not np.isnan(draws).any()


@pytest.mark.parametrize("burn_in,error", [
    (-1, ValueError), (-1000, ValueError), (np.int64(-2), ValueError),
    (True, TypeError), (np.bool_(False), TypeError), (1.5, TypeError),
    (np.float64(10), TypeError), ("10", TypeError),
])
def test_rtmvnorm_gibbs_rejects_invalid_burn_in(burn_in, error):
    # the kernels loop range(-burn_in, n_sim): a negative burn-in silently skipped
    # initial output rows while n_sim draws were still credited, and a float was
    # silently truncated
    with pytest.raises(error, match="burn_in must be a non-negative integer"):
        rtmvnorm_gibbs(np.eye(1), n_sim=10, burn_in=burn_in)


def test_rtmvnorm_gibbs_accepts_zero_burn_in():
    s = rtmvnorm_gibbs(np.eye(1), n_sim=20, burn_in=0, seed=1)
    assert s.shape == (20, 1)
    assert np.isfinite(s).all()


def test_rtmvnorm_gibbs_rejects_coincident_infinite_bounds():
    # a pin at ±inf is not an observation (review 2026-08, F5)
    import numpy as _np
    from ltpred.gibbs import rtmvnorm_gibbs as _rtm
    with pytest.raises(ValueError, match="point pin"):
        _rtm(_np.eye(1), lower=[_np.inf], upper=[_np.inf], seed=1)
    with pytest.raises(ValueError, match="point pin"):
        _rtm(_np.eye(1), lower=[-_np.inf], upper=[-_np.inf], seed=1)


def test_gibbs_advance_moment_accumulates_mean_outer_product():
    # direct unit test of the research E-step kernel (review 2026-08, F36)
    import numpy as _np
    from ltpred.gibbs import gibbs_advance_moment, gibbs_params
    P, sd = gibbs_params(_np.array([[1.0]]))
    # an untruncated 1-d chain draws N(0, 1) each sweep: mean outer product -> 1
    x = _np.zeros((1, 1))
    out = gibbs_advance_moment(P, sd, _np.array([[-_np.inf]]),
                               _np.array([[_np.inf]]), _np.array([[False]]),
                               x, 50_000)
    assert out[0, 0, 0] == pytest.approx(1.0, abs=0.05)
    # a fixed coordinate never moves: the moment is exactly the pinned square
    x = _np.full((1, 1), 1.3)
    out = gibbs_advance_moment(P, sd, _np.array([[1.3]]), _np.array([[1.3]]),
                               _np.array([[True]]), x, 100)
    assert out[0, 0, 0] == pytest.approx(1.3 ** 2)
