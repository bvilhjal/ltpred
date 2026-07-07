"""The truncated-MVN Gibbs sampler against closed-form truncated-normal results."""

import numpy as np
import pytest
from scipy import stats

from ltpred.gibbs import rtmvnorm_gibbs, gibbs_params


def _imr(t):
    """Inverse Mills ratio phi(t)/(1-Phi(t)) = mean of N(0,1) above t."""
    return stats.norm.pdf(t) / stats.norm.sf(t)


def test_univariate_truncated_mean_and_var():
    # N(0,1) truncated to (0.5, 2.0): compare to scipy.truncnorm
    lo, hi = 0.5, 2.0
    s = rtmvnorm_gibbs(np.array([[1.0]]), lower=[lo], upper=[hi], out=(0,),
                       n_sim=200_000, burn_in=1000, seed=3)[:, 0]
    ref = stats.truncnorm(lo, hi)
    assert s.mean() == pytest.approx(ref.mean(), abs=0.01)
    assert s.std() == pytest.approx(ref.std(), abs=0.01)


def test_bivariate_case_posterior_genetic_liability():
    # (g, o) with cov [[h2,h2],[h2,1]]; proband is a case: o in (T, inf).
    # E[o | o>T] = IMR(T); E[g | .] = h2 * E[o | .]  since E[g|o] = h2 * o.
    h2, prev = 0.5, 0.05
    t = stats.norm.isf(prev)
    cov = np.array([[h2, h2], [h2, 1.0]])
    s = rtmvnorm_gibbs(cov, lower=[-np.inf, t], upper=[np.inf, np.inf],
                       out=(0, 1), n_sim=200_000, burn_in=1000, seed=1)
    gen, full = s.mean(axis=0)
    assert full == pytest.approx(_imr(t), abs=0.02)
    assert gen == pytest.approx(h2 * _imr(t), abs=0.02)


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
