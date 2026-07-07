"""The scalar normal CDF/PPF used by the Gibbs kernel must match SciPy."""

import numpy as np
import pytest
from scipy import stats

from ltpred._mathfun import _norm_cdf, _norm_ppf, norm_cdf, norm_ppf


@pytest.mark.parametrize("x", [-4.0, -1.5, -0.3, 0.0, 0.3, 1.5, 4.0])
def test_scalar_cdf_matches_scipy(x):
    assert _norm_cdf(x) == pytest.approx(stats.norm.cdf(x), abs=1e-12)


def test_scalar_cdf_infinities():
    assert _norm_cdf(np.inf) == 1.0
    assert _norm_cdf(-np.inf) == 0.0


@pytest.mark.parametrize("p", [1e-8, 0.01, 0.25, 0.5, 0.75, 0.99, 1 - 1e-8])
def test_scalar_ppf_matches_scipy(p):
    assert _norm_ppf(p) == pytest.approx(stats.norm.ppf(p), abs=1e-9)


def test_scalar_ppf_boundaries():
    assert _norm_ppf(0.0) == -np.inf
    assert _norm_ppf(1.0) == np.inf


def test_cdf_ppf_round_trip():
    p = np.linspace(0.001, 0.999, 50)
    assert np.allclose(norm_cdf(norm_ppf(p)), p, atol=1e-12)


def test_vectorised_matches_scipy():
    x = np.linspace(-3, 3, 25)
    assert np.allclose(norm_cdf(x), stats.norm.cdf(x))
    p = np.linspace(0.01, 0.99, 25)
    assert np.allclose(norm_ppf(p), stats.norm.ppf(p))
