import warnings

import numpy as np
import pytest
from scipy.special import ndtr

from ltpred.tetrachoric import (_bvn_cdf, tetrachoric, tetrachoric_matrix,
                                tetrachoric_table)


def sim_bvn(rng, rho, n, t1=0.5, t2=0.5):
    z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=n)
    return z[:, 0] > t1, z[:, 1] > t2


class TestBvnCdf:
    def test_limits_and_marginal_bound(self):
        assert _bvn_cdf(1.0, 0.3, 1.0) == pytest.approx(0.6179, abs=5e-4)
        assert _bvn_cdf(-np.inf, 0.0, 0.5) == 0.0
        assert _bvn_cdf(1.0, -0.5, -1.0) == pytest.approx(0.1499, abs=5e-5)
        assert _bvn_cdf(0.5, 0.5, 0.9) <= 0.6915

    def test_independent_is_product(self):
        value = _bvn_cdf(0.5, -0.3, 0.0)
        assert value == pytest.approx(float(ndtr(0.5) * ndtr(-0.3)), abs=1e-10)


class TestMleRecovery:
    def test_recovers_rho_across_range(self):
        rng = np.random.default_rng(0)
        for rho in (-0.5, 0.0, 0.3, 0.7, 0.9):
            x, y = sim_bvn(rng, rho, 8000)
            assert tetrachoric(x, y).rho == pytest.approx(rho, abs=0.035)

    def test_skewed_marginals(self):
        x, y = sim_bvn(np.random.default_rng(1), 0.6, 8000, t1=1.5, t2=-0.5)
        assert tetrachoric(x, y).rho == pytest.approx(0.6, abs=0.05)

    def test_independent_binary_data_gives_zero(self):
        rng = np.random.default_rng(2)
        x = rng.uniform(size=5000) < 0.3
        y = rng.uniform(size=5000) < 0.7
        assert tetrachoric(x, y).rho == pytest.approx(0.0, abs=0.05)

    def test_table_matches_array_api(self):
        x, y = sim_bvn(np.random.default_rng(3), 0.4, 4000)
        by_array = tetrachoric(x, y)
        counts = (np.sum(x & y), np.sum(x & ~y),
                  np.sum(~x & y), np.sum(~x & ~y))
        assert tetrachoric_table(*counts).rho == pytest.approx(
            by_array.rho, abs=1e-12)

    def test_zero_cell_continuity_correction(self):
        result = tetrachoric_table(50, 0, 0, 50)
        assert result.corrected
        assert result.rho > 0.9
        assert result.n == 100
        assert tetrachoric_table(30, 20, 25, 25).n == 100

    def test_se_shrinks_with_n(self):
        x, y = sim_bvn(np.random.default_rng(4), 0.5, 2000)
        x2, y2 = sim_bvn(np.random.default_rng(4), 0.5, 32000)
        assert tetrachoric(x2, y2).se < tetrachoric(x, y).se

    def test_validation(self):
        with pytest.raises(ValueError, match="monomorphic"):
            tetrachoric_table(50, 50, 0, 0)
        with pytest.raises(ValueError, match="nonnegative"):
            tetrachoric_table(-1, 5, 5, 5)
        with pytest.raises(ValueError, match="zero cell"):
            tetrachoric_table(50, 0, 0, 50, continuity_correction=False)
        with pytest.raises(ValueError, match="equal length"):
            tetrachoric([0, 1], [0])
        for invalid in ([0, -9], [0, 2], [0.0, np.nan], ["0", "1"]):
            with pytest.raises(ValueError, match="boolean or 0/1"):
                tetrachoric(invalid, [0, 1])

        with pytest.raises(ValueError, match="integers"):
            tetrachoric_table(10.5, 5, 3, 4)
        assert np.isfinite(tetrachoric_table(10.0, 5.0, 3.0, 4.0).rho)
        for bad in (np.inf, np.nan):
            with pytest.raises(ValueError, match="finite"):
                tetrachoric_table(bad, 5, 3, 4)


class TestMatrix:
    def test_matrix_matches_pairwise(self):
        rng = np.random.default_rng(5)
        z = rng.multivariate_normal(
            [0, 0, 0], [[1, 0.5, -0.2], [0.5, 1, 0.3], [-0.2, 0.3, 1]],
            size=4000)
        correlation = tetrachoric_matrix(z > 0.5)
        np.testing.assert_allclose(correlation, correlation.T, rtol=0, atol=1e-12)
        np.testing.assert_allclose(np.diag(correlation), 1.0, rtol=0, atol=1e-12)
        for i, j, rho in ((0, 1, 0.5), (0, 2, -0.2), (1, 2, 0.3)):
            assert correlation[i, j] == pytest.approx(rho, abs=0.04)

    def test_matrix_validation(self):
        with pytest.raises(ValueError, match="2-D"):
            tetrachoric_matrix(np.zeros(5))
        with pytest.raises(TypeError, match="check_psd"):
            tetrachoric_matrix(np.zeros((5, 2)), check_psd="yes")

    def test_pairwise_matrix_warns_when_not_psd(self):
        rng = np.random.default_rng(10)
        x = rng.binomial(1, rng.uniform(0.15, 0.85, 5), size=(20, 5))
        with pytest.warns(RuntimeWarning, match="not positive-semidefinite"):
            correlation = tetrachoric_matrix(x)
        assert np.linalg.eigvalsh(correlation).min() < -1e-3
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            tetrachoric_matrix(x, check_psd=False)
