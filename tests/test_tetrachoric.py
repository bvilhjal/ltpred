import unittest
import warnings

import numpy as np

from ltpred.tetrachoric import (_bvn_cdf, tetrachoric, tetrachoric_matrix,
                                tetrachoric_table)


def sim_bvn(rng, rho, n, t1=0.5, t2=0.5):
    Z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=n)
    return Z[:, 0] > t1, Z[:, 1] > t2


class BvnCdfTests(unittest.TestCase):
    def test_limits_and_symmetry(self):
        # rho = 1: P(X < a, Y < b) = Phi(min(a, b))
        self.assertAlmostEqual(_bvn_cdf(1.0, 0.3, 1.0), 0.6179, places=3)
        self.assertEqual(_bvn_cdf(-np.inf, 0.0, 0.5), 0.0)
        # rho = -1 (Y = -X): P(X < a, Y < b) = max(0, Phi(a) - Phi(-b))
        self.assertAlmostEqual(_bvn_cdf(1.0, -0.5, -1.0), 0.1499, places=4)
        # joint CDF is bounded by each marginal
        v = _bvn_cdf(0.5, 0.5, 0.9)
        self.assertLessEqual(v, 0.6915)

    def test_independent_is_product(self):
        v = _bvn_cdf(0.5, -0.3, 0.0)
        from scipy.special import ndtr
        self.assertAlmostEqual(v, float(ndtr(0.5) * ndtr(-0.3)), places=10)


class MleRecoveryTests(unittest.TestCase):
    def test_recovers_rho_across_range(self):
        rng = np.random.default_rng(0)
        for rho in (-0.5, 0.0, 0.3, 0.7, 0.9):
            x, y = sim_bvn(rng, rho, 8000)
            r = tetrachoric(x, y)
            self.assertAlmostEqual(r.rho, rho, delta=0.035)

    def test_skewed_marginals(self):
        rng = np.random.default_rng(1)
        x, y = sim_bvn(rng, 0.6, 8000, t1=1.5, t2=-0.5)
        r = tetrachoric(x, y)
        self.assertAlmostEqual(r.rho, 0.6, delta=0.05)

    def test_independent_binary_data_gives_zero(self):
        rng = np.random.default_rng(2)
        x = rng.uniform(size=5000) < 0.3
        y = rng.uniform(size=5000) < 0.7
        r = tetrachoric(x, y)
        self.assertAlmostEqual(r.rho, 0.0, delta=0.05)

    def test_table_matches_array_api(self):
        rng = np.random.default_rng(3)
        x, y = sim_bvn(rng, 0.4, 4000)
        r1 = tetrachoric(x, y)
        a = np.sum(x & y)
        b = np.sum(x & ~y)
        c = np.sum(~x & y)
        d = np.sum(~x & ~y)
        r2 = tetrachoric_table(a, b, c, d)
        self.assertAlmostEqual(r1.rho, r2.rho, places=12)

    def test_zero_cell_continuity_correction(self):
        # diagonal-only table: a zero cell triggers the correction, not a crash
        r = tetrachoric_table(50, 0, 0, 50)
        self.assertTrue(r.corrected)
        self.assertGreater(r.rho, 0.9)
        # `n` reports the observed pairs, not the +2 the correction adds
        self.assertEqual(r.n, 100)
        self.assertEqual(tetrachoric_table(30, 20, 25, 25).n, 100)

    def test_se_shrinks_with_n(self):
        rng = np.random.default_rng(4)
        x, y = sim_bvn(rng, 0.5, 2000)
        x2, y2 = sim_bvn(rng, 0.5, 32000)
        self.assertLess(tetrachoric(x2, y2).se, tetrachoric(x, y).se)

    def test_validation(self):
        with self.assertRaisesRegex(ValueError, "monomorphic"):
            tetrachoric_table(50, 50, 0, 0)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            tetrachoric_table(-1, 5, 5, 5)
        with self.assertRaisesRegex(ValueError, "zero cell"):
            tetrachoric_table(50, 0, 0, 50, continuity_correction=False)
        with self.assertRaisesRegex(ValueError, "equal length"):
            tetrachoric([0, 1], [0])
        for invalid in ([0, -9], [0, 2], [0.0, np.nan], ["0", "1"]):
            with self.assertRaisesRegex(ValueError, "boolean or 0/1"):
                tetrachoric(invalid, [0, 1])


class MatrixTests(unittest.TestCase):
    def test_matrix_matches_pairwise(self):
        rng = np.random.default_rng(5)
        Z = rng.multivariate_normal([0, 0, 0],
                                    [[1, 0.5, -0.2],
                                     [0.5, 1, 0.3],
                                     [-0.2, 0.3, 1]], size=4000)
        X = Z > 0.5
        R = tetrachoric_matrix(X)
        np.testing.assert_allclose(R, R.T, rtol=0, atol=1e-12)
        np.testing.assert_allclose(np.diag(R), 1.0, rtol=0, atol=1e-12)
        for i, j, rho in ((0, 1, 0.5), (0, 2, -0.2), (1, 2, 0.3)):
            self.assertAlmostEqual(R[i, j], rho, delta=0.04)

    def test_matrix_validation(self):
        with self.assertRaisesRegex(ValueError, "2-D"):
            tetrachoric_matrix(np.zeros(5))
        with self.assertRaisesRegex(TypeError, "check_psd"):
            tetrachoric_matrix(np.zeros((5, 2)), check_psd="yes")

    def test_pairwise_matrix_warns_when_not_psd(self):
        rng = np.random.default_rng(10)
        X = rng.binomial(1, rng.uniform(0.15, 0.85, 5), size=(20, 5))
        with self.assertWarnsRegex(RuntimeWarning, "not positive-semidefinite"):
            R = tetrachoric_matrix(X)
        self.assertLess(np.linalg.eigvalsh(R).min(), -1e-3)
        # The caller can suppress the diagnostic after making an explicit choice.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            tetrachoric_matrix(X, check_psd=False)


if __name__ == "__main__":
    unittest.main()
