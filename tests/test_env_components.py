import unittest

import numpy as np

from ltpred.covariance import construct_covmat_single
from ltpred.estimate import estimate_liability
from ltpred.simulate import simulate_under_LTM_single


class CovarianceStructureTests(unittest.TestCase):
    def test_components_add_to_the_right_pairs(self):
        m = construct_covmat_single(fam_vec=["m", "f", "s1"], h2=0.4,
                                    c2=0.15, m2=0.1)
        r = {role: i for i, role in enumerate(m.roles)}
        cov = m.matrix
        # full sibs o-s1: h2 * 0.5 + c2
        self.assertAlmostEqual(cov[r["o"], r["s1"]], 0.2 + 0.15, places=12)
        # mates m-f: 0 + m2
        self.assertAlmostEqual(cov[r["m"], r["f"]], 0.1, places=12)
        # parent-offspring and diagonal: unchanged (g's variance is h2 by
        # design; the full liabilities keep unit variance)
        self.assertAlmostEqual(cov[r["o"], r["m"]], 0.2, places=12)
        non_g = [i for role, i in r.items() if role != "g"]
        np.testing.assert_allclose(np.diag(cov)[non_g], 1.0, rtol=0, atol=1e-12)
        self.assertAlmostEqual(cov[r["g"], r["g"]], 0.4, places=12)
        # the genetic row is untouched (g shares no environment)
        np.testing.assert_allclose(cov[r["g"], :],
                                   [0.4, 0.4, 0.2, 0.2, 0.2], rtol=0, atol=1e-12)

    def test_none_components_reproduce_the_base_matrix(self):
        base = construct_covmat_single(fam_vec=["m", "f", "s1", "mgm"], h2=0.3)
        for c2, m2 in ((None, None), (0.0, None), (None, 0.0)):
            m = construct_covmat_single(fam_vec=["m", "f", "s1", "mgm"], h2=0.3,
                                        c2=c2, m2=m2)
            np.testing.assert_allclose(m.matrix, base.matrix, rtol=0, atol=1e-12)

    def test_validation(self):
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            construct_covmat_single(fam_vec=["m"], h2=0.5, c2=-0.1)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            construct_covmat_single(fam_vec=["m"], h2=0.5, m2=-0.1)
        with self.assertRaisesRegex(ValueError, "must not exceed 1"):
            construct_covmat_single(fam_vec=["m"], h2=0.6, c2=0.3, m2=0.2)


class EstimationBehaviorTests(unittest.TestCase):
    def test_estimator_accepts_components_and_shifts_estimates(self):
        sim = simulate_under_LTM_single(fam_vec=["m", "f", "s1"], h2=0.4,
                                        pop_prev=0.1, n_sim=50, seed=1)
        base = estimate_liability(sim.families, h2=0.4)
        wired = estimate_liability(sim.families, h2=0.4, c2=0.1, m2=0.05)
        e_base = np.asarray(base.est["genetic"] if hasattr(base, "est")
                            else base.genetic)
        e_wired = np.asarray(wired.est["genetic"] if hasattr(wired, "est")
                             else wired.genetic)
        self.assertTrue(np.all(np.isfinite(e_wired)))
        self.assertFalse(np.allclose(e_base, e_wired))

    def test_backward_compatible_defaults(self):
        sim = simulate_under_LTM_single(fam_vec=["m", "f", "s1"], h2=0.4,
                                        pop_prev=0.1, n_sim=30, seed=2)
        a = estimate_liability(sim.families, h2=0.4)
        b = estimate_liability(sim.families, h2=0.4, c2=None, m2=None)
        ea = np.asarray(a.est["genetic"] if hasattr(a, "est") else a.genetic)
        eb = np.asarray(b.est["genetic"] if hasattr(b, "est") else b.genetic)
        np.testing.assert_allclose(ea, eb, rtol=0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
