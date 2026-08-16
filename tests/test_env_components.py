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
        e_base = np.asarray(base.est["genetic"])
        e_wired = np.asarray(wired.est["genetic"])
        self.assertTrue(np.all(np.isfinite(e_wired)))
        self.assertFalse(np.allclose(e_base, e_wired))

    def test_c_covers_the_probands_children_sibship(self):
        # c1.1/c1.2 are full sibs in one partner group; the sibship regex used
        # to match only o/s*, so C silently skipped them and a family described
        # from the children's side got a different C structure than from the
        # parents' side.
        m = construct_covmat_single(fam_vec=["c1.1", "c1.2", "c2.1", "s1"],
                                    h2=0.4, c2=0.2)
        r = {role: i for i, role in enumerate(m.roles)}
        cov = m.matrix
        self.assertAlmostEqual(cov[r["c1.1"], r["c1.2"]], 0.2 + 0.2, places=12)
        self.assertAlmostEqual(cov[r["o"], r["s1"]], 0.2 + 0.2, places=12)
        # cross-group children are only half sibs -> no C, like mhs/phs
        self.assertAlmostEqual(cov[r["c1.1"], r["c2.1"]], 0.1, places=12)
        # the kernel stays a disjoint partition, so the matrix stays PD
        self.assertGreater(np.linalg.eigvalsh(cov).min(), 0.0)


    def test_backward_compatible_defaults(self):
        sim = simulate_under_LTM_single(fam_vec=["m", "f", "s1"], h2=0.4,
                                        pop_prev=0.1, n_sim=30, seed=2)
        a = estimate_liability(sim.families, h2=0.4)
        b = estimate_liability(sim.families, h2=0.4, c2=None, m2=None)
        ea = np.asarray(a.est["genetic"])
        eb = np.asarray(b.est["genetic"])
        np.testing.assert_allclose(ea, eb, rtol=0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()


def test_pa_estimation_under_c2_matches_closed_form():
    # pinning a full sib at v is exact Gaussian conditioning:
    # E[o | s1 = v] = (0.5*h2 + c2)*v and E[g | s1 = v] = 0.5*h2*v
    # (review 2026-08, F36: first exact oracle for estimation under c2)
    import pytest
    from ltpred import estimate_liability_pa_arrays
    h2, c2, v = 0.4, 0.1, 1.1
    lower = np.array([[-np.inf, v]])
    upper = np.array([[np.inf, v]])
    est_o, var_o = estimate_liability_pa_arrays(["o", "s1"], lower, upper,
                                                h2=h2, c2=c2, out="full")
    assert est_o[0] == pytest.approx((0.5 * h2 + c2) * v, abs=1e-12)
    assert var_o[0] == pytest.approx(1.0 - (0.5 * h2 + c2) ** 2, abs=1e-12)
    est_g, var_g = estimate_liability_pa_arrays(["o", "s1"], lower, upper,
                                                h2=h2, c2=c2, out="genetic")
    assert est_g[0] == pytest.approx(0.5 * h2 * v, abs=1e-12)
