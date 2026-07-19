import unittest

import numpy as np

from ltpred.cip import CipCurve, aalen_johansen_cip, kaplan_meier_cip
from ltpred.thresholds import thresholds_from_cip


class KaplanMeierTests(unittest.TestCase):
    def test_hand_computed_textbook_example(self):
        # 6 subjects, entries 0; exits 6(e), 7(c), 10(e), 13(c), 15(e), 20(c).
        # S: 5/6 at 6, 3/4 of that at 10, 1/2 of that at 15.
        c = kaplan_meier_cip(np.zeros(6), [6, 7, 10, 13, 15, 20],
                             [1, 0, 1, 0, 1, 0])
        np.testing.assert_allclose(c.ages, [6, 10, 15])
        np.testing.assert_allclose(c.values, [1 / 6, 0.375, 0.6875],
                                   rtol=0, atol=1e-12)
        # Greenwood at 6: var = (5/6)^2 * 1/(6*5)
        self.assertAlmostEqual(c.se[0], np.sqrt((5 / 6) ** 2 / 30), places=6)

    def test_left_truncation_changes_the_risk_set(self):
        # entries 0,0,3,5; all exits are events at 6,10,12,15.
        c = kaplan_meier_cip([0, 0, 3, 5], [6, 10, 12, 15], [1, 1, 1, 1])
        np.testing.assert_allclose(c.values, [0.25, 0.5, 0.75, 1.0],
                                   rtol=0, atol=1e-12)

    def test_cip_is_monotone_and_bounded(self):
        rng = np.random.default_rng(0)
        n = 200
        entry = np.zeros(n)
        exit = rng.uniform(1, 50, n)
        ev = rng.uniform(size=n) < 0.4
        c = kaplan_meier_cip(entry, exit, ev)
        self.assertTrue(np.all(np.diff(c.values) >= 0))
        self.assertTrue(np.all(c.values >= 0) and np.all(c.values < 1.0 + 1e-12))
        self.assertTrue(np.all(c.se >= 0))

    def test_validation(self):
        with self.assertRaisesRegex(ValueError, "equal length"):
            kaplan_meier_cip([0, 0], [1], [1])
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            kaplan_meier_cip([-1], [1], [1])
        with self.assertRaisesRegex(ValueError, "precede"):
            kaplan_meier_cip([5], [1], [1])
        with self.assertRaisesRegex(ValueError, "finite"):
            kaplan_meier_cip([0], [np.inf], [1])
        with self.assertRaisesRegex(ValueError, "no events"):
            kaplan_meier_cip([0, 0], [1, 2], [0, 0])
        with self.assertRaisesRegex(ValueError, "at least one person"):
            kaplan_meier_cip([], [], [])


class AalenJohansenTests(unittest.TestCase):
    def test_hand_computed_competing_risks(self):
        # exits 5(c1), 7(c2), 9(c1), 11(censor), 13(c1)
        c1 = aalen_johansen_cip(np.zeros(5), [5, 7, 9, 11, 13],
                                [1, 2, 1, 0, 1], n_boot=0)
        c2 = aalen_johansen_cip(np.zeros(5), [5, 7, 9, 11, 13],
                                [1, 2, 1, 0, 1], cause=2, n_boot=0)
        np.testing.assert_allclose(c1.ages, [5, 7, 9, 13])
        np.testing.assert_allclose(c1.values, [0.2, 0.2, 0.4, 0.8],
                                   rtol=0, atol=1e-12)
        np.testing.assert_allclose(c2.values, [0.0, 0.2, 0.2, 0.2],
                                   rtol=0, atol=1e-12)
        # F1 + F2 = 1 - S at the horizon
        self.assertAlmostEqual(c1.values[-1] + c2.values[-1], 1.0, places=12)

    def test_matches_km_without_competing_events(self):
        rng = np.random.default_rng(1)
        n = 500
        entry = rng.uniform(0, 10, n)
        exit = entry + rng.uniform(1, 40, n)
        ev = (rng.uniform(size=n) < 0.5).astype(int)
        km = kaplan_meier_cip(entry, exit, ev)
        aj = aalen_johansen_cip(entry, exit, ev)
        # same grid (only cause-1 events exist), same curve
        np.testing.assert_allclose(aj.ages, km.ages)
        np.testing.assert_allclose(aj.values, km.values, rtol=0, atol=1e-12)
        # the closed-form Aalen SE agrees with Greenwood where the risk set is
        # non-degenerate (the two are only asymptotically equal estimators)
        bulk = slice(0, len(km.se) // 2)
        np.testing.assert_allclose(aj.se[bulk], km.se[bulk], rtol=0.05, atol=2e-4)

    def test_bootstrap_se_deterministic_and_nonnegative(self):
        kw = dict(n_boot=25, seed=3)
        c = aalen_johansen_cip(np.zeros(5), [5, 7, 9, 11, 13],
                               [1, 2, 1, 0, 1], **kw)
        c2 = aalen_johansen_cip(np.zeros(5), [5, 7, 9, 11, 13],
                                [1, 2, 1, 0, 1], **kw)
        np.testing.assert_array_equal(c.se, c2.se)
        self.assertTrue(np.all(c.se >= 0))
        self.assertTrue(np.all(np.isfinite(c.se)))

    def test_validation(self):
        with self.assertRaisesRegex(ValueError, "integer"):
            aalen_johansen_cip([0], [1], [0.5])
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            aalen_johansen_cip([0], [1], [-1])
        with self.assertRaisesRegex(ValueError, "no events"):
            aalen_johansen_cip([0, 0], [1, 2], [0, 2])


class ThresholdIntegrationTests(unittest.TestCase):
    def test_curve_feeds_thresholds_from_cip(self):
        c = kaplan_meier_cip(np.zeros(6), [6, 7, 10, 13, 15, 20],
                             [1, 0, 1, 0, 1, 0])
        status = np.array([True, False])
        age = np.array([10.0, 12.0])
        lower, upper, K_i, K_pop = thresholds_from_cip(
            status, age, c.ages, c.values, k_pop=0.9)
        # case pinned at interp(CIP, 10) = 0.375; control below interp at 12
        from scipy.stats import norm
        self.assertAlmostEqual(lower[0], norm.ppf(1 - 0.375), places=10)
        self.assertAlmostEqual(upper[0], lower[0], places=12)
        self.assertAlmostEqual(upper[1], norm.ppf(1 - np.interp(12, c.ages, c.values)),
                               places=10)
        self.assertTrue(np.isnan(K_i[0]) and np.isnan(K_pop[0]))
        self.assertAlmostEqual(K_pop[1], 0.9, places=12)

    def test_curve_dataclass_fields(self):
        c = kaplan_meier_cip(np.zeros(6), [6, 7, 10, 13, 15, 20],
                             [1, 0, 1, 0, 1, 0])
        self.assertIsInstance(c, CipCurve)
        self.assertEqual(c.n_events, 3)
        self.assertEqual(c.n_entered, 6)
        self.assertEqual(c.estimator, "kaplan-meier")


if __name__ == "__main__":
    unittest.main()
