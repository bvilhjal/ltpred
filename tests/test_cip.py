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
        # Monotone + bounded + se >= 0 are all satisfied by the constant-zero
        # curve, so on their own they cannot tell a working estimator from one
        # that returns zeros. Pin the level against the textbook product-limit
        # form S(t) = prod_{t_i <= t} (1 - d_i / n_i), CIP = 1 - S.
        order = np.argsort(exit)
        surv, ref = 1.0, {}
        for i, (t, e) in enumerate(zip(exit[order], ev[order])):
            if e:
                surv *= 1.0 - 1.0 / (n - i)      # one event per distinct time
            ref[t] = 1.0 - surv
        expected = np.array([ref[t] for t in c.ages])
        np.testing.assert_allclose(c.values, expected, atol=1e-12)
        # ...and the curve must genuinely rise, so a flat estimator fails even
        # if the oracle above were ever loosened.
        self.assertGreater(c.values[-1] - c.values[0], 0.5)

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
        for invalid in ([2], [-1], ["0"]):
            with self.assertRaisesRegex(ValueError, "boolean or 0/1"):
                kaplan_meier_cip([0], [1], invalid)
        with self.assertRaisesRegex(ValueError, "at least one person"):
            kaplan_meier_cip([], [], [])


class AalenJohansenTests(unittest.TestCase):
    def test_hand_computed_competing_risks(self):
        # exits 5(c1), 7(c2), 9(c1), 11(censor), 13(c1)
        c1 = aalen_johansen_cip(np.zeros(5), [5, 7, 9, 11, 13],
                                [1, 2, 1, 0, 1])
        c2 = aalen_johansen_cip(np.zeros(5), [5, 7, 9, 11, 13],
                                [1, 2, 1, 0, 1], cause=2)
        np.testing.assert_allclose(c1.ages, [5, 7, 9, 13])
        np.testing.assert_allclose(c1.values, [0.2, 0.2, 0.4, 0.8],
                                   rtol=0, atol=1e-12)
        np.testing.assert_allclose(c2.values, [0.0, 0.2, 0.2, 0.2],
                                   rtol=0, atol=1e-12)
        # F1 + F2 = 1 - S at the horizon
        self.assertAlmostEqual(c1.values[-1] + c2.values[-1], 1.0, places=12)
        # Finite-risk Aalen variances, computed from the cmprsk recurrence.
        # The final Y=1 cause-1 event exercises the exhausted-risk-set limit.
        np.testing.assert_allclose(
            c1.se ** 2, [1 / 25, 1 / 25, 241 / 3600, 81 / 400],
            rtol=0, atol=1e-14,
        )
        np.testing.assert_allclose(
            c2.se ** 2, [0, 17 / 400, 17 / 400, 17 / 400],
            rtol=0, atol=1e-14,
        )

    def test_tied_events_use_finite_risk_set_correction(self):
        # At ages 1, 2, 3:
        #   Y       = 8, 5, 2
        #   d_cause = 2, 0, 1
        #   d_other = 1, 2, 0
        # One censor at 2.5 creates Y=2 at age 3; the other exits at age 4.
        c = aalen_johansen_cip(
            np.zeros(8),
            [1, 1, 1, 2, 2, 2.5, 3, 4],
            [1, 1, 2, 2, 2, 0, 1, 0],
        )
        np.testing.assert_allclose(
            c.values, [1 / 4, 1 / 4, 7 / 16],
            rtol=0, atol=1e-14,
        )
        # Hard-coded oracle from the finite-risk, tie-correct Aalen formula.
        # The former d/Y**2 approximation gives different values.
        np.testing.assert_allclose(
            c.se ** 2, [3 / 112, 3 / 112, 711 / 12800],
            rtol=0, atol=1e-14,
        )

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

    def test_validation(self):
        with self.assertRaisesRegex(ValueError, "integer"):
            aalen_johansen_cip([0], [1], [0.5])
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            aalen_johansen_cip([0], [1], [-1])
        with self.assertRaisesRegex(ValueError, "no events"):
            aalen_johansen_cip([0, 0], [1, 2], [0, 2])
        with self.assertRaisesRegex(ValueError, "0 denotes censoring"):
            aalen_johansen_cip(
                [0, 0, 0], [1, 1, 1], [0, 0, 1], cause=0)
        with self.assertRaisesRegex(TypeError, "positive integer"):
            aalen_johansen_cip([0], [1], [1], cause=1.0)
        with self.assertRaisesRegex(TypeError, "not bool"):
            aalen_johansen_cip([0], [1], [1], cause=True)


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

    def test_n_entered_counts_positive_followup_contributors(self):
        km = kaplan_meier_cip([0, 1], [1, 1], [1, 0])
        aj = aalen_johansen_cip([0, 1], [1, 1], [1, 0])
        self.assertEqual(km.n_entered, 1)
        self.assertEqual(aj.n_entered, 1)

class AalenJohansenGuardTests(unittest.TestCase):
    def test_zero_length_event_raises_like_kaplan_meier(self):
        # An event at entry occurs outside the documented risk interval; this
        # used to return values=[0.0] with se=1.3e154 instead of failing.
        entry, exit_ = [50, 50], [50, 80]
        with self.assertRaisesRegex(ValueError, "zero-length follow-up"):
            aalen_johansen_cip(entry, exit_, np.array([1, 0]))
        with self.assertRaisesRegex(ValueError, "zero-length follow-up"):
            kaplan_meier_cip(entry, exit_, np.array([True, False]))

    def test_zero_length_event_raises_even_with_other_risk(self):
        # One valid event and one zero-length event at the same age: Y=1,
        # d=2. The old implementation produced a negative survival step.
        with self.assertRaisesRegex(ValueError, "zero-length follow-up"):
            aalen_johansen_cip([0, 1], [1, 1], np.array([1, 1]))
        with self.assertRaisesRegex(ValueError, "zero-length follow-up"):
            kaplan_meier_cip([0, 1], [1, 1], np.array([True, True]))


if __name__ == "__main__":
    unittest.main()


def test_aalen_johansen_accepts_integral_float_event_type():
    # registry columns often load as float; KM accepted them, AJ now does too
    # (review 2026-08, F23)
    import pytest
    from ltpred import aalen_johansen_cip
    entry = np.zeros(6)
    exit_ = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    as_int = np.array([1, 0, 2, 1, 0, 2])
    ref = aalen_johansen_cip(entry, exit_, as_int, cause=1)
    via_float = aalen_johansen_cip(entry, exit_, as_int.astype(float), cause=1)
    np.testing.assert_allclose(via_float.values, ref.values)
    with pytest.raises(ValueError, match="integer code"):
        aalen_johansen_cip(entry[:3], exit_[:3], np.array([1.0, 0.5, 2.0]),
                           cause=1)
