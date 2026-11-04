import numpy as np
import pytest

from ltpred.cip import CipCurve, aalen_johansen_cip, kaplan_meier_cip
from ltpred.thresholds import thresholds_from_cip


class TestKaplanMeier:
    def test_hand_computed_example_and_metadata(self):
        # Events at 6, 10 and 15; censorings at 7, 13 and 20.
        c = kaplan_meier_cip(np.zeros(6), [6, 7, 10, 13, 15, 20],
                             [1, 0, 1, 0, 1, 0])
        assert isinstance(c, CipCurve)
        assert (c.n_events, c.n_entered, c.estimator) == (3, 6, "kaplan-meier")
        np.testing.assert_allclose(c.ages, [6, 10, 15])
        np.testing.assert_allclose(c.values, [1 / 6, 0.375, 0.6875],
                                   rtol=0, atol=1e-12)
        assert c.se[0] == pytest.approx(np.sqrt((5 / 6) ** 2 / 30), abs=1e-6)

    def test_left_truncation_changes_risk_set(self):
        entry, exit_, events = [0, 0, 8, 8], [6, 10, 12, 15], [1, 0, 1, 1]
        c = kaplan_meier_cip(entry, exit_, events)
        np.testing.assert_allclose(c.values, [0.5, 0.75, 1.0],
                                   rtol=0, atol=1e-12)
        aj = aalen_johansen_cip(entry, exit_, np.array(events), cause=1)
        np.testing.assert_allclose(aj.values, c.values, rtol=0, atol=1e-12)

    def test_curve_matches_product_limit_oracle(self):
        rng = np.random.default_rng(0)
        n = 200
        exit_ = rng.uniform(1, 50, n)
        events = rng.uniform(size=n) < 0.4
        c = kaplan_meier_cip(np.zeros(n), exit_, events)
        assert np.all(np.diff(c.values) >= 0)
        assert np.all((c.values >= 0) & (c.values < 1.0 + 1e-12))
        assert np.all(c.se >= 0)

        order = np.argsort(exit_)
        surv, reference = 1.0, {}
        for i, (age, event) in enumerate(zip(exit_[order], events[order])):
            if event:
                surv *= 1.0 - 1.0 / (n - i)
            reference[age] = 1.0 - surv
        np.testing.assert_allclose(
            c.values, [reference[age] for age in c.ages], atol=1e-12)
        assert c.values[-1] - c.values[0] > 0.5

    def test_validation(self):
        with pytest.raises(ValueError, match="equal length"):
            kaplan_meier_cip([0, 0], [1], [1])
        with pytest.raises(ValueError, match="nonnegative"):
            kaplan_meier_cip([-1], [1], [1])
        with pytest.raises(ValueError, match="precede"):
            kaplan_meier_cip([5], [1], [1])
        with pytest.raises(ValueError, match="finite"):
            kaplan_meier_cip([0], [np.inf], [1])
        with pytest.raises(ValueError, match="no events"):
            kaplan_meier_cip([0, 0], [1, 2], [0, 0])
        for invalid in ([2], [-1], ["0"]):
            with pytest.raises(ValueError, match="boolean or 0/1"):
                kaplan_meier_cip([0], [1], invalid)
        with pytest.raises(ValueError, match="at least one person"):
            kaplan_meier_cip([], [], [])


class TestAalenJohansen:
    def test_hand_computed_competing_risks(self):
        entry, exit_, events = np.zeros(5), [5, 7, 9, 11, 13], [1, 2, 1, 0, 1]
        c1 = aalen_johansen_cip(entry, exit_, events)
        c2 = aalen_johansen_cip(entry, exit_, events, cause=2)
        np.testing.assert_allclose(c1.ages, [5, 7, 9, 13])
        np.testing.assert_allclose(c1.values, [0.2, 0.2, 0.4, 0.8],
                                   rtol=0, atol=1e-12)
        np.testing.assert_allclose(c2.values, [0.0, 0.2, 0.2, 0.2],
                                   rtol=0, atol=1e-12)
        assert c1.values[-1] + c2.values[-1] == pytest.approx(1.0, abs=1e-12)
        np.testing.assert_allclose(
            c1.se ** 2, [1 / 25, 1 / 25, 241 / 3600, 81 / 400],
            rtol=0, atol=1e-14)
        np.testing.assert_allclose(
            c2.se ** 2, [0, 17 / 400, 17 / 400, 17 / 400],
            rtol=0, atol=1e-14)

    def test_tied_events_use_finite_risk_set_correction(self):
        c = aalen_johansen_cip(
            np.zeros(8), [1, 1, 1, 2, 2, 2.5, 3, 4],
            [1, 1, 2, 2, 2, 0, 1, 0])
        np.testing.assert_allclose(c.values, [1 / 4, 1 / 4, 7 / 16],
                                   rtol=0, atol=1e-14)
        np.testing.assert_allclose(c.se ** 2, [3 / 112, 3 / 112, 711 / 12800],
                                   rtol=0, atol=1e-14)

    def test_matches_km_without_competing_events(self):
        rng = np.random.default_rng(1)
        n = 500
        entry = rng.uniform(0, 10, n)
        exit_ = entry + rng.uniform(1, 40, n)
        events = (rng.uniform(size=n) < 0.5).astype(int)
        km = kaplan_meier_cip(entry, exit_, events)
        aj = aalen_johansen_cip(entry, exit_, events)
        np.testing.assert_allclose(aj.ages, km.ages)
        np.testing.assert_allclose(aj.values, km.values, rtol=0, atol=1e-12)
        bulk = slice(0, len(km.se) // 2)
        np.testing.assert_allclose(aj.se[bulk], km.se[bulk], rtol=0.05, atol=2e-4)

    def test_validation_and_integral_float_event_codes(self):
        with pytest.raises(ValueError, match="integer"):
            aalen_johansen_cip([0], [1], [0.5])
        with pytest.raises(ValueError, match="nonnegative"):
            aalen_johansen_cip([0], [1], [-1])
        with pytest.raises(ValueError, match="no events"):
            aalen_johansen_cip([0, 0], [1, 2], [0, 2])
        with pytest.raises(ValueError, match="0 denotes censoring"):
            aalen_johansen_cip([0, 0, 0], [1, 1, 1], [0, 0, 1], cause=0)
        with pytest.raises(TypeError, match="positive integer"):
            aalen_johansen_cip([0], [1], [1], cause=1.0)
        with pytest.raises(TypeError, match="not bool"):
            aalen_johansen_cip([0], [1], [1], cause=True)

        entry = np.zeros(6)
        exit_ = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        events = np.array([1, 0, 2, 1, 0, 2])
        reference = aalen_johansen_cip(entry, exit_, events, cause=1)
        via_float = aalen_johansen_cip(entry, exit_, events.astype(float), cause=1)
        np.testing.assert_allclose(via_float.values, reference.values)
        with pytest.raises(ValueError, match="integer code"):
            aalen_johansen_cip(entry[:3], exit_[:3], [1.0, 0.5, 2.0], cause=1)

    def test_zero_length_events_are_rejected(self):
        scenarios = [
            ([50, 50], [50, 80], np.array([1, 0])),
            ([0, 1], [1, 1], np.array([1, 1])),
        ]
        for entry, exit_, events in scenarios:
            for estimator in (kaplan_meier_cip, aalen_johansen_cip):
                with pytest.raises(ValueError, match="zero-length follow-up"):
                    estimator(entry, exit_, events)

    def test_n_entered_counts_positive_followup_contributors(self):
        for estimator in (kaplan_meier_cip, aalen_johansen_cip):
            assert estimator([0, 1], [1, 1], [1, 0]).n_entered == 1


class TestThresholdIntegration:
    def test_curve_feeds_thresholds_from_cip(self):
        c = kaplan_meier_cip(np.zeros(6), [6, 7, 10, 13, 15, 20],
                             [1, 0, 1, 0, 1, 0])
        lower, upper, K_i, K_pop = thresholds_from_cip(
            np.array([True, False]), np.array([10.0, 12.0]),
            c.ages, c.values, k_pop=0.9)
        from scipy.stats import norm
        assert lower[0] == pytest.approx(norm.ppf(1 - 0.375), abs=1e-10)
        assert upper[0] == pytest.approx(lower[0], abs=1e-12)
        expected = norm.ppf(1 - np.interp(12, c.ages, c.values))
        assert upper[1] == pytest.approx(expected, abs=1e-10)
        assert np.isnan(K_i[0]) and np.isnan(K_pop[0])
        assert K_pop[1] == pytest.approx(0.9, abs=1e-12)
