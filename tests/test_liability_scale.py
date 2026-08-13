import numpy as np
import pytest
from scipy import stats

from ltpred.liability_scale import (liability_r2_from_z,
                                    liability_to_observed_h2,
                                    observed_to_liability_h2,
                                    probit_liability_r2)


class TestH2Bridges:
    def test_lee2011_discussion_fixture(self):
        # Lee et al. 2011 (AJHG) Discussion: K=0.01, P=0.5, factor 0.55191:
        # observed 0.18 / 0.54 / 0.91 -> liability 0.1 / 0.3 / 0.5
        out = observed_to_liability_h2([0.18, 0.54, 0.91], 0.01, 0.5)
        assert out == pytest.approx([0.1, 0.3, 0.5], abs=5e-3)

    def test_lee2011_crohns_fixture(self):
        # Table 3 (Crohn's): K=0.001, P=0.3924, factor 0.36925: 0.61 -> 0.22
        assert observed_to_liability_h2(0.61, 0.001, 0.3924) == \
            pytest.approx(0.2252, abs=5e-3)

    def test_inverse_roundtrip(self):
        liab = observed_to_liability_h2(0.15, 0.05, 0.3)
        assert liability_to_observed_h2(liab, 0.05, 0.3) == pytest.approx(0.15)

    def test_liability_to_observed_formula(self):
        liab_h2, k, p = 0.5, 0.02, 0.4
        z = stats.norm.pdf(stats.norm.isf(k))
        expected = (liab_h2 * z ** 2 / (k * (1 - k)) ** 2) * (p * (1 - p))
        assert liability_to_observed_h2(liab_h2, k, p) == pytest.approx(expected)

    def test_validation(self):
        with pytest.raises(ValueError, match="pop_prev"):
            observed_to_liability_h2(0.1, 1.5)
        with pytest.raises(ValueError, match="prop_cases"):
            observed_to_liability_h2(0.1, 0.05, 0.0)


class TestProbitLiabilityR2:
    def test_identity_values(self):
        assert probit_liability_r2(0.1, 0.5) == pytest.approx(2 * 0.25 * 0.01)
        assert probit_liability_r2(0.2, 0.1) == pytest.approx(
            2 * 0.1 * 0.9 * 0.04)

    def test_mckelvey_zavoina_fraction(self):
        # Lee et al. 2012 (Genet Epidemiol) eq. 9: var(xb) / (1 + var(xb))
        v = probit_liability_r2(0.1, 0.5)
        assert probit_liability_r2(0.1, 0.5, fraction=True) == \
            pytest.approx(v / (1 + v))

    def test_maf_validation(self):
        # 2f(1-f) is defined for any allele frequency, not only the minor one
        assert probit_liability_r2(0.1, 0.6) == pytest.approx(
            probit_liability_r2(0.1, 0.4))
        with pytest.raises(ValueError, match="maf"):
            probit_liability_r2(0.1, 1.5)
        with pytest.raises(ValueError, match="maf"):
            probit_liability_r2(0.1, -0.1)


class TestLiabilityR2FromZ:
    def test_lee_wray_form(self):
        # signal = ((z²-1)/N) * [K(1-K)/z_K²] * [K(1-K)/(P(1-P))]
        z = np.array([2.0, 3.0])
        K, P = 0.01, 0.5
        z_k = stats.norm.pdf(stats.norm.isf(K))
        c = (K * (1 - K) / z_k ** 2) * (K * (1 - K) / (P * (1 - P)))
        assert liability_r2_from_z(z, 10_000, K, P) == \
            pytest.approx(np.array([3.0, 8.0]) / 10_000 * c)

    def test_no_ascertainment_uses_link_factor_only(self):
        z = np.array([2.0])
        K = 0.05
        z_k = stats.norm.pdf(stats.norm.isf(K))
        assert liability_r2_from_z(z, 10_000, K) == \
            pytest.approx(3.0 / 10_000 * K * (1 - K) / z_k ** 2)

    def test_null_subtraction_and_raw_compatibility(self):
        K = 0.1
        assert liability_r2_from_z(1.0, 1000, K) == pytest.approx(0.0)
        assert liability_r2_from_z(0.5, 1000, K) < 0.0
        raw = liability_r2_from_z(1.0, 1000, K, subtract_null=False)
        assert raw > 0
        with pytest.raises(TypeError, match="subtract_null"):
            liability_r2_from_z(1.0, 1000, K, subtract_null="yes")

    def test_validation(self):
        with pytest.raises(ValueError, match="positive"):
            liability_r2_from_z([1.0], 0, 0.1)
        with pytest.raises(ValueError, match="pop_prev"):
            liability_r2_from_z([1.0], 100, 1.5)


class TestRecoverySimulation:
    def test_z_route_matches_identity_route(self):
        # probit-truth data: y = 1(beta * xc + eps > 0) with eps ~ N(0, 1)
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                               / "benchmarks"))
        from bench_liability_scale import probit_mle
        rng = np.random.default_rng(0)
        n, f, beta = 20_000, 0.3, 0.08
        x = rng.binomial(2, f, n)
        xc = x - x.mean()
        y = (beta * xc + rng.standard_normal(n)) > 0.0
        b, se = probit_mle(xc, y)
        P = y.mean()
        # for an unascertained sample (P ~ K) the z route and the identity
        # route agree approximately at this signal strength after subtracting
        # the unit expected null contribution: (z² - 1)/N.
        r2_z = liability_r2_from_z(b / se, n, P, P)
        r2_id = probit_liability_r2(b, f)
        assert r2_z == pytest.approx(r2_id, rel=0.3)
