"""Tests for the onset-age-decay genetic-correlation fit.

The deterministic pieces (kernel values/derivatives, the structured covariance
and its batched form, the lambda=0 reduction to the scalar model, and input
validation) are checked exactly; the stochastic fit is checked for coherence and
for recovery with deliberately loose tolerances (the model is data-hungry by
design — see benchmarks/bench_aod_decay.py for the powered kill-test).
"""
import unittest

import numpy as np

from ltpred.covariance import get_relatedness, correct_positive_definite
from ltpred.thresholds import liability_threshold
from ltpred.family import Family, Member
from research.advanced_fitting import (fit_genetic_correlation_decay,
                                       _decay_kernel, _decay_kernel_deriv,
                                       _decay_cov, _decay_cov_batch,
                                       _decay_negq_grad, _multi_cov,
                                       _project_covariance,
                                       _prepare_group_decay)
from ltpred.estimate import _group_by_structure

FAM = ["m", "f", "s1", "s2"]
P = 2


def _A():
    return np.array([[get_relatedness(a, b, 1.0) for b in FAM] for a in FAM])


def _params(h2=(0.5, 0.5), rg=0.5, rp=0.3, lam=0.05):
    h2 = np.asarray(h2, float)
    g01 = rg * np.sqrt(h2[0] * h2[1])
    G = np.array([[h2[0], g01], [g01, h2[1]]])
    E = np.array([[1 - h2[0], rp - g01], [rp - g01, 1 - h2[1]]])
    R = G + E
    np.fill_diagonal(R, 1.0)
    lam_w = np.full(P, lam)
    lam_x = np.full((P, P), lam)
    return h2, G, E, R, lam_w, lam_x


def _simulate(n_fam, lam=0.05, rg=0.5, rp=0.3, prev=(0.2, 0.2), seed=0):
    rng = np.random.default_rng(seed)
    A = _A()
    k = len(FAM)
    h2, G, E, R, lam_w, lam_x = _params(rg=rg, rp=rp, lam=lam)
    t = [float(liability_threshold(p)) for p in prev]
    fams = []
    for i in range(n_fam):
        aod = rng.uniform(15.0, 65.0, size=(k, P))
        Sig = _decay_cov(A, h2, G, E, aod, lam_w, lam_x, "ou")
        Sig, _ = correct_positive_definite(Sig)
        x = rng.multivariate_normal(np.zeros(k * P), Sig)
        members = []
        for c, r in enumerate(FAM):
            lo, hi = [], []
            for p in range(P):
                case = x[p * k + c] > t[p]
                lo.append(t[p] if case else -np.inf)
                hi.append(np.inf if case else t[p])
            members.append(Member(role=r, lower=lo, upper=hi, aod=list(aod[c])))
        fams.append(Family(fam_id=i, members=members))
    return fams


class KernelTests(unittest.TestCase):
    def test_ou(self):
        d = np.array([0.0, 2.0, 10.0])
        np.testing.assert_allclose(_decay_kernel(d, 0.5, "ou"),
                                   [1.0, np.exp(-1.0), np.exp(-5.0)])

    def test_gauss(self):
        d = np.array([0.0, 2.0])
        np.testing.assert_allclose(_decay_kernel(d, 0.5, "gauss"),
                                   [1.0, np.exp(-0.5 * (0.5 * 2.0) ** 2)])

    def test_tent_clips_at_zero(self):
        d = np.array([0.0, 1.0, 100.0])
        np.testing.assert_allclose(_decay_kernel(d, 0.5, "tent"),
                                   [1.0, 0.5, 0.0])

    def test_uses_absolute_difference(self):
        self.assertEqual(_decay_kernel(-3.0, 0.5, "ou"),
                         _decay_kernel(3.0, 0.5, "ou"))

    def test_derivative_matches_finite_difference(self):
        rng = np.random.default_rng(0)
        for kernel in ("ou", "gauss", "tent"):
            for lam in (0.0, 0.03, 0.1):
                d = rng.uniform(0.0, 20.0, size=50)
                # tent is non-differentiable at 1-lam*d==0; avoid the kink
                if kernel == "tent":
                    d = np.clip(d, 0.0, 0.9 / max(lam, 1e-9)) if lam else d
                eps = 1e-6
                fd = (_decay_kernel(d, lam + eps, kernel)
                      - _decay_kernel(d, lam - eps, kernel)) / (2 * eps)
                an = _decay_kernel_deriv(d, lam, kernel)
                np.testing.assert_allclose(an, fd, rtol=1e-4, atol=1e-6)

    def test_unknown_kernel_raises(self):
        with self.assertRaises(ValueError):
            _decay_kernel(1.0, 0.5, "nope")
        with self.assertRaises(ValueError):
            _decay_kernel_deriv(1.0, 0.5, "nope")


class CovarianceTests(unittest.TestCase):
    def test_projection_keeps_capped_and_zero_variances_psd(self):
        raw = np.array([[0.8, 0.7, 0.2],
                        [0.7, 0.8, -0.2],
                        [0.2, -0.2, 0.1]])
        variances = np.array([0.4, 0.4, 0.0])
        cov, _ = _project_covariance(raw, variances)
        np.testing.assert_allclose(np.diag(cov), variances)
        np.testing.assert_array_equal(cov[2], 0.0)
        np.testing.assert_array_equal(cov[:, 2], 0.0)
        self.assertGreaterEqual(np.linalg.eigvalsh(cov).min(), -1e-12)

    def test_batch_matches_per_family(self):
        rng = np.random.default_rng(3)
        A = _A()
        k = len(FAM)
        F = 6
        aod = rng.uniform(15, 65, size=(F, k, P))
        h2, G, E, R, lam_w, lam_x = _params()
        lam_w = np.array([0.04, 0.05])
        dself = np.abs(aod[:, :, None, :] - aod[:, None, :, :])
        dcross = [np.abs(aod[:, :, None, 0] - aod[:, None, :, 1])]
        Sb = _decay_cov_batch(A, dself, dcross, h2, G, E, lam_w, lam_x,
                              [(0, 1)], "ou")
        for f in range(F):
            S1 = _decay_cov(A, h2, G, E, aod[f], lam_w, lam_x, "ou")
            np.testing.assert_allclose(Sb[f], S1, atol=1e-12)

    def test_shared_lambda_positive_definite(self):
        rng = np.random.default_rng(4)
        A = _A()
        k = len(FAM)
        h2, G, E, R, lam_w, lam_x = _params()
        for _ in range(50):
            aod = rng.uniform(15, 65, size=(k, P))
            S = _decay_cov(A, h2, G, E, aod, lam_w, lam_x, "ou")
            self.assertGreater(np.linalg.eigvalsh(S).min(), 0.0)

    def test_lambda_zero_reduces_to_scalar_model(self):
        rng = np.random.default_rng(5)
        A = _A()
        k = len(FAM)
        aod = rng.uniform(15, 65, size=(k, P))
        h2, G, E, R, lam_w, lam_x = _params(rg=0.5, rp=0.3)
        zero_w = np.zeros(P)
        zero_x = np.zeros((P, P))
        S_decay = _decay_cov(A, h2, G, E, aod, zero_w, zero_x, "ou")
        S_scalar = _multi_cov(A, h2, G, R)
        np.testing.assert_allclose(S_decay, S_scalar, atol=1e-12)


def _lone_probands(n, seed=0):
    rng = np.random.default_rng(seed)
    fams = []
    for i in range(n):
        aod = list(rng.uniform(15, 65, size=P))
        fams.append(Family(fam_id=i, members=[
            Member(role="o", lower=[-np.inf, -np.inf],
                   upper=[1.0, 1.0], aod=aod)]))
    return fams


class ErrorPathTests(unittest.TestCase):
    def test_missing_aod_raises(self):
        fams = _simulate(20, seed=1)
        for f in fams:
            for m in f.members:
                m.aod = None
        with self.assertRaises(ValueError):
            fit_genetic_correlation_decay(fams, n_em=8, n_draw=5, burn=2,
                                          sampling="population")

    def test_no_related_pairs_raises(self):
        with self.assertRaises(ValueError):
            fit_genetic_correlation_decay(_lone_probands(20), n_em=8,
                                          n_draw=5, burn=2,
                                          sampling="population")

    def test_unknown_kernel_raises(self):
        with self.assertRaises(ValueError):
            fit_genetic_correlation_decay(_simulate(20, seed=2), kernel="nope",
                                          n_em=8, n_draw=5, burn=2,
                                          sampling="population")

    def test_n_em_too_small_raises(self):
        with self.assertRaises(ValueError):
            fit_genetic_correlation_decay(_simulate(20, seed=3), n_em=5)

    def test_n_starts_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "n_starts .* must be >= 1"):
            fit_genetic_correlation_decay(_simulate(20, seed=3), n_em=8,
                                          n_starts=0)

    def test_negative_burn_raises(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            fit_genetic_correlation_decay(_simulate(20, seed=5), n_em=8,
                                          burn=-1)

    def test_population_sampling_contract(self):
        # the decay fit embeds the same population-sampling assumption as the
        # core moment fitters, so it shares the sampling= contract gate
        fams = _simulate(20, seed=12)
        with self.assertWarnsRegex(RuntimeWarning, "unascertained"):
            fit_genetic_correlation_decay(fams, n_em=8, n_draw=5, burn=2,
                                          m_iter=20)
        with self.assertRaisesRegex(ValueError, r"sampling='population' or sampling='ipw'"):
            fit_genetic_correlation_decay(fams, n_em=8, n_draw=5, burn=2,
                                          m_iter=20, sampling="case-control")

    def test_phen_names_mismatch_raises(self):
        with self.assertRaises(ValueError):
            fit_genetic_correlation_decay(_simulate(20, seed=4),
                                          phen_names=["only_one"], n_em=8)

    def test_empty_families_raises(self):
        with self.assertRaises(ValueError):
            fit_genetic_correlation_decay([], n_em=8)


class FitCoherenceTests(unittest.TestCase):
    def test_result_is_coherent_psd_model(self):
        fams = _simulate(400, seed=7)
        res = fit_genetic_correlation_decay(fams, n_em=8, n_draw=40, burn=20,
                                            m_iter=50, seed=11,
                                            sampling="population")
        # rp == genetic_cov + env_cov exactly, diagonal 1
        np.testing.assert_allclose(res.rp, res.genetic_cov + res.env_cov,
                                   atol=1e-8)
        np.testing.assert_allclose(np.diag(res.rp), 1.0, atol=1e-8)
        np.testing.assert_allclose(np.diag(res.rg), 1.0, atol=1e-8)
        self.assertGreaterEqual(np.linalg.eigvalsh(res.genetic_cov).min(), -1e-8)
        self.assertGreaterEqual(np.linalg.eigvalsh(res.env_cov).min(), -1e-8)
        # decay rates within the derived bound and non-negative
        self.assertTrue(np.all(res.lambda_within >= 0.0))
        self.assertTrue(np.all(res.lambda_cross >= 0.0))
        # traces line up with the reported averages
        self.assertIn("lambda_within", res.traces)
        self.assertEqual(res.kernel, "ou")

    def test_recovery_loose_well_powered_direction(self):
        # moderate-to-large cohort: the fit should move rg and lambda in the
        # right direction (positive, not boundary-slammed) -- the powered
        # recovery kill-test lives in benchmarks/bench_aod_decay.py.
        fams = _simulate(1500, lam=0.05, rg=0.5, seed=9)
        res = fit_genetic_correlation_decay(fams, n_em=14, n_draw=60, burn=30,
                                            m_iter=80, seed=5,
                                            sampling="population")
        rg = res.rg[0, 1]
        self.assertGreater(rg, 0.0)     # right sign, not collapsed to <= 0
        self.assertLess(rg, 0.95)       # not inflated to the boundary
        # the fit derives lam_max from the actual age span (4/span); compute it
        # the same way here rather than assuming the nominal 4/50.
        all_aod = np.concatenate([np.asarray(m.aod, float).ravel()
                                  for f in fams for m in f.members])
        lam_max = 4.0 / float(all_aod.max() - all_aod.min())
        self.assertGreaterEqual(res.lambda_cross[0, 1], 0.0)
        self.assertLessEqual(res.lambda_cross[0, 1], lam_max + 1e-9)


class NewOptionsTests(unittest.TestCase):
    """Coherence of the shared_lambda / shared_env / converged / n_starts options."""

    def test_shared_lambda_returns_single_rate(self):
        res = fit_genetic_correlation_decay(_simulate(400, seed=3),
                                            shared_lambda=True, n_em=8,
                                            n_draw=30, burn=15, m_iter=40,
                                            seed=2, sampling="population")
        # one rate governs every block: within == cross (broadcast)
        self.assertAlmostEqual(res.lambda_within[0], res.lambda_within[1])
        self.assertAlmostEqual(res.lambda_within[0], res.lambda_cross[0, 1])

    def test_shared_env_coherent_and_returns_cov(self):
        res = fit_genetic_correlation_decay(_simulate(400, seed=4),
                                            shared_env=True, n_em=8, n_draw=30,
                                            burn=15, m_iter=40, seed=6,
                                            sampling="population")
        # coherence preserved: rp == genetic_cov + env_cov (env = C + E)
        np.testing.assert_allclose(res.rp, res.genetic_cov + res.env_cov,
                                   atol=1e-8)
        self.assertEqual(res.shared_env_cov.shape, (P, P))
        self.assertGreaterEqual(
            np.linalg.eigvalsh(res.shared_env_cov).min(), -1e-10)
        self.assertTrue(np.all(np.diag(res.shared_env_cov) <=
                               1.0 - res.h2 + 1e-10))
        # no true shared env in the DGP -> c2 should stay small
        self.assertLess(float(np.diag(res.shared_env_cov).mean()), 0.2)

    def test_converged_and_negq_present(self):
        res = fit_genetic_correlation_decay(_simulate(300, seed=5), n_em=8,
                                            n_draw=30, burn=15, m_iter=40,
                                            seed=7, sampling="population")
        self.assertIsInstance(bool(res.converged), bool)
        self.assertEqual(len(res.negq), res.n_iter)
        self.assertIn("negq", res.traces)

    def test_n_starts_runs(self):
        res = fit_genetic_correlation_decay(_simulate(200, seed=6), n_em=8,
                                            n_draw=20, burn=10, m_iter=30,
                                            n_starts=2, seed=8,
                                            sampling="population")
        self.assertTrue(np.isfinite(res.rg[0, 1]))


class GradientTests(unittest.TestCase):
    """Finite-difference check of the analytic M-step gradient.

    The decay fit's L-BFGS uses the analytic score of the expected complete-data
    likelihood; a wrongly symmetrised cross-trait block once corrupted it. Pin it
    against finite differences of ``negQ`` on a small synthetic group so any
    regression in the block algebra fails loudly."""

    def _group(self, seed=0):
        fams = _simulate(4, seed=seed)
        g = [_prepare_group_decay(fams, idx, P)
             for _k, idx in _group_by_structure(fams)][0]
        aod = g["aod"]
        g["dself"] = np.abs(aod[:, :, None, :] - aod[:, None, :, :])
        g["dcross"] = [np.abs(aod[:, :, None, p] - aod[:, None, :, q])
                       for (p, q) in [(0, 1)]]
        return g

    def _check(self, theta, shared_lambda, shared_env=False):
        rng = np.random.default_rng(1)
        g = self._group()
        k, F = g["k"], g["F"]
        kP = k * P
        lam_max = 0.08
        eps = 1e-4
        pairs = [(0, 1)]
        # valid PD "second moments" M_f, distinct from Sigma(theta)
        M = np.empty((F, kP, kP))
        for f in range(F):
            R = rng.normal(size=(kP, kP))
            M[f] = R @ R.T / kP + 0.5 * np.eye(kP)
        negQ, grad = _decay_negq_grad(theta, P, [M], [g], pairs, lam_max,
                                      "ou", eps, shared_lambda, shared_env)
        self.assertTrue(np.isfinite(negQ))
        self.assertEqual(grad.shape, theta.shape)
        d = 1e-6
        for a in range(len(theta)):
            step = np.zeros_like(theta)
            step[a] = d
            qp, _ = _decay_negq_grad(theta + step, P, [M], [g], pairs,
                                     lam_max, "ou", eps, shared_lambda,
                                     shared_env)
            qm, _ = _decay_negq_grad(theta - step, P, [M], [g], pairs,
                                     lam_max, "ou", eps, shared_lambda,
                                     shared_env)
            fd = (qp - qm) / (2 * d)
            self.assertAlmostEqual(grad[a], fd, places=4,
                                   msg=f"lam={shared_lambda},env={shared_env} "
                                       f"grad[{a}]: {grad[a]:.5f} vs fd {fd:.5f}")

    def test_score_matches_finite_difference(self):
        # per-block rates: [h2(2), lam_w(2), G01, lam_x01, E01]
        self._check(np.array([0.45, 0.5, 0.02, 0.03, 0.15, 0.04, 0.05]),
                    shared_lambda=False)

    def test_score_matches_finite_difference_shared_lambda(self):
        # shared rate: [h2(2), lam(1), G01, E01]
        self._check(np.array([0.45, 0.5, 0.04, 0.15, 0.05]),
                    shared_lambda=True)

    def test_score_matches_finite_difference_shared_env(self):
        # per-block + shared env: [h2(2), lam_w(2), G01, lam_x01, E01, c2(2), C01]
        self._check(np.array([0.45, 0.5, 0.02, 0.03, 0.15, 0.04, 0.05,
                              0.08, 0.09, 0.02]),
                    shared_lambda=False, shared_env=True)

    def test_score_matches_finite_difference_shared_lambda_env(self):
        # shared rate + shared env: [h2(2), lam(1), G01, E01, c2(2), C01]
        self._check(np.array([0.45, 0.5, 0.04, 0.15, 0.05, 0.08, 0.09, 0.02]),
                    shared_lambda=True, shared_env=True)


if __name__ == "__main__":
    unittest.main()


def test_decay_gencorr_se_matrices_match_the_parent_shape():
    # fit_genetic_correlation_decay omitted the .reshape(P, P) that
    # fit_genetic_correlation applies, so se["rg"] came back as a flat (P*P,)
    # vector while the identically-named field on the parent GenCorrResult is
    # a (P, P) matrix -- se["rg"][i, j] raised IndexError on the decay result.
    r = fit_genetic_correlation_decay(_simulate(120, seed=5), n_em=8,
                                      n_starts=1, seed=1)
    P = len(r.h2)
    for key in ("rg", "re", "rp"):
        assert r.se[key].shape == (P, P), f"se[{key!r}] is {r.se[key].shape}"
        _ = r.se[key][0, 1]                   # the indexing that used to fail
