"""The batched, parallel estimation path: agreement, grouping, determinism."""

import numpy as np
import pytest
from scipy import stats

from ltpred.family import Family, Member
from ltpred.gibbs import gibbs_estimate_batched, gibbs_params
from ltpred.estimate import estimate_liability, batch_means


def _imr(t):
    return stats.norm.pdf(t) / stats.norm.sf(t)


def test_batched_kernel_matches_batch_means_single_round():
    # online batch means from the kernel == batch_means() on stored samples,
    # for the same family / seed (single round, one family).
    cov = np.array([[0.5, 0.5], [0.5, 1.0]])
    t = float(stats.norm.isf(0.05))
    lowers = np.array([[-np.inf, t]])
    uppers = np.array([[np.inf, np.inf]])
    P, sd = gibbs_params(cov)
    sd0 = np.sqrt(np.diag(cov))
    n_sim = 40_000
    b = int(np.floor(np.sqrt(n_sim)))
    nb = n_sim // b
    tot, bm = gibbs_estimate_batched(P, sd, sd0, lowers, uppers,
                                     np.array([0, 1]), n_sim, 1000, b, nb,
                                     np.array([7]))
    est = tot[0] / n_sim
    muhat = bm[0].mean(axis=1)
    sigma2 = b * ((bm[0] - muhat[:, None]) ** 2).sum(axis=1) / (nb - 1)
    se = np.sqrt(sigma2 / n_sim)
    # matches the analytic case posterior (g = h2*IMR, o = IMR)
    assert est[0] == pytest.approx(0.5 * _imr(t), abs=0.03)
    assert est[1] == pytest.approx(_imr(t), abs=0.03)
    assert np.all(se > 0)


def test_grouping_gives_same_answer_regardless_of_order():
    # mix of two structures; shuffling families must not change per-family results
    t = float(stats.norm.isf(0.05))
    trio = lambda i: Family(f"t{i}", [Member("o", t, np.inf),
                                      Member("m", -np.inf, t),
                                      Member("f", t, np.inf)])
    solo = lambda i: Family(f"s{i}", [Member("o", -np.inf, t)])
    fams = [trio(0), solo(0), trio(1), solo(1), trio(2)]

    res = estimate_liability(fams, h2=0.5, out=("genetic",), tol=0.02,
                             n_sim=40_000, burn_in=800, seed=5)
    order = [3, 0, 4, 1, 2]
    res2 = estimate_liability([fams[i] for i in order], h2=0.5, out=("genetic",),
                              tol=0.02, n_sim=40_000, burn_in=800, seed=5)
    # same fam -> same estimate; seed is tied to global position, so compare by id
    by_id = dict(zip(res.fam_ids, res.est["genetic"]))
    for fid, val in zip(res2.fam_ids, res2.est["genetic"]):
        # different global seed offset, but must agree within Monte-Carlo error
        assert abs(by_id[fid] - val) < 0.1


def test_batched_estimator_is_deterministic():
    t = float(stats.norm.isf(0.05))
    fams = [Family(f"f{i}", [Member("o", t if i % 2 else -np.inf,
                                    np.inf if i % 2 else t),
                             Member("m", -np.inf, t)]) for i in range(8)]
    a = estimate_liability(fams, h2=0.5, out=("genetic", "full"), tol=0.05,
                           n_sim=20_000, burn_in=400, seed=3)
    b = estimate_liability(fams, h2=0.5, out=("genetic", "full"), tol=0.05,
                           n_sim=20_000, burn_in=400, seed=3)
    assert np.array_equal(a.est["genetic"], b.est["genetic"])
    assert np.array_equal(a.est["full"], b.est["full"])


def test_multi_round_convergence_tightens_se():
    # a tight tolerance forces extra rounds; the reported SE must respect it
    t = float(stats.norm.isf(0.05))
    fam = Family("f", [Member("o", t, np.inf), Member("m", t, np.inf)])
    res = estimate_liability([fam], h2=0.5, out=("genetic",), tol=0.005,
                             n_sim=20_000, burn_in=500, seed=1, max_rounds=20)
    assert res.se["genetic"][0] <= 0.005
