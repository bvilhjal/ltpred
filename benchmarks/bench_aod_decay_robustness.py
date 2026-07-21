"""Adversarial robustness probes for `fit_genetic_correlation_decay`.

The main `bench_aod_decay.py` grid is *circular* -- the simulator draws from
exactly the fitted covariance -- so it cannot reveal misspecification fragility.
This benchmark probes two realistic failure modes:

  arm A  **kernel misspecification**: truth is GAUSSIAN decay, fit the OU kernel;
  arm B  **unmodelled shared family environment**: the data carry a cross-relative
         environmental correlation `c2` (a family-level random effect) that the
         model does NOT include -- the extra same-trait familial covariance is
         attributed to genetics, inflating `h2` and thereby attenuating
         `r_g = G / sqrt(h2_0 h2_1)`.
  arm B0 reference: OU decay, no shared environment.

    python benchmarks/bench_aod_decay_robustness.py
    python benchmarks/bench_aod_decay_robustness.py --reps 4 --n-fam 3000 --c2 0.15
Writes bench_aod_decay_robustness.csv.
"""

import os
import csv
import time
import argparse

import numpy as np

from ltpred.covariance import get_relatedness, correct_positive_definite
from ltpred.thresholds import liability_threshold
from ltpred.family import Family, Member
from ltpred.fit import fit_genetic_correlation_decay, _decay_cov

HERE = os.path.dirname(os.path.abspath(__file__))

FAM = ["m", "f", "s1", "s2"]
H2 = [0.5, 0.5]
RG = 0.5
RP = 0.3
PREV = [0.2, 0.2]
AGE_LO, AGE_HI = 15.0, 65.0


def simulate(n_fam, lam, seed, true_kernel="ou", c2=0.0):
    """Two-trait families; optionally wrong-form decay and/or shared-family env."""
    rng = np.random.default_rng(seed)
    A = np.array([[get_relatedness(a, b, 1.0) for b in FAM] for a in FAM])
    k = len(FAM)
    h2 = np.array(H2)
    g01 = RG * np.sqrt(H2[0] * H2[1])
    G = np.array([[H2[0], g01], [g01, H2[1]]])
    E = np.array([[1 - H2[0], RP - g01], [RP - g01, 1 - H2[1]]])
    R = G + E
    np.fill_diagonal(R, 1.0)
    lam_w = np.full(2, lam)
    lam_x = np.full((2, 2), lam)
    t = [float(liability_threshold(p)) for p in PREV]
    J = np.ones((k, k)) - np.eye(k)          # shared family environmental effect
    fams = []
    for i in range(n_fam):
        aod = rng.uniform(AGE_LO, AGE_HI, size=(k, 2))
        Sig = _decay_cov(A, h2, G, R, E, aod, lam_w, lam_x, true_kernel)
        if c2 > 0.0:
            for p in range(2):               # cross-relative env correlation
                Sig[p * k:(p + 1) * k, p * k:(p + 1) * k] += c2 * J
        Sig, _ = correct_positive_definite(Sig)
        x = rng.multivariate_normal(np.zeros(k * 2), Sig)
        members = []
        for c, r in enumerate(FAM):
            lo, hi = [], []
            for p in range(2):
                case = x[p * k + c] > t[p]
                lo.append(t[p] if case else -np.inf)
                hi.append(np.inf if case else t[p])
            members.append(Member(role=r, lower=lo, upper=hi, aod=list(aod[c])))
        fams.append(Family(fam_id=i, members=members))
    return fams


def fit_rep(kw, n_fam, rep, args):
    fams = simulate(n_fam, kw["lam"], seed=args.seed + rep,
                    true_kernel=kw["true_kernel"], c2=kw["c2"])
    r = fit_genetic_correlation_decay(fams, kernel="ou", n_em=args.n_em,
                                      n_draw=args.n_draw, burn=args.burn,
                                      m_iter=args.m_iter,
                                      seed=args.seed + 100_000 + rep)
    return r.rg[0, 1], r.h2.copy(), r.lambda_cross[0, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-fam", type=int, default=2000)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--n-em", type=int, default=35)
    ap.add_argument("--n-draw", type=int, default=80)
    ap.add_argument("--burn", type=int, default=30)
    ap.add_argument("--m-iter", type=int, default=80)
    ap.add_argument("--c2", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=31)
    args = ap.parse_args()

    arms = [
        ("A_kernel_misspec", dict(true_kernel="gauss", lam=0.05, c2=0.0),
         "truth gauss lam=0.05, fit ou"),
        ("B_shared_env", dict(true_kernel="ou", lam=0.04, c2=args.c2),
         f"truth ou lam=0.04 + shared env c2={args.c2}"),
        ("B0_reference", dict(true_kernel="ou", lam=0.04, c2=0.0),
         "truth ou lam=0.04, no shared env"),
    ]
    rows = []
    print(f"robustness probes: n_fam={args.n_fam} reps={args.reps} n_em={args.n_em} "
          f"(true rg={RG})")
    for name, kw, desc in arms:
        t0 = time.time()
        rgs, h2s, lxs = [], [], []
        for rep in range(args.reps):
            rg, h2, lx = fit_rep(kw, args.n_fam, rep, args)
            rgs.append(rg); h2s.append(h2); lxs.append(lx)
        rg_m, rg_s = float(np.mean(rgs)), float(np.std(rgs, ddof=1)) if args.reps > 1 else float("nan")
        h2_m = np.mean(h2s, axis=0)
        rows.append(dict(arm=name, desc=desc, rg_mean=rg_m, rg_sd=rg_s,
                         h2_0=float(h2_m[0]), h2_1=float(h2_m[1]),
                         lam_x=float(np.mean(lxs)), n_fam=args.n_fam,
                         c2=kw["c2"], reps=args.reps))
        print(f"  {name:18s} | rg={rg_m:+.3f}  h2=[{h2_m[0]:.3f},{h2_m[1]:.3f}]  "
              f"lam_x={np.mean(lxs):.4f}   [{time.time()-t0:.0f}s]  ({desc})")

    fields = ["arm", "desc", "rg_mean", "rg_sd", "h2_0", "h2_1", "lam_x",
              "n_fam", "c2", "reps"]
    path = os.path.join(HERE, "bench_aod_decay_robustness.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {os.path.basename(path)}")


if __name__ == "__main__":
    main()
