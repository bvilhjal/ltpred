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
import sys
import time
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root: `research`

from _common import simulate_families_multi
from research.advanced_fitting import fit_genetic_correlation_decay

HERE = os.path.dirname(os.path.abspath(__file__))

FAM = ["m", "f", "s1", "s2"]
H2 = [0.5, 0.5]
RG = 0.5
RP = 0.3
PREV = [0.2, 0.2]
AGE_LO, AGE_HI = 15.0, 65.0


def fit_rep(kw, n_fam, rep, args):
    fams = simulate_families_multi(FAM, H2, RG, RP, n_fam, PREV, args.seed + rep,
                                   aod=(AGE_LO, AGE_HI), lam=kw["lam"],
                                   kernel=kw["true_kernel"], c2=kw["c2"])
    r = fit_genetic_correlation_decay(fams, kernel="ou", n_em=args.n_em,
                                      n_draw=args.n_draw, burn=args.burn,
                                      m_iter=args.m_iter,
                                      seed=args.seed + 100_000 + rep,
                                      sampling="population")
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
