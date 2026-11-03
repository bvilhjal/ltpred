"""Genetic-correlation inference: does `fit_genetic_correlation` recover r_g?

`fit_genetic_correlation` estimates the genetic correlation between traits from
family case/control data by a cross-trait Haseman–Elston regression (the
multi-trait analogue of `fit_heritability`). As for the single-trait fits, the
properties that matter are **bias** and **across-dataset sampling variability**,
so this benchmark fits many independent simulated two-trait cohorts and reads the
spread of the fitted r_g as its sampling distribution. It reports:

  (a) **bias & precision vs true r_g** — including the **null (r_g = 0 with a
      non-zero phenotypic correlation)**, which must not manufacture a genetic
      correlation;
  (b) **precision vs #families** — the across-replicate SD of r_g as N grows;
  (c) **common-factor structure of r_g** (P = 5 traits, absorbed from
      `bench_genetic_factor.py`) — the full pipeline simulate →
      `fit_genetic_correlation` → `fit_genetic_factor` (a Genomic-SEM-style
      model `r_g ≈ Λ Λ' + Ψ`), reporting planted single-factor loading
      recovery and the `srmr` contrast of 1- vs 2-factor fits when the truth
      is two independent genetic factors (a planted diagnostic, not a
      calibrated factor-number test).

Ground truth is the simulator's genetic-correlation matrix / factor loadings.

    python benchmarks/bench_genetic_correlation.py
    python benchmarks/bench_genetic_correlation.py --reps 40 --n-fam 4000
Writes bench_genetic_correlation.csv.
"""

import os
import sys
import time
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root: `research`

from _common import (simulate_families_multi,
                     write_rows)
from research.advanced_fitting import fit_genetic_correlation, fit_genetic_factor

HERE = os.path.dirname(os.path.abspath(__file__))

FAM = ["m", "f", "s1", "s2"]
H2 = [0.5, 0.4]
RP_OFFDIAG = 0.2                       # phenotypic (full-liability) correlation

# panel (c): factor model over P = 5 traits
H2_FAC = [0.5, 0.45, 0.4, 0.35, 0.3]              # per-trait heritabilities
LOADINGS_1F = np.array([0.8, 0.7, 0.6, 0.5, 0.4])   # one common factor
# two-factor truth: block {0,1,2} and block {3,4}, zero genetic corr across blocks
BLOCK_A = np.array([0.8, 0.7, 0.6])
BLOCK_B = np.array([0.7, 0.6])


def rg_one_factor():
    R = np.outer(LOADINGS_1F, LOADINGS_1F)
    np.fill_diagonal(R, 1.0)
    return R


def rg_two_factor():
    P = len(H2_FAC)
    R = np.zeros((P, P))
    R[:3, :3] = np.outer(BLOCK_A, BLOCK_A)
    R[3:, 3:] = np.outer(BLOCK_B, BLOCK_B)
    np.fill_diagonal(R, 1.0)
    return R


def fit_pipeline(rg_true, n_fam, prev, seed, n_iter, burn_in, n_factors=1):
    fams = simulate_families_multi(FAM, H2_FAC, rg_true, None, n_fam, prev, seed)
    gc = fit_genetic_correlation(fams, n_iter=n_iter, burn_in=burn_in,
                                 seed=seed + 100_000, sampling="population")
    fac = fit_genetic_factor(gc, n_factors=n_factors)
    return fac


def fit_replicates(rg_val, n_fam, prev, reps, seed0, n_iter, burn_in):
    fitted = np.empty(reps)
    for r in range(reps):
        fams = simulate_families_multi(FAM, H2, rg_val, RP_OFFDIAG, n_fam,
                                       (prev, prev), seed0 + r)
        res = fit_genetic_correlation(fams, n_iter=n_iter, burn_in=burn_in,
                                      seed=seed0 + 100_000 + r,
                                      sampling="population")
        fitted[r] = res.rg[0, 1]
    return fitted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rg", type=float, nargs="+", default=[0.0, 0.3, 0.6])
    ap.add_argument("--n-fam", type=int, default=3000)
    ap.add_argument("--sizes", type=int, nargs="+", default=[1000, 2000, 4000, 8000])
    ap.add_argument("--prev", type=float, default=0.1)
    ap.add_argument("--reps", type=int, default=25)
    ap.add_argument("--reps-scaling", type=int, default=15)
    ap.add_argument("--n-iter", type=int, default=800)
    ap.add_argument("--burn-in", type=int, default=250)
    ap.add_argument("--seed", type=int, default=200)
    ap.add_argument("--reps-factor", type=int, default=15)
    ap.add_argument("--seed-factor", type=int, default=300)
    args = ap.parse_args()

    fit_genetic_correlation(simulate_families_multi(FAM, H2, 0.5, 0.2, 60,
                                                    (0.1, 0.1), 0),
                            n_iter=20, burn_in=5,
                            sampling="population")   # warm JIT
    fit_pipeline(rg_one_factor(), 60, 0.1, 0, 20, 5)     # warm JIT (factor fit)

    rows = []

    # (a) bias & precision vs true r_g (incl null) ------------------------------
    print("== (a) bias & precision vs true r_g (n_fam=%d, h2=%s, r_p=%.1f) =="
          % (args.n_fam, H2, RP_OFFDIAG))
    panel_a = []
    for rg in args.rg:
        t0 = time.time()
        fitted = fit_replicates(rg, args.n_fam, args.prev, args.reps, args.seed,
                                args.n_iter, args.burn_in)
        bias = float(fitted.mean() - rg)
        sd = float(fitted.std(ddof=1))
        panel_a.append(dict(rg=rg, bias=bias, sd=sd))
        rows.append(dict(panel="bias_precision", rg=rg, n_fam=args.n_fam,
                         bias=bias, sd=sd, reps=args.reps))
        tag = " (null)" if rg == 0.0 else ""
        print(f"  r_g={rg:.1f}{tag} | fitted={fitted.mean():+.3f}  bias={bias:+.3f}  "
              f"SD={sd:.3f}  [{time.time()-t0:.0f}s]")

    # (b) precision vs #families (r_g=0.5) --------------------------------------
    print("== (b) precision (SD of r_g) vs #families (r_g=0.5) ==")
    panel_b = []
    for n in args.sizes:
        fitted = fit_replicates(0.5, n, args.prev, args.reps_scaling, args.seed + 1000,
                                args.n_iter, args.burn_in)
        sd = float(fitted.std(ddof=1))
        panel_b.append(dict(n_fam=n, sd=sd, reps=args.reps_scaling))
        rows.append(dict(panel="scaling", rg=0.5, n_fam=n,
                         bias=float(fitted.mean() - 0.5), sd=sd, reps=args.reps_scaling))
        print(f"  n_fam={n:5d} | SD={sd:.3f}")

    # (c) genetic factor model: `fit_genetic_factor` on the fitted r_g (P = 5) --
    print("== (c) genetic factor model (P=%d, n_fam=%d, prev=%.2f) =="
          % (len(H2_FAC), args.n_fam, args.prev))
    P = len(H2_FAC)

    # (c.i) loading recovery under a true single factor -------------------------
    L = np.empty((args.reps_factor, P))
    srmr_1f = np.empty(args.reps_factor)
    prop_1f = np.empty(args.reps_factor)
    t0 = time.time()
    for r in range(args.reps_factor):
        fac = fit_pipeline(rg_one_factor(), args.n_fam, args.prev,
                           args.seed_factor + r, args.n_iter, args.burn_in,
                           n_factors=1)
        L[r] = fac.loadings.ravel()
        srmr_1f[r] = fac.srmr
        prop_1f[r] = fac.prop_explained
        if (r + 1) % 5 == 0 or r + 1 == args.reps_factor:
            print(f"  recovery replicate {r + 1}/{args.reps_factor}", flush=True)
    Lmean, Lsd = L.mean(0), L.std(0, ddof=1)
    for p in range(P):
        print("  trait %d: true=%.2f  fitted=%.3f ± %.3f" % (p, LOADINGS_1F[p],
              Lmean[p], Lsd[p]))
        rows.append(dict(panel="recovery", trait=p, true=LOADINGS_1F[p],
                         fitted=Lmean[p], sd=Lsd[p], n_fam=args.n_fam,
                         reps=args.reps_factor))
    print("  one-factor fit: srmr=%.3f ± %.3f   prop_explained=%.3f   [%.0fs]"
          % (srmr_1f.mean(), srmr_1f.std(ddof=1), prop_1f.mean(), time.time() - t0))
    rows.append(dict(panel="fit_1f_truth", srmr=srmr_1f.mean(),
                     srmr_sd=srmr_1f.std(ddof=1), prop=prop_1f.mean(),
                     n_fam=args.n_fam, reps=args.reps_factor))

    # (c.ii) mis-specification: two-factor truth, 1- vs 2-factor fit ------------
    s1 = np.empty(args.reps_factor)
    s2 = np.empty(args.reps_factor)
    t0 = time.time()
    for r in range(args.reps_factor):
        seed = args.seed_factor + 1000 + r
        fams = simulate_families_multi(FAM, H2_FAC, rg_two_factor(), None,
                                       args.n_fam, args.prev, seed)
        gc = fit_genetic_correlation(fams, n_iter=args.n_iter,
                                     burn_in=args.burn_in,
                                     seed=seed + 100_000,
                                     sampling="population")
        s1[r] = fit_genetic_factor(gc, n_factors=1).srmr
        s2[r] = fit_genetic_factor(gc, n_factors=2).srmr
        if (r + 1) % 5 == 0 or r + 1 == args.reps_factor:
            print(f"  misspecification replicate {r + 1}/{args.reps_factor}",
                  flush=True)
    print("  1-factor srmr=%.3f ± %.3f   2-factor srmr=%.3f ± %.3f   [%.0fs]"
          % (s1.mean(), s1.std(ddof=1), s2.mean(), s2.std(ddof=1), time.time() - t0))
    print("  (compare: srmr under the true 1-factor model was %.3f)" % srmr_1f.mean())
    rows.append(dict(panel="misspec", trait="1-factor fit",
                     srmr=s1.mean(), srmr_sd=s1.std(ddof=1),
                     n_fam=args.n_fam, reps=args.reps_factor))
    rows.append(dict(panel="misspec", trait="2-factor fit",
                     srmr=s2.mean(), srmr_sd=s2.std(ddof=1),
                     n_fam=args.n_fam, reps=args.reps_factor))

    write_csv(rows)
    print("\nwrote bench_genetic_correlation.csv")


def write_csv(rows):
    fields = ["panel", "rg", "n_fam", "bias", "sd", "reps",
              "trait", "true", "fitted", "srmr", "srmr_sd", "prop"]  # panel (c)
    return write_rows(os.path.join(HERE, "bench_genetic_correlation.csv"), rows, fields)


if __name__ == "__main__":
    main()
