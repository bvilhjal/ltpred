"""Variance-component inference: does `fit_heritability` recover h2, how precisely?

The important properties of a heritability estimator are **bias** and **sampling
variability across datasets** — not runtime. This benchmark measures them by
fitting many independent simulated cohorts (each a fresh draw), so the spread of
the fitted values estimates sampling variability. It reports:

  (a) **bias & precision vs true h2** — mean(fitted) - h2_true and the
      across-replicate SD, for h2 from 0.2 to 0.8;
  (b) **precision vs #families** — the across-replicate SD as N grows (the
      1/sqrt(N) law), and how family structure (informativeness) changes it;
  (c) **SE calibration** — the reported within-dataset `h2_se` vs the true
      across-dataset SD. The reported `se` is a *Monte-Carlo* error of the fit on
      one dataset, NOT the sampling SD, so it under-states the real uncertainty;
      this panel quantifies the gap (bootstrap families for a genuine CI).

Ground truth is known (the simulator's `h2`) and families are unascertained
population draws, so this characterises the estimator only under its supported
`sampling="population"` contract, not under case/control ascertainment.

    python benchmarks/bench_fit_heritability.py
    python benchmarks/bench_fit_heritability.py --reps 40 --n-fam 5000
Writes bench_fit_heritability.csv.
"""

import os
import time
import argparse

import numpy as np

from _common import write_rows
from ltpred import simulate_under_LTM_single, fit_heritability

HERE = os.path.dirname(os.path.abspath(__file__))

STRUCTURES = {
    "parents": ["m", "f"],
    "parents+2 sibs": ["m", "f", "s1", "s2"],
    "extended": ["m", "f", "s1", "s2", "mgm", "mgf", "pgm", "pgf"],
}


def fit_replicates(fam_vec, h2_true, n_fam, prev, reps, seed0, n_iter, burn_in,
                   inner_sweeps):
    """Fit `reps` independent simulated cohorts; return (fitted[], reported_se[])."""
    fitted = np.empty(reps)
    rep_se = np.empty(reps)
    for r in range(reps):
        sim = simulate_under_LTM_single(fam_vec=fam_vec, h2=h2_true, n_sim=n_fam,
                                        pop_prev=prev, seed=seed0 + r)
        res = fit_heritability(sim.families, sampling="population",
                               n_iter=n_iter, burn_in=burn_in,
                               inner_sweeps=inner_sweeps,
                               seed=seed0 + 100_000 + r)
        fitted[r] = res.h2
        rep_se[r] = res.h2_se
    return fitted, rep_se


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h2", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.8])
    ap.add_argument("--n-fam", type=int, default=3000)
    ap.add_argument("--sizes", type=int, nargs="+", default=[1000, 2000, 4000, 8000])
    ap.add_argument("--prev", type=float, default=0.1)
    ap.add_argument("--reps", type=int, default=25)
    ap.add_argument("--reps-scaling", type=int, default=15)
    ap.add_argument("--n-iter", type=int, default=500)
    ap.add_argument("--burn-in", type=int, default=150)
    ap.add_argument("--inner-sweeps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=100)
    args = ap.parse_args()

    fam_a = STRUCTURES["parents+2 sibs"]
    simulate_under_LTM_single(fam_vec=fam_a, h2=0.5, n_sim=50, pop_prev=args.prev, seed=0)
    fit_heritability(simulate_under_LTM_single(fam_vec=fam_a, h2=0.5, n_sim=50,
                     pop_prev=args.prev, seed=0).families, sampling="population",
                     n_iter=20, burn_in=5)  # warm JIT

    rows = []

    # (a) bias & precision vs true h2  ------------------------------------------
    print("== (a) bias & precision vs true h2 (n_fam=%d, %s) ==" % (args.n_fam, "parents+2 sibs"))
    panel_a = []
    for h2 in args.h2:
        t0 = time.time()
        fitted, rep_se = fit_replicates(fam_a, h2, args.n_fam, args.prev, args.reps,
                                        args.seed, args.n_iter, args.burn_in, args.inner_sweeps)
        bias = float(fitted.mean() - h2)
        sd = float(fitted.std(ddof=1))
        se_rep = float(np.median(rep_se))
        panel_a.append(dict(h2=h2, bias=bias, sd=sd, reported_se=se_rep))
        rows.append(dict(panel="bias_precision", h2=h2, n_fam=args.n_fam,
                         structure="parents+2 sibs", bias=bias, sd=sd,
                         reported_se=se_rep, reps=args.reps))
        print(f"  h2={h2:.1f} | fitted={fitted.mean():.3f}  bias={bias:+.3f}  "
              f"SD={sd:.3f} | reported se={se_rep:.4f}  (SD/se={sd/se_rep:.0f}x)  "
              f"[{time.time()-t0:.0f}s]")

    # (b) precision vs #families (and structure)  -------------------------------
    print("== (b) precision (SD) vs #families (h2=0.5) ==")
    panel_b = []
    for n in args.sizes:
        fitted, _ = fit_replicates(fam_a, 0.5, n, args.prev, args.reps_scaling,
                                   args.seed + 1000, args.n_iter, args.burn_in, args.inner_sweeps)
        sd = float(fitted.std(ddof=1))
        panel_b.append(dict(n_fam=n, sd=sd, reps=args.reps_scaling))
        rows.append(dict(panel="scaling", h2=0.5, n_fam=n, structure="parents+2 sibs",
                         bias=float(fitted.mean() - 0.5), sd=sd, reported_se=np.nan,
                         reps=args.reps_scaling))
        print(f"  n_fam={n:5d} | SD={sd:.3f}")
    print("== structure at n_fam=%d, h2=0.5 ==" % args.n_fam)
    panel_s = []
    for name, fv in STRUCTURES.items():
        fitted, _ = fit_replicates(fv, 0.5, args.n_fam, args.prev, args.reps_scaling,
                                   args.seed + 2000, args.n_iter, args.burn_in, args.inner_sweeps)
        sd = float(fitted.std(ddof=1))
        panel_s.append(dict(structure=name, sd=sd, n_rel=len(fv)))
        rows.append(dict(panel="structure", h2=0.5, n_fam=args.n_fam, structure=name,
                         bias=float(fitted.mean() - 0.5), sd=sd, reported_se=np.nan,
                         reps=args.reps_scaling))
        print(f"  {name:16s} ({len(fv)} rel) | SD={sd:.3f}")

    write_csv(rows)
    print("\nwrote bench_fit_heritability.csv")


def write_csv(rows):
    fields = ["panel", "h2", "n_fam", "structure", "bias", "sd", "reported_se", "reps"]
    return write_rows(os.path.join(HERE, "bench_fit_heritability.csv"), rows, fields)


if __name__ == "__main__":
    main()
