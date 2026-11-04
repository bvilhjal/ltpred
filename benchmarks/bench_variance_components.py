"""Measure A+C recovery, boundary estimates and precision.

Independent population cohorts from `a2 A + c2 C + e2 I` evaluate component
bias and across-replicate SD, the nonnegative C estimate at true c2=0 (not a
rejection rate), and precision as family count grows. Writes
`bench_variance_components.csv`.
"""

import os
import time
import argparse

import numpy as np

from _common import (simulate_families_components,
                     write_rows)
from ltpred.fit import fit_variance_components

HERE = os.path.dirname(os.path.abspath(__file__))

# a full-sib-rich structure so C (identified from the full-sib excess) is powered
STRUCT = ["m", "f", "s1", "s2", "s3", "s4"]


def fit_replicates(fam_vec, a2, c2, n_fam, prev, reps, seed0, n_iter, burn_in):
    """Fit `reps` independent A+C cohorts; return fitted A[] and C[]."""
    fa = np.empty(reps)
    fc = np.empty(reps)
    for r in range(reps):
        fams = simulate_families_components(fam_vec, {"A": a2, "C": c2}, n_fam,
                                            prev, seed0 + r)
        res = fit_variance_components(fams, ("A", "C"), n_iter=n_iter,
                                      burn_in=burn_in,
                                      sampling="population",
                                      seed=seed0 + 100_000 + r)
        fa[r], fc[r] = res.components["A"], res.components["C"]
    return fa, fc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-fam", type=int, default=3000)
    ap.add_argument("--sizes", type=int, nargs="+", default=[1000, 2000, 4000, 8000])
    ap.add_argument("--prev", type=float, default=0.1)
    ap.add_argument("--reps", type=int, default=25)
    ap.add_argument("--reps-scaling", type=int, default=15)
    ap.add_argument("--n-iter", type=int, default=800)
    ap.add_argument("--burn-in", type=int, default=250)
    ap.add_argument("--seed", type=int, default=100)
    args = ap.parse_args()

    # warm the JIT
    fit_variance_components(
        simulate_families_components(STRUCT, {"A": 0.4, "C": 0.2}, 60, args.prev, 0),
        ("A", "C"), sampling="population", n_iter=20, burn_in=5)

    rows = []

    # (a) A+C recovery ----------------------------------------------------------
    print("== (a) A+C recovery (n_fam=%d, %s) ==" % (args.n_fam, "+".join(STRUCT)))
    settings = [(0.4, 0.2), (0.5, 0.1), (0.3, 0.3), (0.6, 0.0)]
    for a2, c2 in settings:
        t0 = time.time()
        fa, fc = fit_replicates(STRUCT, a2, c2, args.n_fam, args.prev, args.reps,
                                args.seed, args.n_iter, args.burn_in)
        rec = dict(a2=a2, c2=c2, A_bias=float(fa.mean() - a2), A_sd=float(fa.std(ddof=1)),
                   C_bias=float(fc.mean() - c2), C_sd=float(fc.std(ddof=1)))
        rows.append(dict(panel="recovery", **rec, n_fam=args.n_fam, reps=args.reps))
        print(f"  a2={a2:.1f} c2={c2:.1f} | A={fa.mean():.3f}({rec['A_bias']:+.3f}) "
              f"SD={rec['A_sd']:.3f} | C={fc.mean():.3f}({rec['C_bias']:+.3f}) "
              f"SD={rec['C_sd']:.3f}  [{time.time()-t0:.0f}s]")

    # (b) boundary behaviour: C on additive-only data --------------------------
    print("== (b) boundary estimate for C (true c2=0; not a rejection rate) ==")
    fa, fc = fit_replicates(STRUCT, 0.5, 0.0, args.n_fam, args.prev, args.reps,
                            args.seed + 500, args.n_iter, args.burn_in)
    fp = dict(a2=0.5, c2=0.0, A_bias=float(fa.mean() - 0.5), A_sd=float(fa.std(ddof=1)),
              C_bias=float(fc.mean()), C_sd=float(fc.std(ddof=1)), C_mean=float(fc.mean()))
    rows.append(dict(panel="boundary_null", **{k: fp[k] for k in
                ("a2", "c2", "A_bias", "A_sd", "C_bias", "C_sd")},
                n_fam=args.n_fam, reps=args.reps))
    print(f"  A={fa.mean():.3f}  C(boundary)={fc.mean():.3f} +/- {fp['C_sd']:.3f}")

    # (c) precision vs #families (C at a2=0.4, c2=0.2) --------------------------
    print("== (c) precision (SD of C) vs #families ==")
    for n in args.sizes:
        fa, fc = fit_replicates(STRUCT, 0.4, 0.2, n, args.prev, args.reps_scaling,
                                args.seed + 1000, args.n_iter, args.burn_in)
        rec = dict(n_fam=n, C_sd=float(fc.std(ddof=1)), A_sd=float(fa.std(ddof=1)),
                   reps=args.reps_scaling)
        rows.append(dict(panel="scaling", a2=0.4, c2=0.2, A_bias=float(fa.mean() - 0.4),
                         A_sd=rec["A_sd"], C_bias=float(fc.mean() - 0.2), C_sd=rec["C_sd"],
                         n_fam=n, reps=args.reps_scaling))
        print(f"  n_fam={n:5d} | C SD={rec['C_sd']:.3f}  A SD={rec['A_sd']:.3f}")

    write_csv(rows)
    print("\nwrote bench_variance_components.csv")


def write_csv(rows):
    fields = ["panel", "a2", "c2", "A_bias", "A_sd", "C_bias", "C_sd", "n_fam", "reps"]
    return write_rows(os.path.join(HERE, "bench_variance_components.csv"), rows, fields)


if __name__ == "__main__":
    main()
