"""Variance-component inference: does `fit_heritability` recover h2, how precisely?

The important properties of a heritability estimator are **bias** and **sampling
variability across datasets** — not runtime. This benchmark measures them by
fitting many independent simulated cohorts (each a fresh draw), so the spread of
the fitted values *is* the true sampling distribution. It reports:

  (a) **bias & precision vs true h2** — mean(fitted) - h2_true and the
      across-replicate SD, for h2 from 0.2 to 0.8;
  (b) **precision vs #families** — the across-replicate SD as N grows (the
      1/sqrt(N) law), and how family structure (informativeness) changes it;
  (c) **SE calibration** — the reported within-dataset `h2_se` vs the true
      across-dataset SD. The reported `se` is a *Monte-Carlo* error of the fit on
      one dataset, NOT the sampling SD, so it under-states the real uncertainty;
      this panel quantifies the gap (bootstrap families for a genuine CI).

Ground truth is known (the simulator's `h2`), so this is a clean characterisation
of the estimator, not a comparison of methods.

    python benchmarks/bench_fit_heritability.py
    python benchmarks/bench_fit_heritability.py --reps 40 --n-fam 5000
Writes bench_fit_heritability.csv (+ .png if matplotlib is present).
"""

import os
import csv
import time
import argparse

import numpy as np

from _common import get_plt
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
        res = fit_heritability(sim.families, n_iter=n_iter, burn_in=burn_in,
                               inner_sweeps=inner_sweeps, seed=1)
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
                     pop_prev=args.prev, seed=0).families, n_iter=20, burn_in=5)  # warm JIT

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
        panel_b.append(dict(n_fam=n, sd=sd))
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
    plot(panel_a, panel_b, panel_s)
    print("\nwrote bench_fit_heritability.csv and bench_fit_heritability.png")


def write_csv(rows):
    fields = ["panel", "h2", "n_fam", "structure", "bias", "sd", "reported_se", "reps"]
    path = os.path.join(HERE, "bench_fit_heritability.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows([{k: r.get(k, "") for k in fields} for r in rows])


def plot(panel_a, panel_b, panel_s):
    plt = get_plt()
    if plt is None:
        return
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    # (a) fitted vs true, with +/- SD error bars
    h2 = [r["h2"] for r in panel_a]
    fitted = [r["h2"] + r["bias"] for r in panel_a]
    sd = [r["sd"] for r in panel_a]
    ax[0].errorbar(h2, fitted, yerr=sd, fmt="o", capsize=4, label="fitted ± SD")
    lim = [min(h2) - 0.1, max(h2) + 0.1]
    ax[0].plot(lim, lim, "k--", lw=1, label="y = x")
    ax[0].set_xlabel("true h²")
    ax[0].set_ylabel("fitted h²")
    ax[0].set_title("(a) bias & precision")
    ax[0].legend(fontsize=8)
    # (b) SD vs N with 1/sqrt(N) reference
    n = np.array([r["n_fam"] for r in panel_b], float)
    sdb = np.array([r["sd"] for r in panel_b])
    ax[1].plot(n, sdb, "-o", label="across-replicate SD")
    ref = sdb[0] * np.sqrt(n[0] / n)
    ax[1].plot(n, ref, "k:", lw=1, label="∝ 1/√N")
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("number of families")
    ax[1].set_ylabel("SD of fitted h²")
    ax[1].set_title("(b) precision vs #families (h²=0.5)")
    ax[1].legend(fontsize=8)
    # (c) reported se vs true SD (calibration)
    labels = [f"h²={r['h2']}" for r in panel_a]
    xs = np.arange(len(labels))
    ax[2].bar(xs - 0.2, [r["sd"] for r in panel_a], 0.4, label="true SD (across datasets)")
    ax[2].bar(xs + 0.2, [r["reported_se"] for r in panel_a], 0.4, label="reported h2_se (within)")
    ax[2].set_xticks(xs); ax[2].set_xticklabels(labels, fontsize=8)
    ax[2].set_ylabel("uncertainty in h²")
    ax[2].set_title("(c) reported se under-states true SD")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "bench_fit_heritability.png"), dpi=130)


if __name__ == "__main__":
    main()
