"""Multi-component variance-component inference: does `fit_variance_components`
recover additive `A` and common-environment `C`, and how precisely?

`fit_variance_components` generalises `fit_heritability` to a **multiple**
Haseman-Elston regression, fitting `A` and `C` together. As for the single-
component fit, the properties that matter are **bias** and **sampling variability
across datasets**, so this benchmark fits many independent simulated cohorts (each
a fresh draw) and uses their spread as an empirical estimate of sampling
variability. It reports:

  (a) **A+C recovery** — mean(fitted) - true and across-replicate SD for both
      components, at a few (a2, c2) settings;
  (b) **boundary behaviour** — fitting `A,C` on purely additive data (true c2 =
      0): because estimates are constrained nonnegative, report the null estimate's
      mean and SD rather than mislabeling it as a hypothesis-test false-positive rate;
  (c) **precision vs #families** — the across-replicate SD of C as N grows.

Ground truth is set by the simulator (liabilities drawn from
`a2 A + c2 C + e2 I`, thresholded). Families are unascertained population
draws, so this characterises the estimator only under its supported
`sampling="population"` contract.

    python benchmarks/bench_variance_components.py
    python benchmarks/bench_variance_components.py --reps 40 --n-fam 4000
Writes bench_variance_components.csv (+ .png if matplotlib is present).
"""

import os
import csv
import time
import argparse

import numpy as np
from scipy.stats import chi2

from _common import get_plt, simulate_families_components
from ltpred.fit import fit_variance_components

HERE = os.path.dirname(os.path.abspath(__file__))

# a full-sib-rich structure so C (identified from the full-sib excess) is powered
STRUCT = ["m", "f", "s1", "s2", "s3", "s4"]


def _sd_ci(sd, reps, alpha=0.05):
    """Normal-theory confidence interval for an across-replicate SD."""
    df = reps - 1
    return (sd * np.sqrt(df / chi2.ppf(1.0 - alpha / 2.0, df)),
            sd * np.sqrt(df / chi2.ppf(alpha / 2.0, df)))


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
    panel_a = []
    for a2, c2 in settings:
        t0 = time.time()
        fa, fc = fit_replicates(STRUCT, a2, c2, args.n_fam, args.prev, args.reps,
                                args.seed, args.n_iter, args.burn_in)
        rec = dict(a2=a2, c2=c2, A_bias=float(fa.mean() - a2), A_sd=float(fa.std(ddof=1)),
                   C_bias=float(fc.mean() - c2), C_sd=float(fc.std(ddof=1)))
        panel_a.append(rec)
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
    panel_c = []
    for n in args.sizes:
        fa, fc = fit_replicates(STRUCT, 0.4, 0.2, n, args.prev, args.reps_scaling,
                                args.seed + 1000, args.n_iter, args.burn_in)
        rec = dict(n_fam=n, C_sd=float(fc.std(ddof=1)), A_sd=float(fa.std(ddof=1)),
                   reps=args.reps_scaling)
        panel_c.append(rec)
        rows.append(dict(panel="scaling", a2=0.4, c2=0.2, A_bias=float(fa.mean() - 0.4),
                         A_sd=rec["A_sd"], C_bias=float(fc.mean() - 0.2), C_sd=rec["C_sd"],
                         n_fam=n, reps=args.reps_scaling))
        print(f"  n_fam={n:5d} | C SD={rec['C_sd']:.3f}  A SD={rec['A_sd']:.3f}")

    write_csv(rows)
    plot(panel_a, fp, panel_c)
    print("\nwrote bench_variance_components.csv and bench_variance_components.png")


def write_csv(rows):
    fields = ["panel", "a2", "c2", "A_bias", "A_sd", "C_bias", "C_sd", "n_fam", "reps"]
    path = os.path.join(HERE, "bench_variance_components.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows([{k: r.get(k, "") for k in fields} for r in rows])


def plot(panel_a, fp, panel_c):
    plt = get_plt()
    if plt is None:
        return
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    # (a) fitted vs true for A and C
    idx = np.arange(len(panel_a))
    labels = [f"a{r['a2']}/c{r['c2']}" for r in panel_a]
    ax[0].errorbar(idx - 0.08, [r["a2"] + r["A_bias"] for r in panel_a],
                   yerr=[r["A_sd"] for r in panel_a], fmt="o", capsize=3, label="A fitted")
    ax[0].plot(idx - 0.08, [r["a2"] for r in panel_a], "k_", ms=14, label="A true")
    ax[0].errorbar(idx + 0.08, [r["c2"] + r["C_bias"] for r in panel_a],
                   yerr=[r["C_sd"] for r in panel_a], fmt="s", capsize=3, label="C fitted")
    ax[0].plot(idx + 0.08, [r["c2"] for r in panel_a], "r_", ms=14, label="C true")
    ax[0].set_xticks(idx); ax[0].set_xticklabels(labels, fontsize=8)
    ax[0].set_ylabel("proportion of variance")
    ax[0].set_title("(a) A+C recovery")
    ax[0].legend(fontsize=8)
    # (b) nonnegative boundary estimate (not a hypothesis-test rejection rate)
    ax[1].bar(["true c2=0"], [fp["C_mean"]], yerr=[fp["C_sd"]], capsize=5,
              color="tab:red", alpha=0.7)
    ax[1].axhline(0, color="k", lw=1)
    ax[1].set_ylabel("fitted C on additive-only data")
    ax[1].set_title("(b) null boundary estimate (not false-positive rate)")
    # (c) SD of C vs N
    n = np.array([r["n_fam"] for r in panel_c], float)
    csd = np.array([r["C_sd"] for r in panel_c])
    ci = np.array([_sd_ci(r["C_sd"], r["reps"]) for r in panel_c])
    ax[2].errorbar(n, csd, yerr=np.vstack([csd - ci[:, 0], ci[:, 1] - csd]),
                   fmt="-o", capsize=3, label="across-replicate SD of C (95% CI)")
    ref = csd[0] * np.sqrt(n[0] / n)
    ax[2].plot(n, ref, "k:", lw=1, label="∝ 1/√N")
    ax[2].set_xscale("log"); ax[2].set_yscale("log")
    ax[2].set_xlabel("number of families")
    ax[2].set_ylabel("SD of fitted C")
    ax[2].set_title("(c) precision vs #families")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "bench_variance_components.png"), dpi=130)


if __name__ == "__main__":
    main()
