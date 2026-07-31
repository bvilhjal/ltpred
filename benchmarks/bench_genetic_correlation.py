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
  (b) **precision vs #families** — the across-replicate SD of r_g as N grows.

Ground truth is the simulator's genetic-correlation matrix.

    python benchmarks/bench_genetic_correlation.py
    python benchmarks/bench_genetic_correlation.py --reps 40 --n-fam 4000
Writes bench_genetic_correlation.csv (+ .png if matplotlib is present).
"""

import os
import csv
import sys
import time
import argparse
from pathlib import Path

import numpy as np
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root: `research`

from _common import get_plt, simulate_families_multi
from research.advanced_fitting import fit_genetic_correlation

HERE = os.path.dirname(os.path.abspath(__file__))

FAM = ["m", "f", "s1", "s2"]
H2 = [0.5, 0.4]
RP_OFFDIAG = 0.2                       # phenotypic (full-liability) correlation


def _sd_ci(sd, reps, alpha=0.05):
    """Normal-theory confidence interval for an across-replicate SD."""
    df = reps - 1
    return (sd * np.sqrt(df / chi2.ppf(1.0 - alpha / 2.0, df)),
            sd * np.sqrt(df / chi2.ppf(alpha / 2.0, df)))


def fit_replicates(rg_val, n_fam, prev, reps, seed0, n_iter, burn_in):
    fitted = np.empty(reps)
    for r in range(reps):
        fams = simulate_families_multi(FAM, H2, rg_val, RP_OFFDIAG, n_fam,
                                       (prev, prev), seed0 + r)
        res = fit_genetic_correlation(fams, n_iter=n_iter, burn_in=burn_in,
                                      seed=seed0 + 100_000 + r)
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
    args = ap.parse_args()

    fit_genetic_correlation(simulate_families_multi(FAM, H2, 0.5, 0.2, 60,
                                                    (0.1, 0.1), 0),
                            n_iter=20, burn_in=5)   # warm JIT

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

    write_csv(rows)
    plot(panel_a, panel_b)
    print("\nwrote bench_genetic_correlation.csv and bench_genetic_correlation.png")


def write_csv(rows):
    fields = ["panel", "rg", "n_fam", "bias", "sd", "reps"]
    path = os.path.join(HERE, "bench_genetic_correlation.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows([{k: r.get(k, "") for k in fields} for r in rows])


def plot(panel_a, panel_b):
    plt = get_plt()
    if plt is None:
        return
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.4))
    rg = [r["rg"] for r in panel_a]
    fitted = [r["rg"] + r["bias"] for r in panel_a]
    sd = [r["sd"] for r in panel_a]
    ax[0].errorbar(rg, fitted, yerr=sd, fmt="o", capsize=4, label="fitted ± SD")
    lim = [min(rg) - 0.1, max(rg) + 0.1]
    ax[0].plot(lim, lim, "k--", lw=1, label="y = x")
    ax[0].set_xlabel("true r_g")
    ax[0].set_ylabel("fitted r_g")
    ax[0].set_title("(a) bias & precision (incl. null)")
    ax[0].legend(fontsize=8)
    n = np.array([r["n_fam"] for r in panel_b], float)
    sdb = np.array([r["sd"] for r in panel_b])
    ci = np.array([_sd_ci(r["sd"], r["reps"]) for r in panel_b])
    ax[1].errorbar(n, sdb, yerr=np.vstack([sdb - ci[:, 0], ci[:, 1] - sdb]),
                   fmt="-o", capsize=3, label="across-replicate SD (95% CI)")
    ref = sdb[0] * np.sqrt(n[0] / n)
    ax[1].plot(n, ref, "k:", lw=1, label="∝ 1/√N")
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("number of families")
    ax[1].set_ylabel("SD of fitted r_g")
    ax[1].set_title("(b) precision vs #families (r_g=0.5)")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "bench_genetic_correlation.png"), dpi=130)


if __name__ == "__main__":
    main()
