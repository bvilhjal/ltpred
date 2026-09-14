"""Pairwise composite-likelihood recovery: does `fit_pairwise` recover `A`, `C`
and `M` without bias, and are its family-level sandwich SEs calibrated?

`fit_pairwise` is a composite likelihood over relative pairs, not the full
family likelihood, so the properties that matter are **bias**, **SE
calibration** against across-dataset spread, and **interval coverage**. This
benchmark fits many independent simulated cohorts at several cohort sizes and
reports:

  (a) **bias vs #families** — mean(fitted) - true for each component. Replicates
      are allocated proportional to `1/N`, so SE(bias) is constant across sizes
      by construction and the sizes are directly comparable. A bias that is flat
      in `N` is asymptotic; one that decays as `1/N` is finite-sample.
  (b) **SE calibration and coverage** — mean reported SE over across-replicate
      SD, and the realised coverage of the nominal 95% normal interval;
  (c) **boundary pinning** — the share of replicates with a component pinned at
      the non-negativity boundary, where the reported SE is unavailable by
      design and a normal interval is inappropriate.

Ground truth is set by the simulator (liabilities drawn from
`a2 A + c2 C + m2 M + e2 I`, thresholded at a common threshold). Families are
unascertained population draws, so this characterises the estimator only under
its supported `sampling="population"` contract with a known threshold; IPW
weighting, personalised thresholds and ascertained designs are out of scope.

There is a known intrinsic finite-sample bias here, and it is negative: in the
one-component, one-pair-per-family case at threshold zero the estimator is
exactly `2 sin(pi (p_hat - 1/2))` in the concordant fraction, which is concave
for `A > 0`, giving `N x bias -> -pi^2 A p(1-p) / 2`. At the sizes below that is
order `1e-4` and far too small to see.

    python benchmarks/bench_pairwise_recovery.py
    python benchmarks/bench_pairwise_recovery.py --budget 8800000 \
        --sizes 1500 3000 6000 12000 24000 --jobs 6
Writes bench_pairwise_recovery.csv (+ .png if matplotlib is present).
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from _common import get_plt, simulate_families_components, write_rows
from ltpred.pairwise import fit_pairwise

HERE = os.path.dirname(os.path.abspath(__file__))

# Three offspring give the full-sib pairs that identify C; the mated pair
# identifies M; parent-offspring pairs carry A with C = M = 0.
STRUCT = ["o", "m", "f", "s1", "s2"]
COMPONENTS = ("A", "C", "M")
TRUTH = {"A": 0.30, "C": 0.15, "M": 0.10}


def _one(task):
    """Fit one replicate. Deterministic in its seed, so `--jobs` cannot change
    the result; the BLAS pool is pinned to avoid oversubscribing the workers."""
    n_fam, prev, seed = task
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        fams = simulate_families_components(STRUCT, dict(TRUTH), n_fam, prev, seed)
        res = fit_pairwise(fams, components=COMPONENTS, sampling="population")
    return ({c: float(res.components[c]) for c in COMPONENTS},
            {c: float(res.se[c]) for c in COMPONENTS})


def fit_replicates(n_fam, prev, reps, seed0, jobs):
    tasks = [(n_fam, prev, seed0 + r) for r in range(reps)]
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            out = list(pool.map(_one, tasks, chunksize=8))
    else:
        out = [_one(t) for t in tasks]
    est = {c: np.array([o[0][c] for o in out]) for c in COMPONENTS}
    se = {c: np.array([o[1][c] for o in out]) for c in COMPONENTS}
    return est, se


def summarise(n_fam, reps, est, se):
    rows = []
    pinned = np.zeros(reps, dtype=bool)
    for c in COMPONENTS:
        pinned |= est[c] < 1e-5
    for c in COMPONENTS:
        x, s = est[c], se[c]
        good = np.isfinite(s)
        cover = (float(np.mean(np.abs(x[good] - TRUTH[c]) <= 1.96 * s[good]))
                 if good.any() else float("nan"))
        sd = float(x.std(ddof=1))
        rows.append(dict(n_fam=n_fam, reps=reps, comp=c, truth=TRUTH[c],
                         bias=float(x.mean() - TRUTH[c]),
                         se_bias=sd / np.sqrt(reps), sd=sd,
                         mean_se=float(np.nanmean(s)),
                         se_over_sd=float(np.nanmean(s)) / sd,
                         coverage=cover,
                         pinned_frac=float(np.mean(est[c] < 1e-5))))
    return rows, float(np.mean(pinned))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[1500, 3000, 6000, 12000])
    ap.add_argument("--budget", type=int, default=300_000,
                    help="families per size; reps = budget // n_fam, so "
                         "SE(bias) is constant across sizes")
    ap.add_argument("--min-reps", type=int, default=25)
    ap.add_argument("--prev", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()

    print(f"== fit_pairwise recovery ({'+'.join(STRUCT)}, prevalence "
          f"{args.prev}, truth A={TRUTH['A']} C={TRUTH['C']} M={TRUTH['M']}) ==")
    rows, panel = [], []
    for i, n in enumerate(args.sizes):
        reps = max(args.min_reps, args.budget // n)
        t0 = time.time()
        est, se = fit_replicates(n, args.prev, reps, args.seed + 1000 * i,
                                 args.jobs)
        got, pinned_any = summarise(n, reps, est, se)
        rows.extend(got)
        panel.append(dict(n_fam=n, reps=reps, pinned_any=pinned_any,
                          by_comp={r["comp"]: r for r in got}))
        line = "  ".join(
            f"{r['comp']}={r['bias']:+.4f}+-{r['se_bias']:.4f}"
            f"(SE/SD {r['se_over_sd']:.2f}, cov {r['coverage']:.3f})"
            for r in got)
        print(f"  n_fam={n:6d} reps={reps:5d} | {line} | pinned "
              f"{pinned_any:.1%}  [{time.time()-t0:.0f}s]")

    write_csv(rows)
    plot(panel)
    print("\nwrote bench_pairwise_recovery.csv")


def write_csv(rows):
    fields = ["n_fam", "reps", "comp", "truth", "bias", "se_bias", "sd",
              "mean_se", "se_over_sd", "coverage", "pinned_frac"]
    return write_rows(os.path.join(HERE, "bench_pairwise_recovery.csv"), rows, fields)


def plot(panel):
    plt = get_plt()
    if plt is None:
        return
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    n = np.array([p["n_fam"] for p in panel], float)
    for c in COMPONENTS:
        bias = np.array([p["by_comp"][c]["bias"] for p in panel])
        err = np.array([1.96 * p["by_comp"][c]["se_bias"] for p in panel])
        ax[0].errorbar(n, bias, yerr=err, fmt="-o", capsize=3, label=c)
    ax[0].axhline(0.0, color="k", lw=1)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("number of families")
    ax[0].set_ylabel("mean(fitted) - true")
    ax[0].set_title("(a) bias vs #families (95% CI)")
    ax[0].legend(fontsize=8)
    for c in COMPONENTS:
        ax[1].plot(n, [p["by_comp"][c]["se_over_sd"] for p in panel], "-o",
                   label=f"{c} SE/SD")
        ax[1].plot(n, [p["by_comp"][c]["coverage"] for p in panel], "--s",
                   label=f"{c} coverage")
    ax[1].axhline(1.0, color="k", lw=1)
    ax[1].axhline(0.95, color="r", lw=1, ls=":")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("number of families")
    ax[1].set_title("(b) SE calibration and 95% coverage")
    ax[1].legend(fontsize=7, ncol=2)
    ax[2].plot(n, [100.0 * p["pinned_any"] for p in panel], "-o",
               color="tab:red")
    ax[2].set_xscale("log")
    ax[2].set_xlabel("number of families")
    ax[2].set_ylabel("% of replicates with a pinned component")
    ax[2].set_title("(c) non-negativity boundary")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "bench_pairwise_recovery.png"), dpi=130)


if __name__ == "__main__":
    main()
