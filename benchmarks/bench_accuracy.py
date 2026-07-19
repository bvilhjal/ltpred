"""Accuracy of classic LT-FH genetic liability: Gibbs vs Pearson-Aitken.

Inspired by the LT-FH++ and PA-FGRS papers, which score a genetic-liability
estimator by how well it recovers the *true* genetic value. For a grid of
heritability x prevalence x family structure this simulates families under the
liability-threshold model (so the true genetic liability is known), estimates it
with both back-ends, and reports:

  * corr(estimate, true g)          -- the headline accuracy
  * slope of true_g ~ estimate      -- calibration (1.0 = well-calibrated posterior mean)
  * RMSE(estimate, true g)
  * squared-correlation effective-N proxy =
    (corr / corr_status)^2 vs the raw case/control label

The two methods should track each other closely (PA is a fast deterministic
approximation to the Gibbs posterior mean). Absolute accuracy rises with h2,
prevalence, and informative relatives; the *relative* squared-correlation
effective-N proxy over a case/control label is largest at low prevalence.

Each cell is replicated on ``--reps`` independent seeds (default 5); replicate
``r`` uses ``--seed + r`` for both the cohort simulation and the Gibbs sampler,
so the historical single-seed configuration is replicate 0. Reported values are
across-seed means with standard errors (sd/sqrt(reps)). Per-cell settings are
unchanged (1,500 families, 25,000 Gibbs draws), so the means are directly
comparable to the former single-seed numbers.

    python benchmarks/bench_accuracy.py                 # default grid, 5 seeds
    python benchmarks/bench_accuracy.py --n-fam 3000 --reps 3
Writes bench_accuracy.csv (+ .png if matplotlib is present). Metric columns are
across-seed means with matching ``se_`` columns; ``t_gibbs``/``t_pa`` are mean
seconds per replicate.
"""

import os
import csv
import time
import argparse

import numpy as np

from _common import simulate_families, estimate, get_plt

HERE = os.path.dirname(os.path.abspath(__file__))

STRUCTURES = {
    "parents": ["m", "f"],
    "parents+sibs": ["m", "f", "s1", "s2"],
    "extended": ["m", "f", "s1", "mgm", "mgf", "pgm", "pgf"],
}


def _metrics(est, true_g, status):
    corr = np.corrcoef(est, true_g)[0, 1]
    slope = np.polyfit(est, true_g, 1)[0]
    rmse = float(np.sqrt(np.mean((est - true_g) ** 2)))
    corr_status = np.corrcoef(status.astype(float), true_g)[0, 1]
    eff_n_proxy = (corr / corr_status) ** 2 if corr_status > 0 else np.nan
    return corr, slope, rmse, eff_n_proxy


METRICS = ("corr_gibbs", "corr_pa", "slope_gibbs", "slope_pa",
           "rmse_gibbs", "rmse_pa", "eff_n_proxy_gibbs", "eff_n_proxy_pa",
           "gibbs_pa_corr")


def _mean_se(values):
    """Across-seed mean and standard error (sd/sqrt(R)) of one metric."""
    v = np.asarray(values, dtype=float)
    se = v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else np.nan
    return float(v.mean()), float(se)


def run(n_fam, h2s, prevs, n_sim, seed, reps):
    cells = [(sname, h2, prev) for sname in STRUCTURES for h2 in h2s
             for prev in prevs]
    acc = {cell: {k: [] for k in METRICS} for cell in cells}
    timings = {cell: [] for cell in cells}
    for rep in range(reps):
        print(f"\nreplicate {rep + 1}/{reps} (seed {seed + rep})")
        for sname, fam_vec in STRUCTURES.items():
            for h2 in h2s:
                for prev in prevs:
                    sim = simulate_families(fam_vec, h2, prev, n_fam, seed + rep)
                    true_g, status = sim.genetic, sim.status["o"]
                    gibbs, t_g = estimate(sim.families, h2, "gibbs", n_sim=n_sim,
                                          seed=seed + rep)
                    pa, t_p = estimate(sim.families, h2, "pearson-aitken")
                    cg = _metrics(gibbs, true_g, status)
                    cp = _metrics(pa, true_g, status)
                    vals = dict(corr_gibbs=cg[0], corr_pa=cp[0],
                                slope_gibbs=cg[1], slope_pa=cp[1],
                                rmse_gibbs=cg[2], rmse_pa=cp[2],
                                eff_n_proxy_gibbs=cg[3],
                                eff_n_proxy_pa=cp[3],
                                gibbs_pa_corr=np.corrcoef(gibbs, pa)[0, 1])
                    for k, v in vals.items():
                        acc[(sname, h2, prev)][k].append(v)
                    timings[(sname, h2, prev)].append((t_g, t_p))
                    print(f"  {sname:14s} h2={h2:.1f} K={prev:.2f} | "
                          f"corr PA={cp[0]:.3f} Gibbs={cg[0]:.3f} | "
                          f"PA eff-N proxy={cp[3]:.2f}x | "
                          f"agree={vals['gibbs_pa_corr']:.4f} | "
                          f"t: {t_g:.2f}s / {t_p:.3f}s")
    rows = []
    print("\nacross-seed means ± SE (sd/sqrt(reps)):")
    for cell in cells:
        sname, h2, prev = cell
        row = dict(structure=sname, h2=h2, prevalence=prev, reps=reps)
        for k in METRICS:
            row[k], row[f"se_{k}"] = _mean_se(acc[cell][k])
        t_pair = np.asarray(timings[cell])
        row["t_gibbs"] = float(t_pair[:, 0].mean())
        row["t_pa"] = float(t_pair[:, 1].mean())
        rows.append(row)
        print(f"  {sname:14s} h2={h2:.1f} K={prev:.2f} | "
              f"corr PA={row['corr_pa']:.3f}±{row['se_corr_pa']:.3f} "
              f"Gibbs={row['corr_gibbs']:.3f}±{row['se_corr_gibbs']:.3f} | "
              f"PA eff-N proxy={row['eff_n_proxy_pa']:.2f}"
              f"±{row['se_eff_n_proxy_pa']:.2f}x | "
              f"agree={row['gibbs_pa_corr']:.4f}±{row['se_gibbs_pa_corr']:.4f}")
    return rows


def write_csv(rows):
    path = os.path.join(HERE, "bench_accuracy.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return path


def plot(rows):
    plt = get_plt()
    if plt is None:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    # (a) Gibbs vs PA accuracy agreement (across-seed cell means)
    cg = [r["corr_gibbs"] for r in rows]
    cp = [r["corr_pa"] for r in rows]
    axes[0].scatter(cg, cp, s=20, alpha=0.8)
    lim = [min(cg + cp) - 0.02, max(cg + cp) + 0.02]
    axes[0].plot(lim, lim, "k--", lw=1)
    axes[0].set_xlabel("corr(Gibbs, true g)")
    axes[0].set_ylabel("corr(PA estimate, true g)")
    axes[0].set_title("(a) accuracy: Gibbs vs PA")
    # (b) accuracy vs prevalence, by structure (h2=0.5); bars = across-seed SE
    for sname in STRUCTURES:
        sub = [r for r in rows if r["structure"] == sname and r["h2"] == 0.5]
        sub.sort(key=lambda r: r["prevalence"])
        axes[1].errorbar([r["prevalence"] for r in sub],
                         [r["corr_pa"] for r in sub],
                         yerr=[r["se_corr_pa"] for r in sub],
                         fmt="-o", capsize=3, label=sname)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("prevalence")
    axes[1].set_ylabel("corr(PA estimate, true g)")
    axes[1].set_title("(b) PA accuracy vs prevalence (h2=0.5)")
    axes[1].legend(fontsize=8)
    # (c) squared-correlation effective-N proxy, by h2 (extended family)
    for h2 in sorted({r["h2"] for r in rows}):
        sub = [r for r in rows if r["structure"] == "extended" and r["h2"] == h2]
        sub.sort(key=lambda r: r["prevalence"])
        axes[2].errorbar([r["prevalence"] for r in sub],
                         [r["eff_n_proxy_pa"] for r in sub],
                         yerr=[r["se_eff_n_proxy_pa"] for r in sub],
                         fmt="-o", capsize=3, label=f"h2={h2}")
    axes[2].axhline(1.0, color="k", ls=":", lw=1)
    axes[2].set_xscale("log")
    axes[2].set_xlabel("prevalence")
    axes[2].set_ylabel("squared-correlation eff-N proxy vs case/control")
    axes[2].set_title("(c) PA effective-N proxy (extended family)")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "bench_accuracy.png"), dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-fam", type=int, default=1500)
    ap.add_argument("--h2", type=float, nargs="+", default=[0.2, 0.5, 0.8])
    ap.add_argument("--prev", type=float, nargs="+", default=[0.01, 0.05, 0.20])
    ap.add_argument("--n-sim", type=int, default=25_000)
    ap.add_argument("--reps", type=int, default=5,
                    help="independent seeds per grid cell (seed, seed+1, ...)")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    if args.reps < 1:
        ap.error("--reps must be at least 1")

    t_start = time.perf_counter()
    # Compile both inference paths before recording any timing cell. Without
    # this, the first grid cell pays JIT cost and is not comparable to the rest.
    warm = simulate_families(["m", "f", "s1"], args.h2[0], args.prev[0], 64,
                             args.seed + 10_000)
    estimate(warm.families, args.h2[0], "gibbs", n_sim=1_000,
             seed=args.seed + 10_000)
    estimate(warm.families, args.h2[0], "pearson-aitken")

    rows = run(args.n_fam, args.h2, args.prev, args.n_sim, args.seed, args.reps)
    path = write_csv(rows)
    plot(rows)
    print(f"\nwrote {os.path.basename(path)} and bench_accuracy.png")
    print(f"total runtime: {(time.perf_counter() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
