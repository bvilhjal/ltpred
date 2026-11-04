"""Compare Gibbs and PA accuracy on classic LT-FH simulations.

Sweep heritability, prevalence and family structure. Each cell uses independent
seeds and reports across-seed means/SEs for correlation with true g,
calibration slope, RMSE, squared-correlation effective-N proxy, PA–Gibbs
agreement and runtime. The effective-N proxy is relative to the case/control
label, not the causal-SNP NCP ratio used by the GWAS benchmark. Writes
`bench_accuracy.csv`.
"""

import os
import time
import argparse

import numpy as np

from _common import estimate, mean_se, simulate_families, write_rows

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
            row[k], row[f"se_{k}"] = mean_se(acc[cell][k])
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
    return write_rows(os.path.join(HERE, "bench_accuracy.csv"), rows)


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
    print(f"\nwrote {os.path.basename(path)}")
    print(f"total runtime: {(time.perf_counter() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
