"""Stress PA on large, rare and densely affected pedigrees.

Measure PA–Gibbs correlation and each engine's correlation with true g, then
rerun PA under random fold orders and report per-proband spread absolute and
relative to between-proband SD. The ordering spread is not compared with
Gibbs Monte-Carlo noise. Independent seeds produce long-format rows and
across-seed mean/SE summaries. Writes `bench_pa_robustness.csv`.
"""

import os
import csv
import argparse

import numpy as np

from _common import estimate, safe_corr, simulate_families
from ltpred.covariance import construct_covmat_single, correct_positive_definite
from ltpred.thresholds import liability_threshold
from ltpred.pearson_aitken import pa_algorithm

HERE = os.path.dirname(os.path.abspath(__file__))

# increasing pedigrees for the fold-order sweep (and named regimes for (a))
SIZES = {
    "trio (2 rel)": ["m", "f"],
    "parents+2 sibs (4)": ["m", "f", "s1", "s2"],
    "extended (8)": ["m", "f", "s1", "s2", "mgm", "mgf", "pgm", "pgf"],
    "large (11)": ["m", "f", "s1", "s2", "s3", "mgm", "mgf", "pgm", "pgf", "mau1", "pau1"],
}


# --------------------------------------------------------------------------- #
#  (a) PA vs Gibbs on stressful pedigrees                                      #
# --------------------------------------------------------------------------- #
def _dense_families(fam_vec, h2, prev, n_fam, seed, min_affected=3):
    """Simulate, then keep only densely-affected families (>= ``min_affected``
    observed cases among the members) — the high-family-loading regime."""
    keep_g, keep_fams = [], []
    s = 0
    while len(keep_fams) < n_fam:
        sim = simulate_families(fam_vec, h2, prev, 4 * n_fam, seed + s)
        s += 1
        for i, fam in enumerate(sim.families):
            n_aff = sum(1 for m in fam.members if np.isfinite(m.lower) and m.lower > -1e10)
            if n_aff >= min_affected:
                keep_fams.append(fam); keep_g.append(sim.genetic[i])
            if len(keep_fams) >= n_fam:
                break
    return keep_fams, np.array(keep_g)


def regimes(n_fam, seed, n_sim):
    """corr(PA, Gibbs) and accuracy across baseline / large / rare / dense regimes."""
    out = []
    cases = [
        ("baseline", ["m", "f", "s1", "s2"], 0.05, None),
        ("large pedigree", SIZES["large (11)"], 0.05, None),
        ("rare (K=0.005)", ["m", "f", "s1", "s2"], 0.005, None),
        ("dense (>=3 aff)", ["m", "f", "s1", "s2", "s3", "s4"], 0.05, "dense"),
    ]
    for name, fam_vec, prev, mode in cases:
        if mode == "dense":
            fams, g = _dense_families(fam_vec, 0.5, prev, n_fam, seed)
            pa, _ = estimate(fams, 0.5, "pearson-aitken")
            gib, _ = estimate(fams, 0.5, "gibbs", n_sim=n_sim, seed=seed)
        else:
            sim = simulate_families(fam_vec, 0.5, prev, n_fam, seed)
            g = sim.genetic
            pa, _ = estimate(sim.families, 0.5, "pearson-aitken")
            gib, _ = estimate(sim.families, 0.5, "gibbs", n_sim=n_sim, seed=seed)
        row = dict(regime=name, n_rel=len(fam_vec), agree=safe_corr(pa, gib),
                   corr_pa=safe_corr(pa, g), corr_gibbs=safe_corr(gib, g))
        out.append(row)
        print("  %-18s | PA<->Gibbs=%.4f | corr(PA,g)=%.3f corr(Gibbs,g)=%.3f"
              % (name, row["agree"], row["corr_pa"], row["corr_gibbs"]))
    return out


# --------------------------------------------------------------------------- #
#  (b) fold-in ordering sensitivity (target the g node; permute the rest)      #
# --------------------------------------------------------------------------- #
def fold_order(fam_vec, prev, n_fam, n_orders, seed, h2=0.5):
    """Per-proband PA estimate under ``n_orders`` random fold-in orders; returns the
    order-induced SD (absolute and relative to the between-proband SD)."""
    cov_obj = construct_covmat_single(fam_vec=fam_vec, add_ind=True, h2=h2)
    cov, _ = correct_positive_definite(cov_obj.matrix)
    roles = cov_obj.roles
    d = len(roles)
    t = float(liability_threshold(prev))
    rng = np.random.default_rng(seed)
    L = rng.multivariate_normal(np.zeros(d), cov, size=n_fam)      # includes g at col 0
    case = L > t                                                  # threshold every coord
    perms = [np.concatenate([[0], rng.permutation(np.arange(1, d))]) for _ in range(n_orders)]

    per_fam_sd = np.empty(n_fam)
    canon = np.empty(n_fam)
    for i in range(n_fam):
        lower = np.where(case[i], t, -np.inf); upper = np.where(case[i], np.inf, t)
        lower[0], upper[0] = -np.inf, np.inf                      # g is latent, untruncated
        ests = np.empty(n_orders)
        for k, perm in enumerate(perms):
            e, _ = pa_algorithm(cov[np.ix_(perm, perm)], lower[perm], upper[perm], target=0)
            ests[k] = e
        per_fam_sd[i] = ests.std()
        canon[i] = ests[0]
    between = float(np.std(canon))
    p95 = float(np.percentile(per_fam_sd, 95))
    return dict(n_rel=d - 2, median_sd=float(np.median(per_fam_sd)), p95_sd=p95,
                rel_median=float(np.median(per_fam_sd) / between) if between > 0 else 0.0,
                rel_p95=float(p95 / between) if between > 0 else 0.0,
                between_sd=between)


def _mean_se(vals):
    vals = np.asarray(vals, float)
    if len(vals) < 2:
        return float(vals[0]), 0.0
    return float(vals.mean()), float(vals.std(ddof=1) / np.sqrt(len(vals)))


def aggregate(rows, key, metrics):
    """Across-seed mean ± SE per ``key`` value, preserving first-seen order."""
    out = []
    for k in dict.fromkeys(r[key] for r in rows):
        sub = [r for r in rows if r[key] == k]
        agg = {key: k, "n_rel": sub[0]["n_rel"], "n_seeds": len(sub)}
        for m in metrics:
            agg[m], agg[m + "_se"] = _mean_se([r[m] for r in sub])
        out.append(agg)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-fam", type=int, default=2000)
    ap.add_argument("--n-sim", type=int, default=25_000)
    ap.add_argument("--orders", type=int, default=10)
    ap.add_argument("--order-nfam", type=int, default=800)
    ap.add_argument("--prev", type=float, default=0.05)
    ap.add_argument("--reps", type=int, default=3,
                    help="independent seeds per grid cell")
    ap.add_argument("--seed", type=int, default=1, help="base seed")
    args = ap.parse_args()
    seeds = [args.seed + 1000 * r for r in range(args.reps)]

    estimate(simulate_families(["m", "f"], 0.5, 0.1, 40, 0).families, 0.5, "gibbs",
             n_sim=500, seed=0)                              # warm JIT

    print("== (a) PA vs Gibbs on stressful pedigrees (h2=0.5, %d seed(s)) =="
          % len(seeds))
    reg = []
    for seed in seeds:
        print("  seed %d:" % seed)
        reg.extend(dict(seed=seed, **r)
                   for r in regimes(args.n_fam, seed, args.n_sim))
    agg_reg = aggregate(reg, "regime", ["agree", "corr_pa", "corr_gibbs"])
    print("  across-seed mean +/- SE:")
    for r in agg_reg:
        amin = min(s["agree"] for s in reg if s["regime"] == r["regime"])
        print("  %-18s | PA<->Gibbs=%.5f+/-%.5f (min %.5f) | "
              "corr(PA,g)=%.3f+/-%.3f corr(Gibbs,g)=%.3f+/-%.3f"
              % (r["regime"], r["agree"], r["agree_se"], amin,
                 r["corr_pa"], r["corr_pa_se"], r["corr_gibbs"],
                 r["corr_gibbs_se"]))

    print("== (b) fold-in ordering sensitivity vs pedigree size (K=%.3f, %d orders, "
          "%d seed(s)) ==" % (args.prev, args.orders, len(seeds)))
    fo = []
    for seed in seeds:
        for name, fam_vec in SIZES.items():
            d = fold_order(fam_vec, args.prev, args.order_nfam, args.orders, seed)
            fo.append(dict(seed=seed, name=name, **d))
    agg_fo = aggregate(fo, "name",
                       ["median_sd", "p95_sd", "rel_median", "rel_p95"])
    print("  across-seed mean +/- SE:")
    for r in agg_fo:
        print("  %-22s | order-SD median=%.5f+/-%.5f p95=%.5f+/-%.5f | "
              "vs signal SD: median %.2f%%+/-%.2f p95 %.2f%%+/-%.2f"
              % (r["name"], r["median_sd"], r["median_sd_se"],
                 r["p95_sd"], r["p95_sd_se"],
                 100 * r["rel_median"], 100 * r["rel_median_se"],
                 100 * r["rel_p95"], 100 * r["rel_p95_se"]))

    write_csv(reg, fo)
    print("\nwrote bench_pa_robustness.csv")


def write_csv(reg, fo):
    path = os.path.join(HERE, "bench_pa_robustness.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["panel", "label", "n_rel", "seed", "agree", "corr_pa",
                    "corr_gibbs", "median_sd", "p95_sd", "rel_median",
                    "rel_p95", "between_sd"])
        for r in reg:
            w.writerow(["regime", r["regime"], r["n_rel"], r["seed"], r["agree"],
                        r["corr_pa"], r["corr_gibbs"], "", "", "", "", ""])
        for r in fo:
            w.writerow(["foldorder", r["name"], r["n_rel"], r["seed"],
                        "", "", "", r["median_sd"], r["p95_sd"], r["rel_median"],
                        r["rel_p95"], r["between_sd"]])


if __name__ == "__main__":
    main()
