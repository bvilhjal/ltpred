"""Pearson-Aitken robustness: where does the deterministic approximation strain?

Pearson-Aitken folds relatives into the proband's liability one at a time, updating
moments by the Gaussian selection formula. That is *exact* for a single truncation
but an **approximation** for several at once, so two things are worth measuring:

  (a) **Agreement with the Gibbs posterior mean on stressful pedigrees** — large
      (many relatives), rare (very low prevalence), and densely-affected (many case
      relatives) families, the regimes where a moment approximation could drift.
      Metric: corr(PA, Gibbs) and each method's corr with the true `g`.
  (b) **Fold-in ordering sensitivity** — because the sequential update is
      approximate, the estimate can depend on the *order* relatives are folded in.
      Re-running PA under random orderings, this reports the order-induced spread of
      the per-proband estimate (absolute, and relative to the between-proband SD),
      versus pedigree size.

Truth `g` is known from the simulation. Expectation: PA tracks Gibbs to corr ≳ 0.99
across all regimes, and the ordering spread stays small across the tested pedigree
sizes. This script does not compare that spread with Gibbs Monte-Carlo noise.

    python benchmarks/bench_pa_robustness.py
    python benchmarks/bench_pa_robustness.py --n-fam 3000 --orders 12
Writes bench_pa_robustness.csv (+ .png if matplotlib is present).
"""

import os
import csv
import argparse

import numpy as np

from _common import simulate_families, estimate, get_plt
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


def _corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 1e-12 else 0.0


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
        row = dict(regime=name, n_rel=len(fam_vec), agree=_corr(pa, gib),
                   corr_pa=_corr(pa, g), corr_gibbs=_corr(gib, g))
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-fam", type=int, default=2000)
    ap.add_argument("--n-sim", type=int, default=25_000)
    ap.add_argument("--orders", type=int, default=10)
    ap.add_argument("--order-nfam", type=int, default=800)
    ap.add_argument("--prev", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    estimate(simulate_families(["m", "f"], 0.5, 0.1, 40, 0).families, 0.5, "gibbs",
             n_sim=500, seed=0)                              # warm JIT

    print("== (a) PA vs Gibbs on stressful pedigrees (h2=0.5) ==")
    reg = regimes(args.n_fam, args.seed, args.n_sim)

    print("== (b) fold-in ordering sensitivity vs pedigree size (K=%.3f, %d orders) =="
          % (args.prev, args.orders))
    fo = []
    for name, fam_vec in SIZES.items():
        d = fold_order(fam_vec, args.prev, args.order_nfam, args.orders, args.seed)
        fo.append(dict(name=name, **d))
        print("  %-22s | order-SD median=%.5f p95=%.5f | vs signal SD: median %.2f%% p95 %.2f%%"
              % (name, d["median_sd"], d["p95_sd"], 100 * d["rel_median"], 100 * d["rel_p95"]))

    write_csv(reg, fo)
    plot(reg, fo)
    print("\nwrote bench_pa_robustness.csv and bench_pa_robustness.png")


def write_csv(reg, fo):
    path = os.path.join(HERE, "bench_pa_robustness.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["panel", "label", "n_rel", "agree", "corr_pa", "corr_gibbs",
                    "median_sd", "p95_sd", "rel_median", "rel_p95", "between_sd"])
        for r in reg:
            w.writerow(["regime", r["regime"], r["n_rel"], r["agree"], r["corr_pa"],
                        r["corr_gibbs"], "", "", "", "", ""])
        for r in fo:
            w.writerow(["foldorder", r["name"], r["n_rel"], "", "", "", r["median_sd"],
                        r["p95_sd"], r["rel_median"], r["rel_p95"], r["between_sd"]])


def plot(reg, fo):
    plt = get_plt()
    if plt is None:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    names = [r["regime"] for r in reg]
    x = np.arange(len(names))
    ax[0].plot(x, [r["agree"] for r in reg], "-o", color="#2F7D4F", label="corr(PA, Gibbs)")
    ax[0].plot(x, [r["corr_pa"] for r in reg], "-s", color="#3B4A9C", label="corr(PA, g)")
    ax[0].plot(x, [r["corr_gibbs"] for r in reg], "--^", color="#B95C3C", label="corr(Gibbs, g)")
    ax[0].set_xticks(x); ax[0].set_xticklabels(names, fontsize=7.5, rotation=12)
    ax[0].set_ylabel("correlation")
    ax[0].set_title("(a) PA tracks Gibbs across stressful pedigrees")
    ax[0].legend(fontsize=8)

    nrel = [r["n_rel"] for r in fo]
    ax[1].plot(nrel, [100 * r["rel_p95"] for r in fo], "-o", color="#B07A16",
               label="worst-case (p95)")
    ax[1].plot(nrel, [100 * r["rel_median"] for r in fo], "-s", color="#3B4A9C",
               label="typical (median)")
    ax[1].set_xlabel("number of relatives folded in")
    ax[1].set_ylabel("order-induced spread (% of between-proband SD)")
    ax[1].set_title("(b) fold-order sensitivity stays small across sizes")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "bench_pa_robustness.png"), dpi=130)


if __name__ == "__main__":
    main()
