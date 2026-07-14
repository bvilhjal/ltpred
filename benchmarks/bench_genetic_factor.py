"""Genetic factor model: does `fit_genetic_factor` recover a common genetic factor?

`fit_genetic_factor` fits `r_g ≈ Λ Λ' + Ψ` — a Genomic-SEM-style common-factor
model — to the genetic correlation matrix estimated by `fit_genetic_correlation`.
It asks whether one latent genetic factor explains the pairwise `r_g` among a set
of traits (a general genetic axis), with `srmr` as an in-sample misfit diagnostic.
This benchmark runs the **whole pipeline end to end** on simulated family
case/control data — simulate multi-trait families, fit `r_g`, then fit the factor
model — and reports, over independent replicate cohorts:

  (a) **loading recovery** when the truth *is* one factor — the fitted loadings Λ
      (mean ± across-replicate SD) against the planted values, and the `srmr` /
      `prop_explained` fit;
  (b) **mis-specification diagnostic** — when the truth is *two* independent genetic
      factors, a one-factor model should show larger in-sample `srmr` than a
      two-factor model. This planted contrast demonstrates a diagnostic, not a
      calibrated factor-number test or general model-selection rule.

Ground truth is the simulator's factor loadings.

    python benchmarks/bench_genetic_factor.py
    python benchmarks/bench_genetic_factor.py --reps 20 --n-fam 4000
Writes bench_genetic_factor.csv (+ .png if matplotlib is present).
"""

import os
import csv
import time
import argparse

import numpy as np

from _common import get_plt
from ltpred.covariance import construct_covmat_multi, correct_positive_definite
from ltpred.thresholds import liability_threshold
from ltpred.family import Family, Member
from ltpred.fit import fit_genetic_correlation, fit_genetic_factor

HERE = os.path.dirname(os.path.abspath(__file__))

FAM = ["m", "f", "s1", "s2"]
H2 = [0.5, 0.45, 0.4, 0.35, 0.3]                 # per-trait heritabilities (P = 5)
LOADINGS_1F = np.array([0.8, 0.7, 0.6, 0.5, 0.4])   # one common factor
# two-factor truth: block {0,1,2} and block {3,4}, zero genetic corr across blocks
BLOCK_A = np.array([0.8, 0.7, 0.6])
BLOCK_B = np.array([0.7, 0.6])


def rg_one_factor():
    R = np.outer(LOADINGS_1F, LOADINGS_1F)
    np.fill_diagonal(R, 1.0)
    return R


def rg_two_factor():
    P = len(H2)
    R = np.zeros((P, P))
    R[:3, :3] = np.outer(BLOCK_A, BLOCK_A)
    R[3:, 3:] = np.outer(BLOCK_B, BLOCK_B)
    np.fill_diagonal(R, 1.0)
    return R


def simulate_multi_trait(fam_vec, h2_vec, rg, n_fam, prev, seed):
    """Multi-trait families under the LTM with genetic correlation `rg` and **no**
    environmental cross-trait correlation, so the full-liability correlation is the
    purely-genetic `rg[p,q] sqrt(h2_p h2_q)`. Each member gets one interval per trait."""
    h2_vec = np.asarray(h2_vec, float)
    P = len(h2_vec)
    rp = rg * np.sqrt(np.outer(h2_vec, h2_vec))     # env corr = 0
    np.fill_diagonal(rp, 1.0)
    cov = construct_covmat_multi(fam_vec=fam_vec, add_ind=True, genetic_corrmat=rg,
                                 full_corrmat=rp, h2_vec=h2_vec)
    roles = cov.roles
    k = len(roles) // P
    fam_roles = roles[:k]
    Sig, _ = correct_positive_definite(cov.matrix)
    rng = np.random.default_rng(seed)
    liab = rng.multivariate_normal(np.zeros(len(roles)), Sig, size=n_fam)
    t = [float(liability_threshold(prev)) for _ in range(P)]
    obs = [r for r in fam_roles if r != "g"]
    fams = []
    for i in range(n_fam):
        members = []
        for r in obs:
            lo, hi = [], []
            for p in range(P):
                case = liab[i, p * k + fam_roles.index(r)] > t[p]
                lo.append(t[p] if case else -np.inf)
                hi.append(np.inf if case else t[p])
            members.append(Member(role=r, lower=lo, upper=hi))
        fams.append(Family(fam_id=i, members=members))
    return fams


def fit_pipeline(rg_true, n_fam, prev, seed, n_iter, burn_in, n_factors=1):
    fams = simulate_multi_trait(FAM, H2, rg_true, n_fam, prev, seed)
    gc = fit_genetic_correlation(fams, n_iter=n_iter, burn_in=burn_in,
                                 seed=seed + 100_000)
    fac = fit_genetic_factor(gc, n_factors=n_factors)
    return fac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-fam", type=int, default=3000)
    ap.add_argument("--prev", type=float, default=0.1)
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--n-iter", type=int, default=800)
    ap.add_argument("--burn-in", type=int, default=250)
    ap.add_argument("--seed", type=int, default=300)
    args = ap.parse_args()

    fit_pipeline(rg_one_factor(), 60, 0.1, 0, 20, 5)     # warm JIT
    rows = []

    # (a) loading recovery under a true single factor -------------------------
    print("== (a) single-factor recovery (P=%d, n_fam=%d, prev=%.2f) =="
          % (len(H2), args.n_fam, args.prev))
    P = len(H2)
    L = np.empty((args.reps, P))
    srmr_1f = np.empty(args.reps)
    prop_1f = np.empty(args.reps)
    t0 = time.time()
    for r in range(args.reps):
        fac = fit_pipeline(rg_one_factor(), args.n_fam, args.prev, args.seed + r,
                           args.n_iter, args.burn_in, n_factors=1)
        L[r] = fac.loadings.ravel()
        srmr_1f[r] = fac.srmr
        prop_1f[r] = fac.prop_explained
        if (r + 1) % 5 == 0 or r + 1 == args.reps:
            print(f"  recovery replicate {r + 1}/{args.reps}", flush=True)
    Lmean, Lsd = L.mean(0), L.std(0, ddof=1)
    for p in range(P):
        print("  trait %d: true=%.2f  fitted=%.3f ± %.3f" % (p, LOADINGS_1F[p],
              Lmean[p], Lsd[p]))
        rows.append(dict(panel="recovery", trait=p, true=LOADINGS_1F[p],
                         fitted=Lmean[p], sd=Lsd[p], srmr="", srmr_sd="", prop="",
                         n_fam=args.n_fam, reps=args.reps))
    print("  one-factor fit: srmr=%.3f ± %.3f   prop_explained=%.3f   [%.0fs]"
          % (srmr_1f.mean(), srmr_1f.std(ddof=1), prop_1f.mean(), time.time() - t0))
    rows.append(dict(panel="fit_1f_truth", trait="", true="", fitted="", sd="",
                     srmr=srmr_1f.mean(), srmr_sd=srmr_1f.std(ddof=1),
                     prop=prop_1f.mean(),
                     n_fam=args.n_fam, reps=args.reps))

    # (b) mis-specification: two-factor truth, 1- vs 2-factor fit --------------
    print("== (b) two-factor truth: in-sample srmr diagnostic ==")
    s1 = np.empty(args.reps)
    s2 = np.empty(args.reps)
    t0 = time.time()
    for r in range(args.reps):
        seed = args.seed + 1000 + r
        fams = simulate_multi_trait(FAM, H2, rg_two_factor(), args.n_fam,
                                    args.prev, seed)
        gc = fit_genetic_correlation(fams, n_iter=args.n_iter,
                                     burn_in=args.burn_in,
                                     seed=seed + 100_000)
        s1[r] = fit_genetic_factor(gc, n_factors=1).srmr
        s2[r] = fit_genetic_factor(gc, n_factors=2).srmr
        if (r + 1) % 5 == 0 or r + 1 == args.reps:
            print(f"  misspecification replicate {r + 1}/{args.reps}", flush=True)
    print("  1-factor srmr=%.3f ± %.3f   2-factor srmr=%.3f ± %.3f   [%.0fs]"
          % (s1.mean(), s1.std(ddof=1), s2.mean(), s2.std(ddof=1), time.time() - t0))
    print("  (compare: srmr under the true 1-factor model was %.3f)" % srmr_1f.mean())
    rows.append(dict(panel="misspec", trait="1-factor fit", true="", fitted="",
                     sd="", srmr=s1.mean(), srmr_sd=s1.std(ddof=1), prop="",
                     n_fam=args.n_fam, reps=args.reps))
    rows.append(dict(panel="misspec", trait="2-factor fit", true="", fitted="",
                     sd="", srmr=s2.mean(), srmr_sd=s2.std(ddof=1), prop="",
                     n_fam=args.n_fam, reps=args.reps))

    write_csv(rows)
    plot(LOADINGS_1F, Lmean, Lsd,
         (srmr_1f.mean(), srmr_1f.std(ddof=1)),
         (s1.mean(), s1.std(ddof=1)), (s2.mean(), s2.std(ddof=1)))
    print("\nwrote bench_genetic_factor.csv and bench_genetic_factor.png")


def write_csv(rows):
    fields = ["panel", "trait", "true", "fitted", "sd", "srmr", "srmr_sd",
              "prop", "n_fam", "reps"]
    path = os.path.join(HERE, "bench_genetic_factor.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows([{k: r.get(k, "") for k in fields} for r in rows])


def plot(true, fitted, sd, srmr_true1f, srmr_1f, srmr_2f):
    plt = get_plt()
    if plt is None:
        return
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.4))
    ax[0].errorbar(true, fitted, yerr=sd, fmt="o", capsize=4, label="fitted ± SD")
    lim = [0.0, max(true) + 0.15]
    ax[0].plot(lim, lim, "k--", lw=1, label="y = x")
    ax[0].set_xlabel("true loading")
    ax[0].set_ylabel("fitted loading")
    ax[0].set_title("(a) single-factor loading recovery")
    ax[0].legend(fontsize=8)
    labels = ["1-factor\n(true 1F)", "1-factor\n(true 2F)", "2-factor\n(true 2F)"]
    vals = [srmr_true1f[0], srmr_1f[0], srmr_2f[0]]
    errs = [srmr_true1f[1], srmr_1f[1], srmr_2f[1]]
    colors = ["#4c72b0", "#c44e52", "#55a868"]
    ax[1].bar(labels, vals, yerr=errs, capsize=4, color=colors)
    ax[1].axhline(0.08, ls=":", color="gray", lw=1,
                  label="descriptive reference (srmr=0.08)")
    ax[1].set_ylabel("srmr (off-diagonal misfit)")
    ax[1].set_title("(b) srmr diagnoses planted misspecification")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "bench_genetic_factor.png"), dpi=130)


if __name__ == "__main__":
    main()
