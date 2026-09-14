"""Cohort confounding: do personalized thresholds control genomic inflation?

Birth-cohort personalisation is one component of LT-FH++. This benchmark isolates
that component in a family-history model; it does not include onset-age or sex
effects and is therefore not the full LT-FH++ design. It tests the
confounding side directly. It simulates a secular prevalence trend (disease more
common in recent cohorts) together with **birth-cohort-correlated null SNPs**
(population stratification: allele frequency drifts with cohort). None of the test
SNPs affect liability, so any association is spurious. The genetic-liability
estimate is built two ways and run through a linear-regression GWAS against the
null SNPs; the genomic-control inflation `λ_GC` is read off:

  * **cohort-aware FH** — each person thresholded at their own cohort's prevalence
    `K(by) = K·R^((by−1965)/30)` (the LT-FH++ cohort component);
  * **single-K** — one prevalence `K` for everyone (cohort-blind), plus the raw
    **case/control** label as a reference.

Expectation: on cohort-correlated null SNPs the single-K estimate and the raw
case/control label **inflate `λ_GC`** (the cohort trend leaks in), while the
cohort-aware estimate holds `λ_GC ≈ 1`; on cohort-independent null SNPs all stay
at 1. The gap grows with the secular trend `R`.

    python benchmarks/bench_confounding.py
    python benchmarks/bench_confounding.py --trends 1 2 4 --n-fam 10000 --reps 5
Writes bench_confounding.csv (+ .png if matplotlib is present).
"""

import os
import argparse

import numpy as np

from _common import get_plt, gwas_chisq, lambda_gc, write_rows
from ltpred.covariance import construct_covmat_single, correct_positive_definite
from ltpred.thresholds import liability_threshold
from ltpred.family import Family, Member
from ltpred.estimate import estimate_liability

HERE = os.path.dirname(os.path.abspath(__file__))

STRUCT = ["m", "f", "s1", "s2"]
BY_REF = 1965.0
_BY_GAP = {"o": 0, "s": 0, "m": 30, "f": 30}            # birth-year offset (older = earlier)


def _cohort_K(by, K, R):
    return np.clip(K * R ** ((by - BY_REF) / 30.0), 1e-4, 0.9)


def simulate(fam_vec, h2, K, R, n_fam, m_snps, n_strat, strat, seed,
             by_lo=1930.0, by_hi=2000.0):
    """Cohort families + genotypes. ``g`` (drawn from the family covariance) is
    independent of birth year; test SNPs are null, a ``n_strat`` subset with allele
    frequency drifting with the proband's birth year. Returns ``g``, standardized
    genotypes ``Xs``, the stratified mask, and cohort-aware / single-K family
    encodings plus the raw case/control label."""
    cov_obj = construct_covmat_single(fam_vec=fam_vec, add_ind=True, h2=h2)
    cov, _ = correct_positive_definite(cov_obj.matrix)
    roles = cov_obj.roles
    non_g = roles[1:]
    o_pos = non_g.index("o")
    mem_cols = [roles.index(r) for r in non_g]
    rng = np.random.default_rng(seed)

    L = rng.multivariate_normal(np.zeros(len(roles)), cov, size=n_fam)
    g = L[:, 0]
    Lm = L[:, mem_cols]
    by_o = rng.uniform(by_lo, by_hi, n_fam)                 # proband birth year
    by = np.empty((n_fam, len(non_g)))
    for j, r in enumerate(non_g):
        by[:, j] = by_o - _BY_GAP.get(r.rstrip("0123456789"), 0)
    Ki = _cohort_K(by, K, R)                                # each member's cohort prevalence
    T_coh = np.asarray(liability_threshold(Ki), dtype=float)
    T_one = float(liability_threshold(K))
    case = Lm > T_coh                                       # truth uses the cohort threshold

    # genotypes: null SNPs; a subset has frequency drifting with proband birth year
    z = (by_o - by_o.mean()) / by_o.std()
    p_base = rng.uniform(0.1, 0.5, m_snps)
    coef = np.zeros(m_snps)
    coef[:n_strat] = rng.uniform(0.015, 0.04, n_strat) * rng.choice([-1.0, 1.0], n_strat)
    X = np.empty((n_fam, m_snps))
    for j in range(m_snps):
        p = np.clip(p_base[j] + coef[j] * z, 0.02, 0.98) if strat[j] else \
            np.full(n_fam, p_base[j])
        X[:, j] = rng.binomial(2, p)
    sd = X.std(axis=0); sd[sd == 0] = 1.0
    Xs = (X - X.mean(axis=0)) / sd

    fam_coh, fam_one = [], []
    for i in range(n_fam):
        mc, mo = [], []
        for j, r in enumerate(non_g):
            if case[i, j]:
                mc.append(Member(r, T_coh[i, j], np.inf)); mo.append(Member(r, T_one, np.inf))
            else:
                mc.append(Member(r, -np.inf, T_coh[i, j])); mo.append(Member(r, -np.inf, T_one))
        fam_coh.append(Family(i, mc)); fam_one.append(Family(i, mo))
    return dict(g=g, Xs=Xs, cc=case[:, o_pos].astype(float), fam_coh=fam_coh,
                fam_one=fam_one)


def run(fam_vec, h2, K, R, n_fam, m_snps, n_strat, seed):
    strat = np.zeros(m_snps, bool); strat[:n_strat] = True
    s = simulate(fam_vec, h2, K, R, n_fam, m_snps, n_strat, strat, seed)
    preds = dict(cohort=estimate_liability(s["fam_coh"], h2=h2, method="pa").est["genetic"],
                 single_K=estimate_liability(s["fam_one"], h2=h2, method="pa").est["genetic"],
                 casecontrol=s["cc"])
    out = dict(trend_R=R)
    for name, y in preds.items():
        chi2 = gwas_chisq(s["Xs"], y)
        out[f"lgc_strat_{name}"] = lambda_gc(chi2[strat])
        out[f"lgc_null_{name}"] = lambda_gc(chi2[~strat])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-fam", type=int, default=8000)
    ap.add_argument("--m-snps", type=int, default=2000)
    ap.add_argument("--n-strat", type=int, default=1000,
                    help="# null SNPs whose frequency drifts with birth cohort")
    ap.add_argument("--h2", type=float, default=0.5)
    ap.add_argument("--K", type=float, default=0.05)
    ap.add_argument("--trends", type=float, nargs="+", default=[1.0, 2.0, 3.0, 4.0])
    ap.add_argument("--reps", type=int, default=3,
                    help="independent cohorts per trend (reported as mean ± SE)")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    if args.reps < 1:
        ap.error("--reps must be at least 1")

    estimate_liability(simulate(STRUCT, args.h2, args.K, 2.0, 60, 40, 20,
                                np.r_[np.ones(20, bool), np.zeros(20, bool)], 0)["fam_coh"],
                       h2=args.h2, method="pa")             # warm JIT

    print("== cohort confounding & λ_GC (h2=%.2f, K=%.2f, %s, %d SNPs / %d stratified) =="
          % (args.h2, args.K, "+".join(STRUCT), args.m_snps, args.n_strat))
    print("  λ_GC on cohort-correlated null SNPs — should stay ≈1 only if cohort-aware")
    rows = []
    for R in args.trends:
        reps = [run(STRUCT, args.h2, args.K, R, args.n_fam, args.m_snps,
                    args.n_strat, args.seed + rep) for rep in range(args.reps)]
        keys = [key for key in reps[0] if key != "trend_R"]
        m = {"trend_R": R, "reps": args.reps}
        for key in keys:
            values = np.asarray([rep[key] for rep in reps])
            m[key] = float(values.mean())
            m[f"se_{key}"] = (float(values.std(ddof=1) / np.sqrt(args.reps))
                               if args.reps > 1 else 0.0)
        rows.append(m)
        print("  R=%.1f/30y | stratified-null λ_GC: cohort-aware=%.2f±%.2f  "
              "single-K=%.2f±%.2f  case/control=%.2f±%.2f | "
              "(unstratified cohort-aware=%.2f±%.2f)"
              % (R, m["lgc_strat_cohort"], m["se_lgc_strat_cohort"],
                 m["lgc_strat_single_K"], m["se_lgc_strat_single_K"],
                 m["lgc_strat_casecontrol"], m["se_lgc_strat_casecontrol"],
                 m["lgc_null_cohort"], m["se_lgc_null_cohort"]))

    write_csv(rows)
    plot(rows)
    print("\nwrote bench_confounding.csv and bench_confounding.png")


def write_csv(rows):
    metrics = [f"lgc_{s}_{m}" for s in ("strat", "null")
               for m in ("cohort", "single_K", "casecontrol")]
    fields = ["trend_R", "reps", *metrics, *[f"se_{metric}" for metric in metrics]]
    return write_rows(os.path.join(HERE, "bench_confounding.csv"), rows, fields)


def plot(rows):
    plt = get_plt()
    if plt is None:
        return
    rows = sorted(rows, key=lambda r: r["trend_R"])
    R = [r["trend_R"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.4))
    C = dict(cohort="#2F7D4F", single_K="#B95C3C", casecontrol="#888780")
    lab = dict(cohort="FH + cohort-specific K", single_K="single-K (cohort-blind)",
               casecontrol="case/control label")
    for key in ("cohort", "single_K", "casecontrol"):
        ax[0].errorbar(R, [r[f"lgc_strat_{key}"] for r in rows],
                       yerr=[r[f"se_lgc_strat_{key}"] for r in rows],
                       fmt="-o", capsize=3, color=C[key], label=lab[key])
    ax[0].axhline(1.0, color="k", ls=":", lw=1)
    ax[0].set_xlabel("secular prevalence trend  R  (× per 30 y)")
    ax[0].set_ylabel("λ_GC on cohort-correlated null SNPs")
    ax[0].set_title("(a) single-K inflates; cohort-aware holds λ_GC≈1")
    ax[0].legend(fontsize=8)
    # (b) stratified vs unstratified at the strongest trend
    top = rows[-1]
    keys = ["cohort", "single_K", "casecontrol"]
    x = np.arange(len(keys)); w = 0.38
    ax[1].bar(x - w / 2, [top[f"lgc_strat_{k}"] for k in keys], w, label="cohort-correlated SNPs",
              yerr=[top[f"se_lgc_strat_{k}"] for k in keys], capsize=3, color="#B95C3C")
    ax[1].bar(x + w / 2, [top[f"lgc_null_{k}"] for k in keys], w, label="independent null SNPs",
              yerr=[top[f"se_lgc_null_{k}"] for k in keys], capsize=3, color="#3B4A9C")
    ax[1].axhline(1.0, color="k", ls=":", lw=1)
    ax[1].set_xticks(x); ax[1].set_xticklabels([lab[k] for k in keys], fontsize=7.5, rotation=12)
    ax[1].set_ylabel("λ_GC")
    ax[1].set_title(f"(b) inflation is specific to cohort SNPs (R={top['trend_R']:.0f})")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "bench_confounding.png"), dpi=130)


if __name__ == "__main__":
    main()
