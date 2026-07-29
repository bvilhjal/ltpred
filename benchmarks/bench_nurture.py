"""What does ignoring genetic nurture cost the direct-effect score?

Under genetic nurture the proband's liability carries two things: their **own**
additive value `A_o`, and an **indirect** contribution from the parents'
genotypes acting through the rearing environment. A GWAS phenotype should
predict `A_o` alone. A nurture-blind estimator cannot separate them, so its
score is contaminated by the parental path.

Families are drawn from the **true** nurture path model, so `A_o` is known
exactly, and four estimators are scored against it:

  1. **additive, true h2** -- the ordinary covariance at the true *direct*
     heritability. Isolates the covariance misspecification on its own;
  2. **additive, moment h2** -- the realistic nurture-blind pipeline. A
     moment-based heritability fitted from parent-offspring covariance absorbs
     the indirect path and comes out inflated, at `h2*(1 + 2n)`;
  3. **A + C matched to sibs** -- the confounded alternative. `c2` is chosen so
     the model reproduces the *sib-sib* covariance exactly. This is what an
     analyst with no parental phenotypes would fit, and it is indistinguishable
     from nurture on sibling data alone;
  4. **nurture, true parameters** -- the ceiling.

Reported: `corr(estimate, true A_o)` (ranking) and the calibration slope
`regress(A_o on estimate)` (scale).

Two panels:

  (a) sweep the nurture coefficient `n` -- how fast does the cost grow?
  (b) the identifiability trap made consequential. Arm 3 fits sibling
      covariance *perfectly* and still mis-scores the direct effect, because
      it gets parent-offspring covariance wrong. Reports the two mutually
      inconsistent heritabilities a nurture-blind moment fitter would obtain
      from the two relative types -- their disagreement is the diagnostic that
      nurture is present at all.

    python benchmarks/bench_nurture.py
    python benchmarks/bench_nurture.py --n-fam 4000 --reps 5
Writes bench_nurture.csv (+ .png if matplotlib is present).
"""

import os
import csv
import argparse

import numpy as np
from scipy.stats import norm

from ltpred.covariance import (construct_covmat_nurture,
                               construct_covmat_single)
from ltpred.pearson_aitken import pa_algorithm

from _common import get_plt

HERE = os.path.dirname(os.path.abspath(__file__))

FAM_VEC = ("m", "f", "s1")          # proband + both parents + one full sib


def moment_h2_from_parent_offspring(h2, n):
    """What a nurture-blind moment fitter reads off parent-offspring pairs.

    The additive model says that covariance is ``h2/2``; the truth is
    ``h2/2 + n*h2``, so the fitter returns ``h2*(1 + 2n)``.
    """
    return min(1.0, h2 * (1.0 + 2.0 * n))


def moment_h2_from_sibs(h2, n):
    """The same fitter reading off sib pairs: ``h2*(1 + 4n + 4n^2)``.

    It disagrees with the parent-offspring figure whenever ``n != 0``. That
    disagreement is the signature of an indirect path -- one additive model
    cannot satisfy both.
    """
    return min(1.0, h2 * (1.0 + 4.0 * n + 4.0 * n * n))


def c2_matched_to_sibs(h2, n):
    """The sibship environment that reproduces the nurture sib-sib covariance."""
    return 2.0 * n * h2 + 2.0 * n * n * h2


def simulate(n_fam, h2, n, rng):
    cov = construct_covmat_nurture(FAM_VEC, h2=h2, nurture=n)
    draws = rng.multivariate_normal(np.zeros(len(cov.roles)), cov.matrix,
                                    size=n_fam, method="eigh")
    return cov.roles, draws


def _bounds(roles, draws, prevalence):
    thr = norm.ppf(1.0 - prevalence)
    n, d = draws.shape
    lower = np.full((n, d), -np.inf)
    upper = np.full((n, d), np.inf)
    for j, role in enumerate(roles):
        if role == "g":
            continue                       # the target stays unbounded
        case = draws[:, j] > thr
        lower[case, j] = thr
        upper[~case, j] = thr
    return lower, upper


def _score(roles, draws, prevalence, cov):
    lower, upper = _bounds(roles, draws, prevalence)
    tgt = roles.index("g")
    est = np.empty(draws.shape[0])
    for i in range(draws.shape[0]):
        est[i], _ = pa_algorithm(cov, lower[i], upper[i], target=tgt)
    return est, draws[:, tgt]


def _metrics(est, truth):
    ok = np.isfinite(est) & np.isfinite(truth)
    est, truth = est[ok], truth[ok]
    if est.size < 2 or np.std(est) == 0:
        return np.nan, np.nan
    return (float(np.corrcoef(est, truth)[0, 1]),
            float(np.polyfit(est, truth, 1)[0]))


def run_cell(h2, n, *, n_fam, prevalence, reps, seed0):
    arms = ("add_true", "add_moment", "ace_sibs", "nurture")
    acc = {a: {"corr": [], "slope": []} for a in arms}

    h2_po = moment_h2_from_parent_offspring(h2, n)
    h2_ss = moment_h2_from_sibs(h2, n)
    c2 = c2_matched_to_sibs(h2, n)

    covs = {
        "add_true": construct_covmat_single(FAM_VEC, h2=h2).matrix,
        "add_moment": construct_covmat_single(FAM_VEC, h2=h2_po).matrix,
        # the A+C model an analyst without parental phenotypes would fit
        "ace_sibs": (construct_covmat_single(FAM_VEC, h2=h2, c2=c2).matrix
                     if h2 + c2 <= 1.0 else None),
        "nurture": construct_covmat_nurture(FAM_VEC, h2=h2, nurture=n).matrix,
    }

    for r in range(reps):
        rng = np.random.default_rng(seed0 + r)
        roles, draws = simulate(n_fam, h2, n, rng)
        for arm in arms:
            if covs[arm] is None:
                acc[arm]["corr"].append(np.nan)
                acc[arm]["slope"].append(np.nan)
                continue
            c, s = _metrics(*_score(roles, draws, prevalence, covs[arm]))
            acc[arm]["corr"].append(c)
            acc[arm]["slope"].append(s)

    def summarise(v):
        v = np.asarray(v, dtype=float)
        mean = float(np.nanmean(v))
        se = float(np.nanstd(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
        return mean, 1.96 * se

    out = dict(reps=reps, h2=h2, nurture=n, h2_moment_po=h2_po,
               h2_moment_sib=h2_ss, c2_matched=c2)
    for arm in arms:
        for stat in ("corr", "slope"):
            out[f"{arm}_{stat}"], out[f"{arm}_{stat}_ci95"] = summarise(acc[arm][stat])
    cost = np.asarray(acc["nurture"]["corr"]) - np.asarray(acc["add_moment"]["corr"])
    out["cost"], out["cost_ci95"] = summarise(cost)
    trap = np.asarray(acc["nurture"]["corr"]) - np.asarray(acc["ace_sibs"]["corr"])
    out["trap"], out["trap_ci95"] = summarise(trap)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-fam", type=int, default=3000)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--h2", type=float, default=0.4)
    p.add_argument("--nurture", type=float, nargs="+",
                   default=[0.0, 0.1, 0.2, 0.3], help="true indirect coefficients")
    p.add_argument("--prevalence", type=float, default=0.05)
    args = p.parse_args()

    kw = dict(n_fam=args.n_fam, prevalence=args.prevalence, reps=args.reps)
    print(f"(a) cost of ignoring nurture   [h2 = {args.h2}, K = {args.prevalence}, "
          f"{args.reps} reps x {args.n_fam} fam]")
    print(f"{'n':>5} | {'add(true)':>9} {'add(moment)':>11} {'A+C sibs':>9} "
          f"{'nurture':>8} | {'cost':>17}")
    rows = []
    for n in args.nurture:
        m = run_cell(args.h2, n, seed0=args.seed, **kw)
        rows.append(m)
        print(f"{n:5.2f} | {m['add_true_corr']:9.4f} {m['add_moment_corr']:11.4f} "
              f"{m['ace_sibs_corr']:9.4f} {m['nurture_corr']:8.4f} | "
              f"{m['cost']:+.4f} ± {m['cost_ci95']:.4f}")

    print("\n(b) the identifiability trap: A+C reproduces sib-sib covariance exactly")
    print(f"{'n':>5} | {'c2 matched':>10} | {'h2 from par-off':>15} "
          f"{'h2 from sibs':>12} | {'A+C slope':>9} {'nurture slope':>13} | "
          f"{'trap':>17}")
    for m in rows:
        print(f"{m['nurture']:5.2f} | {m['c2_matched']:10.4f} | "
              f"{m['h2_moment_po']:15.4f} {m['h2_moment_sib']:12.4f} | "
              f"{m['ace_sibs_slope']:9.4f} {m['nurture_slope']:13.4f} | "
              f"{m['trap']:+.4f} ± {m['trap_ci95']:.4f}")

    write_csv(rows)
    plot(rows)
    print("\nwrote bench_nurture.csv")


def write_csv(rows):
    fields = ["h2", "nurture", "reps", "h2_moment_po", "h2_moment_sib", "c2_matched"]
    for arm in ("add_true", "add_moment", "ace_sibs", "nurture"):
        for stat in ("corr", "slope"):
            fields += [f"{arm}_{stat}", f"{arm}_{stat}_ci95"]
    fields += ["cost", "cost_ci95", "trap", "trap_ci95"]
    with open(os.path.join(HERE, "bench_nurture.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows([{k: r.get(k, "") for k in fields} for r in rows])


def plot(rows):
    plt = get_plt()
    if plt is None:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    x = [r["nurture"] for r in rows]
    for key, lab in (("add_moment_corr", "additive (moment h2)"),
                     ("ace_sibs_corr", "A+C matched to sibs"),
                     ("nurture_corr", "nurture model")):
        ax[0].errorbar(x, [r[key] for r in rows],
                       yerr=[r[f"{key}_ci95"] for r in rows], fmt="-o", capsize=3,
                       label=lab)
    ax[0].set_xlabel("true nurture coefficient n")
    ax[0].set_ylabel("corr(estimate, true direct effect)")
    ax[0].set_title("(a) cost of ignoring genetic nurture")
    ax[0].legend(fontsize=8)
    ax[1].plot(x, [r["h2_moment_po"] for r in rows], "-o", label="h2 from parent-offspring")
    ax[1].plot(x, [r["h2_moment_sib"] for r in rows], "-o", label="h2 from sibs")
    ax[1].axhline(rows[0]["h2"], color="k", lw=1, ls="--", label="true direct h2")
    ax[1].set_xlabel("true nurture coefficient n")
    ax[1].set_ylabel("heritability a nurture-blind fitter returns")
    ax[1].set_title("(b) the two estimates disagree -- the diagnostic")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "bench_nurture.png"), dpi=130)


if __name__ == "__main__":
    main()
