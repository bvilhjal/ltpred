"""Is putting sex in the covariance worth it, or do sex-specific thresholds suffice?

ltpred has always let sex enter through the **threshold**: a sex-specific
prevalence gives each person their own `T`. `construct_covmat_sex_limited` also
lets it enter the **covariance**, with sex-specific heritabilities and a
cross-sex genetic correlation `rg`.

The algebra guarantees the BLUP weights change -- `w = V^-1 c` depends on `V`,
and `V` now depends on sex. It does *not* guarantee the score gets better, nor
by how much, nor under what conditions. That is an empirical question, and this
benchmark answers it against a known truth.

Families are drawn from the **true** sex-limited covariance, so the proband's
genetic liability `g` is known exactly. Four estimators are then scored:

  1. **pooled** -- one prevalence and one scalar h2 for everyone (sex ignored
     entirely; the floor);
  2. **sex thresholds** -- sex-specific prevalence, scalar h2 covariance. This
     is what ltpred could do before `construct_covmat_sex_limited`;
  3. **sex in Sigma (true)** -- sex-specific prevalence *and* the true
     (h2_female, h2_male, rg) covariance. The ceiling;
  4. **sex in Sigma (rg=1)** -- sex-specific prevalence and sex-specific
     heritabilities, but the cross-sex correlation wrongly set to 1. Isolates
     the *qualitative* half of the model, and shows what mis-specifying `rg`
     costs -- which matters because these parameters are supplied, not fitted.

Reported per arm: `corr(estimate, true g)` (ranking, what a linear GWAS uses)
and the calibration slope `regress(true g on estimate)` (scale).

Two panels:

  (a) sweep `rg` at a fixed heritability gap -- when does *qualitative* sex
      limitation matter?
  (b) sweep the heritability gap at `rg = 1` -- when does *scalar* sex
      limitation matter, with no qualitative component at all?

Arm 2 vs arm 3 is the contrast that matters: both see the same thresholds, so
any difference is attributable to the covariance alone.

    python benchmarks/bench_sex_limitation.py
    python benchmarks/bench_sex_limitation.py --n-fam 4000 --reps 5
Writes bench_sex_limitation.csv (+ .png if matplotlib is present).
"""

import os
import csv
import argparse

import numpy as np
from scipy.stats import norm

from ltpred.covariance import construct_covmat_sex_limited
from ltpred.pearson_aitken import pa_algorithm

from _common import get_plt

HERE = os.path.dirname(os.path.abspath(__file__))

FAM_VEC = ("m", "f", "s1")          # proband + mother + father + one sibling
SEXES = ("F", "M")


def _covmat(sex_o, sex_s, h2f, h2m, rg):
    return construct_covmat_sex_limited(
        FAM_VEC, h2_female=h2f, h2_male=h2m, rg_cross=rg,
        sex={"o": sex_o, "s1": sex_s})


def _person_sexes(roles, sex_o, sex_s):
    """Sex of each row, mirroring the constructor's own resolution."""
    fixed = {"m": "F", "f": "M", "o": sex_o, "g": sex_o, "s1": sex_s}
    return [fixed[r] for r in roles]


def simulate(n_fam, h2f, h2m, rg, k_female, k_male, rng):
    """Draw families from the true sex-limited model.

    Returns per-family liability draws, the row sexes and the role ordering,
    grouped by the (proband sex, sibling sex) cell so each group shares one
    covariance and can be drawn in a single vectorised call.
    """
    groups = []
    for sex_o in SEXES:
        for sex_s in SEXES:
            n = n_fam // 4
            cov = _covmat(sex_o, sex_s, h2f, h2m, rg)
            draws = rng.multivariate_normal(np.zeros(len(cov.roles)),
                                            cov.matrix, size=n,
                                            method="eigh")
            sexes = _person_sexes(cov.roles, sex_o, sex_s)
            groups.append(dict(roles=cov.roles, sexes=sexes, draws=draws,
                               sex_o=sex_o, sex_s=sex_s))
    return groups


def _bounds(group, k_by_sex):
    """Case/control bounds from each person's liability and their threshold.

    ``k_by_sex`` maps sex -> prevalence, so passing the same value for both
    sexes is exactly the threshold-blind arm.
    """
    roles, sexes, draws = group["roles"], group["sexes"], group["draws"]
    n, d = draws.shape
    lower = np.full((n, d), -np.inf)
    upper = np.full((n, d), np.inf)
    for j, role in enumerate(roles):
        if role == "g":
            continue                      # the target stays unbounded
        thr = norm.ppf(1.0 - k_by_sex[sexes[j]])
        case = draws[:, j] > thr
        lower[case, j] = thr
        upper[~case, j] = thr
    return lower, upper


def _score(groups, k_by_sex, cov_of):
    """Run PA over every family under one (threshold, covariance) choice."""
    est, truth = [], []
    for g in groups:
        cov = cov_of(g)
        lower, upper = _bounds(g, k_by_sex)
        tgt = g["roles"].index("g")
        for i in range(g["draws"].shape[0]):
            e, _ = pa_algorithm(cov, lower[i], upper[i], target=tgt)
            est.append(e)
        truth.append(g["draws"][:, tgt])
    return np.asarray(est), np.concatenate(truth)


def _metrics(est, truth):
    est = np.asarray(est, dtype=float)
    ok = np.isfinite(est) & np.isfinite(truth)
    est, truth = est[ok], truth[ok]
    if est.size < 2 or np.std(est) == 0:
        return dict(corr=np.nan, slope=np.nan)
    corr = float(np.corrcoef(est, truth)[0, 1])
    slope = float(np.polyfit(est, truth, 1)[0])   # regress truth on estimate
    return dict(corr=corr, slope=slope, nonfinite=int((~ok).sum()))


def run_cell(h2f, h2m, rg, *, n_fam, k_female, k_male, reps, seed0):
    """One (h2f, h2m, rg) setting, replicated; returns means and 95% CIs."""
    arms = ("pooled", "sex_thr", "sigma_true", "sigma_rg1")
    acc = {a: {"corr": [], "slope": []} for a in arms}

    # A single pooled heritability and prevalence: what a sex-blind analyst uses.
    h2_pooled = 0.5 * (h2f + h2m)
    k_pooled = 0.5 * (k_female + k_male)
    k_sex = {"F": k_female, "M": k_male}
    k_flat = {"F": k_pooled, "M": k_pooled}

    for r in range(reps):
        rng = np.random.default_rng(seed0 + r)
        groups = simulate(n_fam, h2f, h2m, rg, k_female, k_male, rng)

        scalar = lambda g: _covmat(g["sex_o"], g["sex_s"], h2_pooled,      # noqa: E731
                                   h2_pooled, 1.0).matrix
        true_c = lambda g: _covmat(g["sex_o"], g["sex_s"], h2f, h2m, rg).matrix  # noqa: E731
        rg1_c = lambda g: _covmat(g["sex_o"], g["sex_s"], h2f, h2m, 1.0).matrix  # noqa: E731

        for arm, k_map, cov_of in (
                ("pooled", k_flat, scalar),
                ("sex_thr", k_sex, scalar),
                ("sigma_true", k_sex, true_c),
                ("sigma_rg1", k_sex, rg1_c)):
            m = _metrics(*_score(groups, k_map, cov_of))
            acc[arm]["corr"].append(m["corr"])
            acc[arm]["slope"].append(m["slope"])

    out = dict(reps=reps)
    for arm in arms:
        for stat in ("corr", "slope"):
            v = np.asarray(acc[arm][stat], dtype=float)
            out[f"{arm}_{stat}"] = float(np.nanmean(v))
            se = float(np.nanstd(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
            out[f"{arm}_{stat}_ci95"] = 1.96 * se
    # the contrast of interest: covariance-only, thresholds held fixed
    gain = np.asarray(acc["sigma_true"]["corr"]) - np.asarray(acc["sex_thr"]["corr"])
    out["gain"] = float(np.nanmean(gain))
    out["gain_ci95"] = float(1.96 * np.nanstd(gain, ddof=1) / np.sqrt(len(gain))) \
        if len(gain) > 1 else 0.0
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-fam", type=int, default=2000,
                   help="families per replicate (split evenly over the 4 sex cells)")
    p.add_argument("--reps", type=int, default=3, help="independent replicates")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--rg", type=float, nargs="+", default=[1.0, 0.8, 0.6, 0.4, 0.2],
                   help="panel (a): true cross-sex genetic correlations")
    p.add_argument("--h2-female", type=float, default=0.6)
    p.add_argument("--h2-male", type=float, default=0.2)
    p.add_argument("--gaps", type=float, nargs="+", default=[0.0, 0.2, 0.4, 0.6],
                   help="panel (b): h2_female - h2_male, centred on 0.4")
    p.add_argument("--k-female", type=float, default=0.05)
    p.add_argument("--k-male", type=float, default=0.10)
    args = p.parse_args()

    kw = dict(n_fam=args.n_fam, k_female=args.k_female, k_male=args.k_male,
              reps=args.reps)
    hdr = (f"{'':>6} | {'pooled':>8} {'sex thr':>8} {'Sigma':>8} {'Sig rg=1':>9} | "
           f"{'gain(Sigma-thr)':>18}")

    print(f"(a) vs true cross-sex rg   [h2_F={args.h2_female}, h2_M={args.h2_male}, "
          f"K_F={args.k_female}, K_M={args.k_male}, {args.reps} reps x {args.n_fam} fam]")
    print(hdr)
    rows = []
    for rg in args.rg:
        m = run_cell(args.h2_female, args.h2_male, rg, seed0=args.seed, **kw)
        rows.append(dict(panel="rg", rg=rg, h2_female=args.h2_female,
                         h2_male=args.h2_male, **m))
        print(f"{rg:6.2f} | {m['pooled_corr']:8.4f} {m['sex_thr_corr']:8.4f} "
              f"{m['sigma_true_corr']:8.4f} {m['sigma_rg1_corr']:9.4f} | "
              f"{m['gain']:+.4f} ± {m['gain_ci95']:.4f}")

    print("\n(b) vs heritability gap    [rg = 1, mean h2 = 0.4]")
    print(hdr)
    for gap in args.gaps:
        h2f, h2m = 0.4 + gap / 2, 0.4 - gap / 2
        m = run_cell(h2f, h2m, 1.0, seed0=args.seed + 500, **kw)
        rows.append(dict(panel="gap", rg=1.0, h2_female=h2f, h2_male=h2m, **m))
        print(f"{gap:6.2f} | {m['pooled_corr']:8.4f} {m['sex_thr_corr']:8.4f} "
              f"{m['sigma_true_corr']:8.4f} {m['sigma_rg1_corr']:9.4f} | "
              f"{m['gain']:+.4f} ± {m['gain_ci95']:.4f}")

    write_csv(rows)
    plot(rows)
    print("\nwrote bench_sex_limitation.csv")


def write_csv(rows):
    fields = ["panel", "rg", "h2_female", "h2_male", "reps"]
    for arm in ("pooled", "sex_thr", "sigma_true", "sigma_rg1"):
        for stat in ("corr", "slope"):
            fields += [f"{arm}_{stat}", f"{arm}_{stat}_ci95"]
    fields += ["gain", "gain_ci95"]
    with open(os.path.join(HERE, "bench_sex_limitation.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows([{k: r.get(k, "") for k in fields} for r in rows])


def plot(rows):
    plt = get_plt()
    if plt is None:
        return
    a = [r for r in rows if r["panel"] == "rg"]
    b = [r for r in rows if r["panel"] == "gap"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    x = [r["rg"] for r in a]
    for key, lab in (("sex_thr_corr", "sex thresholds only"),
                     ("sigma_true_corr", "sex in Sigma (true)"),
                     ("sigma_rg1_corr", "sex in Sigma (rg=1)")):
        ax[0].errorbar(x, [r[key] for r in a],
                       yerr=[r[f"{key}_ci95"] for r in a], fmt="-o", capsize=3,
                       label=lab)
    ax[0].set_xlabel("true cross-sex genetic correlation rg")
    ax[0].set_ylabel("corr(estimate, true genetic liability)")
    ax[0].set_title("(a) qualitative sex limitation")
    ax[0].legend(fontsize=8)
    gaps = [r["h2_female"] - r["h2_male"] for r in b]
    ax[1].errorbar(gaps, [r["gain"] for r in b],
                   yerr=[r["gain_ci95"] for r in b], fmt="-o", capsize=3,
                   color="tab:green")
    ax[1].axhline(0, color="k", lw=1)
    ax[1].set_xlabel("h2_female - h2_male")
    ax[1].set_ylabel("delta corr (Sigma - thresholds only)")
    ax[1].set_title("(b) scalar sex limitation, rg = 1")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "bench_sex_limitation.png"), dpi=130)


if __name__ == "__main__":
    main()
