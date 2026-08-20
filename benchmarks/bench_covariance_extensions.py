"""Two covariance extensions, one question each: is the fancier Sigma worth it?

Both panels draw families from a **true** extended covariance from
``research.covariance_extensions`` -- so the proband's genetic liability is
known exactly -- then score a ladder of estimators against that truth through
the Pearson-Aitken algorithm. They share the scaffolding: thresholded
case/control bounds, a PA scoring pass, corr + calibration-slope metrics, and
paired-across-replicate t confidence intervals.

Panel (a) -- sex limitation
---------------------------
ltpred has always let sex enter through the **threshold**: a sex-specific
prevalence gives each person their own `T`. `construct_covmat_sex_limited`
also lets it enter the **covariance**, with sex-specific heritabilities and a
cross-sex genetic correlation `rg`. The algebra guarantees the BLUP weights
change -- `w = V^-1 c` depends on `V`, and `V` now depends on sex. It does
*not* guarantee the score gets better, nor by how much, nor under what
conditions. Four estimators:

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

Arm 2 vs arm 3 is the contrast that matters: both see the same thresholds, so
any difference is attributable to the covariance alone. Three sub-panels:

  (a1) sweep `rg` at a fixed heritability gap -- when does *qualitative* sex
       limitation matter?
  (a2) sweep the heritability gap at `rg = 1` -- when does *scalar* sex
       limitation matter, with no qualitative component at all?
  (a3) **mechanism.** `rg` discounts cross-sex pairs and nothing else, so its
       effect must vanish in an all-same-sex family and concentrate in an
       all-cross-sex one. Two matched compositions -- proband + mother +
       sister versus proband + father + brother -- give the proband exactly
       two first-degree relatives each, so they carry the same information
       and differ *only* in sex configuration. Heritability is equal for both
       sexes here, so scalar limitation cannot contribute and every bit of
       any gain is attributable to `rg`.

Panel (b) -- genetic nurture
----------------------------
Under genetic nurture the proband's liability carries two things: their
**own** additive value `A_o`, and an **indirect** contribution from the
parents' genotypes acting through the rearing environment. A GWAS phenotype
should predict `A_o` alone. A nurture-blind estimator cannot separate them,
so its score is contaminated by the parental path. Four estimators, scored
against the true `A_o` from `construct_covmat_nurture`:

  1. **additive, true h2** -- the ordinary covariance at the true *direct*
     heritability. Isolates the covariance misspecification on its own;
  2. **additive, moment h2** -- the realistic nurture-blind pipeline. A
     moment-based heritability fitted from parent-offspring covariance
     absorbs the indirect path and comes out inflated, at `h2*(1 + 2n)`;
  3. **A + C matched to sibs** -- the confounded alternative. `c2` is chosen
     so the model reproduces the *sib-sib* covariance exactly. This is what
     an analyst with no parental phenotypes would fit, and it is
     indistinguishable from nurture on sibling data alone;
  4. **nurture, true parameters** -- the ceiling.

Two sub-panels:

  (b1) sweep the nurture coefficient `n` -- how fast does the cost grow?
  (b2) the identifiability trap made consequential. Arm 3 fits sibling
       covariance *perfectly* and still mis-scores the direct effect, because
       it gets parent-offspring covariance wrong. Reports the two mutually
       inconsistent heritabilities a nurture-blind moment fitter would obtain
       from the two relative types -- their disagreement is the diagnostic
       that nurture is present at all.

Reported per arm in both panels: `corr(estimate, truth)` (ranking, what a
linear GWAS uses) and the calibration slope `regress(truth on estimate)`
(scale). Panel (a) additionally reports the squared-correlation effective-N
proxy for the covariance fix.

    python benchmarks/bench_covariance_extensions.py
    python benchmarks/bench_covariance_extensions.py --panels sex --reps 1
    python benchmarks/bench_covariance_extensions.py --panels nurture --nurture-n-fam 4000
Writes bench_sex_limitation.csv and bench_nurture.csv (+ .png if matplotlib is
present) -- the same artefacts the standalone scripts produced.
"""

import os
import sys
import csv
import argparse
from pathlib import Path

import numpy as np
from scipy.stats import norm, t as student_t

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.covariance_extensions import (construct_covmat_sex_limited,
                                            construct_covmat_nurture)
from ltpred.covariance import construct_covmat_single
from ltpred.pearson_aitken import pa_algorithm

from _common import get_plt

HERE = os.path.dirname(os.path.abspath(__file__))

FAM_VEC = ("m", "f", "s1")          # proband + both parents + one full sib


# --------------------------------------------------------------------------- #
#  Shared scaffolding: bounds -> PA score -> metrics -> replicate summaries   #
# --------------------------------------------------------------------------- #
def _bounds(roles, draws, thr):
    """Case/control bounds from each person's liability and threshold ``thr[j]``.

    The target stays unbounded; passing a constant ``thr`` is exactly the
    threshold-blind arm of either panel.
    """
    n, d = draws.shape
    lower = np.full((n, d), -np.inf)
    upper = np.full((n, d), np.inf)
    for j, role in enumerate(roles):
        if role == "g":
            continue                      # the target stays unbounded
        case = draws[:, j] > thr[j]
        lower[case, j] = thr[j]
        upper[~case, j] = thr[j]
    return lower, upper


def _score(roles, draws, cov, thr):
    """Run PA over every family under one (threshold, covariance) choice."""
    lower, upper = _bounds(roles, draws, thr)
    tgt = roles.index("g")
    est = np.empty(draws.shape[0])
    for i in range(draws.shape[0]):
        est[i], _ = pa_algorithm(cov, lower[i], upper[i], target=tgt)
    return est, draws[:, tgt]


def _metrics(est, truth):
    """corr(estimate, truth) and the calibration slope regress(truth on estimate)."""
    est = np.asarray(est, dtype=float)
    ok = np.isfinite(est) & np.isfinite(truth)
    est, truth = est[ok], truth[ok]
    if est.size < 2 or np.std(est) == 0:
        return dict(corr=np.nan, slope=np.nan)
    corr = float(np.corrcoef(est, truth)[0, 1])
    slope = float(np.polyfit(est, truth, 1)[0])   # regress truth on estimate
    return dict(corr=corr, slope=slope, nonfinite=int((~ok).sum()))


def summarise_t(values):
    """Mean and 95% t CI across replicates (pairing away simulation noise)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    mean = float(np.mean(v))
    if len(v) <= 1:
        return mean, 0.0
    se = float(np.std(v, ddof=1) / np.sqrt(len(v)))
    return mean, float(student_t.ppf(0.975, len(v) - 1) * se)


def write_csv(name, fields, rows):
    with open(os.path.join(HERE, name), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows([{k: r.get(k, "") for k in fields} for r in rows])


# --------------------------------------------------------------------------- #
#  Panel (a): sex-limited covariance                                          #
# --------------------------------------------------------------------------- #
SEXES = ("F", "M")

# Matched-relatedness compositions for the mechanism sub-panel. Both give the
# proband exactly two first-degree relatives (2*phi = 0.5 each), so they carry
# the same amount of information and differ only in sex configuration.
COMPOSITIONS = {
    # every pair same-sex: rg can never apply
    "same-sex": dict(fam_vec=("m", "s1"), sex_o="F", sex_s="F"),
    # every proband-relative pair cross-sex: rg applies to both links
    "cross-sex": dict(fam_vec=("f", "s1"), sex_o="F", sex_s="M"),
}

SEX_ARMS = ("pooled", "sex_thr", "sigma_true", "sigma_rg1")
SEX_FIELDS = ["panel", "composition", "rg", "h2_female", "h2_male", "reps"] + [
    f"{arm}_{stat}{s}" for arm in SEX_ARMS for stat in ("corr", "slope")
    for s in ("", "_ci95")] + ["gain", "gain_ci95", "effn", "effn_ci95",
                               "rg_cost", "rg_cost_ci95"]


def _covmat_sex(fam_vec, sex_o, sex_s, h2f, h2m, rg):
    return construct_covmat_sex_limited(
        fam_vec, h2_female=h2f, h2_male=h2m, rg_cross=rg,
        sex={"o": sex_o, "s1": sex_s})


def _person_sexes(roles, sex_o, sex_s):
    """Sex of each row, mirroring the constructor's own resolution."""
    fixed = {"m": "F", "f": "M", "o": sex_o, "g": sex_o, "s1": sex_s}
    return [fixed[r] for r in roles]


def _draw_sex(fam_vec, sex_o, sex_s, n, h2f, h2m, rg, rng):
    cov = _covmat_sex(fam_vec, sex_o, sex_s, h2f, h2m, rg)
    draws = rng.multivariate_normal(np.zeros(len(cov.roles)), cov.matrix,
                                    size=n, method="eigh")
    return dict(roles=cov.roles, sexes=_person_sexes(cov.roles, sex_o, sex_s),
                draws=draws, fam_vec=fam_vec, sex_o=sex_o, sex_s=sex_s)


def simulate_sex(n_fam, h2f, h2m, rg, rng, cells=None):
    """Draw families from the true sex-limited model.

    ``cells`` is a list of ``(fam_vec, sex_o, sex_s)``; the default balances the
    four (proband sex, sibling sex) combinations of the standard family. Each
    cell shares one covariance and is drawn in a single vectorised call.
    """
    if cells is None:
        cells = [(FAM_VEC, so, ss) for so in SEXES for ss in SEXES]
    n = n_fam // len(cells)
    return [_draw_sex(fv, so, ss, n, h2f, h2m, rg, rng) for fv, so, ss in cells]


def _score_sex(groups, k_by_sex, cov_of):
    """Score every cell under one (per-sex threshold, covariance) choice."""
    est, truth = [], []
    for g in groups:
        thr = np.array([norm.ppf(1.0 - k_by_sex[s]) for s in g["sexes"]])
        e, t = _score(g["roles"], g["draws"], cov_of(g), thr)
        est.append(e)
        truth.append(t)
    return np.concatenate(est), np.concatenate(truth)


def run_sex_cell(h2f, h2m, rg, *, n_fam, k_female, k_male, reps, seed0,
                 cells=None):
    """One (h2f, h2m, rg) setting, replicated; returns means and 95% CIs."""
    acc = {a: {"corr": [], "slope": []} for a in SEX_ARMS}
    effn = []

    # A single pooled heritability and prevalence: what a sex-blind analyst uses.
    h2_pooled = 0.5 * (h2f + h2m)
    k_pooled = 0.5 * (k_female + k_male)
    k_sex = {"F": k_female, "M": k_male}
    k_flat = {"F": k_pooled, "M": k_pooled}

    def cov_with(h2a, h2b, rgx):
        return lambda g: _covmat_sex(g["fam_vec"], g["sex_o"], g["sex_s"],
                                     h2a, h2b, rgx).matrix

    for r in range(reps):
        rng = np.random.default_rng(seed0 + r)
        groups = simulate_sex(n_fam, h2f, h2m, rg, rng, cells=cells)

        for arm, k_map, cov_of in (
                ("pooled", k_flat, cov_with(h2_pooled, h2_pooled, 1.0)),
                ("sex_thr", k_sex, cov_with(h2_pooled, h2_pooled, 1.0)),
                ("sigma_true", k_sex, cov_with(h2f, h2m, rg)),
                ("sigma_rg1", k_sex, cov_with(h2f, h2m, 1.0))):
            m = _metrics(*_score_sex(groups, k_map, cov_of))
            acc[arm]["corr"].append(m["corr"])
            acc[arm]["slope"].append(m["slope"])
        # squared-correlation effective-N proxy, the convention used elsewhere
        # in RESULTS.md: how many times more case/control-equivalent data the
        # covariance fix is worth, thresholds held fixed.
        effn.append((acc["sigma_true"]["corr"][-1] / acc["sex_thr"]["corr"][-1]) ** 2)

    out = dict(reps=reps)
    for arm in SEX_ARMS:
        for stat in ("corr", "slope"):
            out[f"{arm}_{stat}"], out[f"{arm}_{stat}_ci95"] = summarise_t(acc[arm][stat])
    # the contrast of interest: covariance-only, thresholds held fixed
    gain = np.asarray(acc["sigma_true"]["corr"]) - np.asarray(acc["sex_thr"]["corr"])
    out["gain"], out["gain_ci95"] = summarise_t(gain)
    out["effn"], out["effn_ci95"] = summarise_t(effn)
    # cost of asserting rg = 1 when it is not
    rgcost = np.asarray(acc["sigma_true"]["corr"]) - np.asarray(acc["sigma_rg1"]["corr"])
    out["rg_cost"], out["rg_cost_ci95"] = summarise_t(rgcost)
    return out


def sex_panel(args):
    """Panel (a): pooled / sex-thresholds / true-Sigma / Sigma-with-rg=1."""
    kw = dict(n_fam=args.sex_n_fam, k_female=args.k_female, k_male=args.k_male,
              reps=args.sex_reps)
    hdr = (f"{'':>6} | {'sex thr':>8} {'Sigma':>8} | {'gain ± 95% t CI':>17} "
           f"{'eff-N ± 95% t CI':>19} | {'slope thr':>9} {'slope Sig':>9}")

    def show(label, m):
        print(f"{label:>6} | {m['sex_thr_corr']:8.4f} {m['sigma_true_corr']:8.4f} | "
              f"{m['gain']:+.4f} ± {m['gain_ci95']:.4f} "
              f"{m['effn']:6.3f}x ± {m['effn_ci95']:.3f} | "
              f"{m['sex_thr_slope']:9.4f} {m['sigma_true_slope']:9.4f}")

    print("(a) sex-limited covariance: is putting sex in Sigma worth it, "
          "or do sex-specific thresholds suffice?")
    print(f"  (a1) vs true cross-sex rg   [h2_F={args.h2_female}, h2_M={args.h2_male}, "
          f"K_F={args.k_female}, K_M={args.k_male}, {args.sex_reps} reps x "
          f"{args.sex_n_fam} fam]")
    print(hdr)
    rows = []
    for rg in args.rg:
        m = run_sex_cell(args.h2_female, args.h2_male, rg, seed0=args.seed, **kw)
        rows.append(dict(panel="rg", rg=rg, composition="mixed",
                         h2_female=args.h2_female, h2_male=args.h2_male, **m))
        show(f"{rg:.2f}", m)

    print("\n  (a2) vs heritability gap    [rg = 1, mean h2 = 0.4]")
    print(hdr)
    for gap in args.gaps:
        h2f, h2m = 0.4 + gap / 2, 0.4 - gap / 2
        m = run_sex_cell(h2f, h2m, 1.0, seed0=args.seed + 500, **kw)
        rows.append(dict(panel="gap", rg=1.0, composition="mixed",
                         h2_female=h2f, h2_male=h2m, **m))
        show(f"{gap:.2f}", m)

    # (a3) mechanism: rg discounts cross-sex pairs only, so its effect must
    # vanish in an all-same-sex family and concentrate in an all-cross-sex one.
    # Both compositions give the proband two first-degree relatives, so they
    # are matched on relatedness and differ only in sex configuration.
    print("\n  (a3) mechanism: matched two-relative families, equal h2 both sexes")
    print(f"      [h2_F = h2_M = {args.h2_mech}, so *all* of any gain is rg]")
    print(f"{'':>6} {'composition':>12} | {'sex thr':>8} {'Sigma':>8} | "
          f"{'gain ± 95% t CI':>17}")
    for name, spec in COMPOSITIONS.items():
        cells = [(spec["fam_vec"], spec["sex_o"], spec["sex_s"])]
        for rg in args.rg_mech:
            m = run_sex_cell(args.h2_mech, args.h2_mech, rg,
                             seed0=args.seed + 900, cells=cells, **kw)
            rows.append(dict(panel="mech", rg=rg, composition=name,
                             h2_female=args.h2_mech, h2_male=args.h2_mech, **m))
            print(f"{rg:6.2f} {name:>12} | {m['sex_thr_corr']:8.4f} "
                  f"{m['sigma_true_corr']:8.4f} | "
                  f"{m['gain']:+.4f} ± {m['gain_ci95']:.4f}")

    write_csv("bench_sex_limitation.csv", SEX_FIELDS, rows)
    plot_sex(rows)
    print("\nwrote bench_sex_limitation.csv")


def plot_sex(rows):
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


# --------------------------------------------------------------------------- #
#  Panel (b): genetic nurture                                                 #
# --------------------------------------------------------------------------- #
NUR_ARMS = ("add_true", "add_moment", "ace_sibs", "nurture")
NUR_FIELDS = ["h2", "nurture", "reps", "h2_moment_po", "h2_moment_sib",
              "c2_matched"] + [
    f"{arm}_{stat}{s}" for arm in NUR_ARMS for stat in ("corr", "slope")
    for s in ("", "_ci95")] + ["cost", "cost_ci95", "trap", "trap_ci95"]


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


def simulate_nurture(n_fam, h2, n, rng):
    cov = construct_covmat_nurture(FAM_VEC, h2=h2, nurture=n)
    draws = rng.multivariate_normal(np.zeros(len(cov.roles)), cov.matrix,
                                    size=n_fam, method="eigh")
    return cov.roles, draws


def run_nurture_cell(h2, n, *, n_fam, prevalence, reps, seed0):
    """One (h2, n) setting, replicated; returns means and 95% CIs."""
    acc = {a: {"corr": [], "slope": []} for a in NUR_ARMS}

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
        roles, draws = simulate_nurture(n_fam, h2, n, rng)
        thr = np.full(len(roles), norm.ppf(1.0 - prevalence))
        for arm in NUR_ARMS:
            if covs[arm] is None:
                acc[arm]["corr"].append(np.nan)
                acc[arm]["slope"].append(np.nan)
                continue
            m = _metrics(*_score(roles, draws, covs[arm], thr))
            acc[arm]["corr"].append(m["corr"])
            acc[arm]["slope"].append(m["slope"])

    out = dict(reps=reps, h2=h2, nurture=n, h2_moment_po=h2_po,
               h2_moment_sib=h2_ss, c2_matched=c2)
    for arm in NUR_ARMS:
        for stat in ("corr", "slope"):
            out[f"{arm}_{stat}"], out[f"{arm}_{stat}_ci95"] = summarise_t(acc[arm][stat])
    cost = np.asarray(acc["nurture"]["corr"]) - np.asarray(acc["add_moment"]["corr"])
    out["cost"], out["cost_ci95"] = summarise_t(cost)
    trap = np.asarray(acc["nurture"]["corr"]) - np.asarray(acc["ace_sibs"]["corr"])
    out["trap"], out["trap_ci95"] = summarise_t(trap)
    return out


def nurture_panel(args):
    """Panel (b): additive-true / additive-moment / A+C-sibs / nurture-true."""
    kw = dict(n_fam=args.nurture_n_fam, prevalence=args.prevalence,
              reps=args.nurture_reps)
    print("\n(b) genetic nurture: what does ignoring the indirect path cost "
          "the direct-effect score?")
    print(f"  (b1) cost of ignoring nurture   [h2 = {args.h2}, "
          f"K = {args.prevalence}, {args.nurture_reps} reps x "
          f"{args.nurture_n_fam} fam]")
    print(f"{'n':>5} | {'add(true)':>9} {'add(moment)':>11} {'A+C sibs':>9} "
          f"{'nurture':>8} | {'cost ± 95% t CI':>20}")
    rows = []
    for n in args.nurture:
        m = run_nurture_cell(args.h2, n, seed0=args.seed, **kw)
        rows.append(m)
        print(f"{n:5.2f} | {m['add_true_corr']:9.4f} {m['add_moment_corr']:11.4f} "
              f"{m['ace_sibs_corr']:9.4f} {m['nurture_corr']:8.4f} | "
              f"{m['cost']:+.4f} ± {m['cost_ci95']:.4f}")

    print("\n  (b2) the identifiability trap: A+C reproduces sib-sib covariance "
          "exactly")
    print(f"{'n':>5} | {'c2 matched':>10} | {'h2 from par-off':>15} "
          f"{'h2 from sibs':>12} | {'A+C slope':>9} {'nurture slope':>13} | "
          f"{'trap ± 95% t CI':>20}")
    for m in rows:
        print(f"{m['nurture']:5.2f} | {m['c2_matched']:10.4f} | "
              f"{m['h2_moment_po']:15.4f} {m['h2_moment_sib']:12.4f} | "
              f"{m['ace_sibs_slope']:9.4f} {m['nurture_slope']:13.4f} | "
              f"{m['trap']:+.4f} ± {m['trap_ci95']:.4f}")

    write_csv("bench_nurture.csv", NUR_FIELDS, rows)
    plot_nurture(rows)
    print("\nwrote bench_nurture.csv")


def plot_nurture(rows):
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


# --------------------------------------------------------------------------- #
#  CLI                                                                        #
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--panels", nargs="+", choices=("sex", "nurture"),
                   default=["sex", "nurture"], help="which panels to run")
    p.add_argument("--reps", type=int, default=None,
                   help="override both --sex-reps and --nurture-reps")
    p.add_argument("--seed", type=int, default=1)
    # panel (a): sex limitation
    p.add_argument("--sex-n-fam", type=int, default=2000,
                   help="families per replicate (split evenly over the 4 sex cells)")
    p.add_argument("--sex-reps", type=int, default=3, help="independent replicates")
    p.add_argument("--rg", type=float, nargs="+", default=[1.0, 0.8, 0.6, 0.4, 0.2],
                   help="sub-panel (a1): true cross-sex genetic correlations")
    p.add_argument("--h2-female", type=float, default=0.6)
    p.add_argument("--h2-male", type=float, default=0.2)
    p.add_argument("--gaps", type=float, nargs="+", default=[0.0, 0.2, 0.4, 0.6],
                   help="sub-panel (a2): h2_female - h2_male, centred on 0.4")
    p.add_argument("--h2-mech", type=float, default=0.5,
                   help="sub-panel (a3): heritability, equal for both sexes")
    p.add_argument("--rg-mech", type=float, nargs="+", default=[1.0, 0.6, 0.2],
                   help="sub-panel (a3): rg values for the composition contrast")
    p.add_argument("--k-female", type=float, default=0.05)
    p.add_argument("--k-male", type=float, default=0.10)
    # panel (b): genetic nurture
    p.add_argument("--nurture-n-fam", type=int, default=3000)
    p.add_argument("--nurture-reps", type=int, default=5, help="independent replicates")
    p.add_argument("--h2", type=float, default=0.4)
    p.add_argument("--nurture", type=float, nargs="+",
                   default=[0.0, 0.1, 0.2, 0.3], help="true indirect coefficients")
    p.add_argument("--prevalence", type=float, default=0.05)
    args = p.parse_args()
    if args.reps is not None:
        args.sex_reps = args.nurture_reps = args.reps

    if "sex" in args.panels:
        sex_panel(args)
    if "nurture" in args.panels:
        nurture_panel(args)


if __name__ == "__main__":
    main()
