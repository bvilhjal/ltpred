"""Generative validation of the PA-FGRS age-censored-control mixture.

The PA-FGRS censoring mixture (`use_mixture=True`; Dybdahl Krebs et al. 2024)
has unit tests, but this generative benchmark is needed to test calibration and
ranking against known genetic liability: simulate families under the
liability-threshold model with
an age-dependent CIP, censor honestly (a case is observed only if its onset age
precedes the current age; otherwise it is a censored control carrying its
individual cumulative incidence K_i), then ask whether the mixture recovers the
posterior mean genetic liability better than the naive no-mixture encoding.

Two observation models, because the mixture's value depends on how onset works:

  * CROSSING (the LT-FH++ convention, used by `simulate_under_LTM_single`):
    liability is fixed at birth and onset occurs when the age-specific
    threshold drops below it (deterministic l -> aoo map). Under this model a
    censored control IS exactly l < T(age), so the plain age truncation is
    already exact and the mixture has no correct work to do -- it should
    approximately reproduce the truncation.
  * STOCHASTIC-ONSET (the mixture's native model): case status is set by the
    lifetime threshold (l > T_pop) and onset age is drawn from the CIP
    independent of liability. A censored control is then a genuine mixture of
    true controls (l < T_pop) and future cases (l > T_pop, onset > age) --
    exactly what the mixture models, and what a naive age truncation
    mis-encodes. This is the arm that validates the mixture's purpose.
  * LIABILITY-DEPENDENT (rho=0.6): lifetime status at T_pop, but among
    lifetime cases onset is coupled to liability by a Gaussian copula
    (ltpred.simulate onset_model="liability_dependent"). Higher liability
    advances onset, with residual noise. Plain age truncation is then not
    exact, pinning is not exact, and the mixture's independence assumption
    is false: the not-yet-onset tail is depleted of high-liability people.
    This is the assumption-stress arm. Pre-registered: the mixture still
    lowers slope; a ranking cost may appear; pinned slope will not sit on 1.

Design. h2=0.5, lifetime prevalence K_pop=0.10, logistic CIP
K(age)=K_pop/(1+exp((mid_point-age)*slope)) (the repo's pa_thresholds family).
Two censoring regimes: MID-life (heavy censoring -- most lifetime cases are
still unobserved; the mixture should help most) and OLD (light censoring).
Encodings on identical bounds:

  base + no-mixture : lifetime case intervals [T_pop, inf); censored controls
                      truncated at their age threshold; plain PA (naive).
  base + mixture    : identical bounds + per-member (K_i, K_pop) and
                      use_mixture=True (the PA-FGRS model).
  interval + mixture: age-specific case intervals [T(aoo), inf) instead of the
                      lifetime interval -- the pa_thresholds PA-FGRS-style
                      variant, included to document its known calibration slope
                      (~0.84 in the unit guard).
  pinned + no-mixture / pinned + mixture: LT-FH++-exact case encoding
                      (lower == upper == T(aoo)), without and with the censoring
                      mixture. Under this generative model (deterministic
                      threshold crossing) the pinned case encoding is the exact
                      one, so these arms separate the case-encoding effect from
                      the censoring-mixture effect.

A Gibbs arm on the base bounds (no mixture) cross-checks the PA reference.
Metrics against the true simulated genetic liability g: Pearson correlation,
and calibration slope of regressing the true g on the estimate (1.0 =
calibrated posterior mean).

Reading the slopes: a lifetime case interval under-conditions cases (their true
liabilities are >= T(onset age), more extreme than [T_pop, inf) assumes) ->
slope > 1, worst in the MID regime; an age-specific interval over-conditions
(l > T(aoo) allows mass the pinned truth does not have) -> slope < 1. The
censoring mixture's own effect is the no-mixture -> mixture shift within an
encoding.

Pre-registered expectations (fixed before running): (1) the mixture does not
hurt calibration anywhere; (2) it improves the calibration slope toward 1 in
the MID regime; (3) correlation improves or matches; (4) the pinned encodings
calibrate best (slope nearest 1). If (1) fails the implementation is suspect.

Every replicate's corr/slope is retained and written to
``bench_pafgrs_mixture.csv`` (long format: one ``replicate`` row per
model x regime x rep x arm, plus ``paired_contrast`` rows). The paired
mixture-minus-no-mixture differences per replicate are reported as mean +/-
across-replicate SE with a t-based 95% CI (the bench_ltfhpp_personalization
convention), which is what the "no correlation cost" question needs.

Run:  python benchmarks/run_benchmark.py bench_pafgrs_mixture.py
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ltpred.covariance import construct_covmat_single, correct_positive_definite  # noqa: E402
from ltpred.estimate import estimate_liability  # noqa: E402
from ltpred.family import Family, Member  # noqa: E402
from ltpred.simulate import _onset_times  # noqa: E402
from ltpred.thresholds import liability_threshold, convert_age_to_thresh  # noqa: E402

SEED = 20260719
H2 = 0.5
K_POP = 0.10
MID_POINT = 60.0
SLOPE = 1.0 / 8.0
FAM_VEC = ["m", "f", "s1"]
N_FAM = 20_000
REPS = 5
GIBBS_N = 2_000          # smaller subset for the sampling reference
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_PREFIX = os.path.join(HERE, "bench_pafgrs_mixture")
AGE_RANGES = {           # (low, high) current-age ranges per regime
    "mid": {"o": (35, 55), "s": (35, 55), "m": (55, 75), "f": (55, 75)},
    "old": {"o": (60, 85), "s": (60, 85), "m": (75, 95), "f": (75, 95)},
}


def cip(age):
    """Logistic cumulative incidence proportion at ``age``."""
    return K_POP / (1.0 + np.exp((MID_POINT - np.asarray(age, dtype=float)) * SLOPE))


_ONSET_MAP = {
    "crossing": ("threshold_crossing", 0.0),
    "stochastic": ("stochastic", 0.0),
    "dependent": ("liability_dependent", 0.6),
}


def simulate_cohort(rng, regime, n_fam, onset_model="crossing"):
    """Liabilities + honest censoring; returns truth and per-member observations.

    onset_model='crossing': onset is the CIP inverse of liability.
    'stochastic': lifetime case if l > T_pop, onset ~ CIP independent of l.
    'dependent': same lifetime rule, onset coupled to l at rho=0.6."""
    cov_obj = construct_covmat_single(fam_vec=FAM_VEC, add_ind=True, h2=H2)
    cov, _ = correct_positive_definite(cov_obj.matrix)
    roles = cov_obj.roles                       # g, o, m, f, s1
    d = len(roles)
    liab = rng.multivariate_normal(np.zeros(d), cov, size=n_fam)

    t_pop = float(liability_threshold(K_POP))
    ages = {}
    for r in roles[1:]:
        lo, hi = AGE_RANGES[regime][r.rstrip("0123456789")]
        ages[r] = rng.integers(lo, hi, size=n_fam)

    sim_model, onset_rho = _ONSET_MAP[onset_model]
    obs = {r: [] for r in roles[1:]}
    for r in roles[1:]:
        col = roles.index(r)
        onset_r = _onset_times(
            liab[:, col], K_POP, MID_POINT, SLOPE, sim_model, t_pop, rng,
            onset_rho=onset_rho)
        for i in range(n_fam):
            aoo = onset_r[i]
            age = float(ages[r][i])
            is_case = np.isfinite(aoo) and float(aoo) <= age
            obs[r].append((is_case, float(aoo) if is_case else age))
    return liab, roles, obs


def build_families(obs, roles, n_fam, case_mode):
    """Family objects with bounds + K's. case_mode: 'base' (lifetime interval),
    'interval' (age-specific interval), or 'pinned' (LT-FH++-exact point mass).
    Controls carry K_i = CIP(age)."""
    t_pop = float(liability_threshold(K_POP))
    families = []
    for i in range(n_fam):
        members = []
        for r in roles[1:]:
            is_case, a = obs[r][i]
            if is_case:
                if case_mode == "base":
                    lower, upper = t_pop, np.inf
                elif case_mode == "interval":
                    lower = float(convert_age_to_thresh(a, pop_prev=K_POP,
                                                        mid_point=MID_POINT,
                                                        slope=SLOPE))
                    upper = np.inf
                else:  # pinned: exact conditioning on T(aoo)
                    lower = upper = float(convert_age_to_thresh(
                        a, pop_prev=K_POP, mid_point=MID_POINT, slope=SLOPE))
                ki, kp = np.nan, np.nan       # observed case: no mixture
            else:
                lower, upper = -np.inf, float(convert_age_to_thresh(
                    a, pop_prev=K_POP, mid_point=MID_POINT, slope=SLOPE))
                ki, kp = float(cip(a)), K_POP
            members.append(Member(role=r, lower=lower, upper=upper,
                                  K_i=ki, K_pop=kp))
        families.append(Family(fam_id=i, members=members))
    return families


def metrics(est, true_g):
    corr = float(np.corrcoef(est, true_g)[0, 1])
    slope = float(np.polyfit(est, true_g, 1)[0])
    return corr, slope


def run_arm(families, true_g, use_mixture, gibbs_subset=None):
    est = estimate_liability(families, h2=H2, use_mixture=use_mixture)
    pa = np.asarray(est.est["genetic"] if hasattr(est, "est") else est.genetic)
    out = {"pa": metrics(pa, true_g)}
    if gibbs_subset is not None:
        g = estimate_liability(families[:gibbs_subset], h2=H2, method="gibbs",
                               tol=0.03, n_sim=25_000, burn_in=800, seed=1)
        ge = np.asarray(g.est["genetic"] if hasattr(g, "est") else g.genetic)
        out["gibbs"] = metrics(ge, true_g[:gibbs_subset])
        out["pa_gibbs_corr"] = float(np.corrcoef(pa[:gibbs_subset], ge)[0, 1])
    return out


def _mean_ci(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return np.nan, np.nan, np.nan, np.nan, 0
    mean = float(values.mean())
    if values.size == 1:
        return mean, np.nan, np.nan, np.nan, 1
    sd = float(values.std(ddof=1))
    se = sd / np.sqrt(values.size)
    ci95 = float(stats.t.ppf(0.975, values.size - 1) * se)
    return mean, sd, float(se), ci95, int(values.size)


def _contrast_rows(replicate_rows):
    """Paired mixture-minus-no-mixture differences per cell and case encoding."""
    out = []
    cells = []
    for row in replicate_rows:
        cell = (row["onset_model"], row["regime"])
        if cell not in cells:
            cells.append(cell)
    for onset_model, regime in cells:
        cell_rows = [row for row in replicate_rows
                     if row["onset_model"] == onset_model
                     and row["regime"] == regime]
        for case_mode in sorted({row["case_mode"] for row in cell_rows}):
            nomix = {row["rep"]: row for row in cell_rows
                     if row["case_mode"] == case_mode and not row["use_mixture"]}
            mix = {row["rep"]: row for row in cell_rows
                   if row["case_mode"] == case_mode and row["use_mixture"]}
            paired_reps = sorted(set(nomix) & set(mix))
            if not paired_reps:
                continue
            base = nomix[paired_reps[0]]
            row = {key: base[key] for key in
                   ("onset_model", "regime", "n_fam", "fam", "h2", "K_pop",
                    "mid", "slope", "reps", "seed")}
            row.update(
                row_type="paired_contrast",
                rep="",
                arm="",
                case_mode=case_mode,
                use_mixture="",
                comparison=f"{mix[paired_reps[0]]['arm']} - {base['arm']}",
                contrast_left=base["arm"],
                contrast_right=mix[paired_reps[0]]["arm"],
                contrast_n=len(paired_reps),
            )
            for metric in ("corr_pa", "slope_pa"):
                delta = [mix[rep][metric] - nomix[rep][metric]
                         for rep in paired_reps]
                mean, sd, se, ci95, _ = _mean_ci(delta)
                row[f"delta_{metric}"] = mean
                row[f"delta_{metric}_sd"] = sd
                row[f"delta_{metric}_se"] = se
                row[f"delta_{metric}_ci95"] = ci95
            out.append(row)
    return out


def write_csv(rows, output_prefix):
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path = f"{output_prefix}.csv"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-fam", type=int, default=N_FAM)
    parser.add_argument("--reps", type=int, default=REPS)
    parser.add_argument("--gibbs-n", type=int, default=GIBBS_N,
                        help="subset size for the Gibbs cross-check; 0 disables")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX,
                        help="path stem for the .csv output")
    args = parser.parse_args()

    t0 = time.time()
    print("PA-FGRS censoring-mixture generative validation")
    print(f"seed={args.seed}  h2={H2}  K_pop={K_POP}  CIP logistic(mid={MID_POINT},"
          f" slope={SLOPE})  fams={args.n_fam} reps={args.reps}  fam_vec={FAM_VEC}")
    print("truth = simulated genetic liability; slope = regress(true ~ est)"
          " (1.0 = calibrated); +/- is across-replicate SE\n")

    grids = [
        ("CROSSING", "crossing",
         [("base + no-mixture", "base", False),
          ("base + mixture", "base", True),
          ("interval + mixture", "interval", True),
          ("pinned + no-mixture", "pinned", False),
          ("pinned + mixture", "pinned", True)]),
        ("STOCHASTIC-ONSET", "stochastic",
         [("base + no-mixture", "base", False),
          ("base + mixture", "base", True)]),
        ("LIABILITY-DEPENDENT (rho=0.6)", "dependent",
         [("base + no-mixture", "base", False),
          ("base + mixture", "base", True),
          ("pinned + no-mixture", "pinned", False),
          ("pinned + mixture", "pinned", True)]),
    ]
    replicate_rows = []
    for model_label, onset_model, arms in grids:
        print(f"===== observation model: {model_label} =====")
        for regime in ("mid", "old"):
            cell_rows = []
            for rep in range(args.reps):
                rng = np.random.default_rng(np.random.PCG64(args.seed + 31 * rep))
                liab, roles, obs = simulate_cohort(rng, regime, args.n_fam,
                                                   onset_model=onset_model)
                true_g = liab[:, roles.index("g")]
                for name, case_mode, use_mix in arms:
                    fams = build_families(obs, roles, args.n_fam, case_mode)
                    gibbs = (args.gibbs_n if name.startswith("base + no")
                             and args.gibbs_n else None)
                    res = run_arm(fams, true_g, use_mix, gibbs_subset=gibbs)
                    row = dict(
                        row_type="replicate",
                        onset_model=onset_model,
                        regime=regime,
                        rep=rep,
                        arm=name,
                        case_mode=case_mode,
                        use_mixture=int(use_mix),
                        comparison="",
                        n_fam=args.n_fam,
                        fam="+".join(FAM_VEC),
                        h2=H2,
                        K_pop=K_POP,
                        mid=MID_POINT,
                        slope=SLOPE,
                        reps=args.reps,
                        seed=args.seed,
                        corr_pa=res["pa"][0],
                        slope_pa=res["pa"][1],
                        corr_gibbs=np.nan,
                        slope_gibbs=np.nan,
                        corr_pa_gibbs=np.nan,
                    )
                    if "gibbs" in res:
                        row.update(corr_gibbs=res["gibbs"][0],
                                   slope_gibbs=res["gibbs"][1],
                                   corr_pa_gibbs=res["pa_gibbs_corr"])
                    replicate_rows.append(row)
                    cell_rows.append(row)
            print(f"--- {regime.upper()} regime "
                  f"({'heavy' if regime == 'mid' else 'light'} censoring) ---")
            print(f"  {'arm':22s} {'corr(PA)':>17s} {'slope(PA)':>17s} "
                  f"{'corr(Gibbs)':>12s} {'slope(Gibbs)':>12s} {'corr(PA,Gibbs)':>14s}")
            for name, _, _ in arms:
                sub = [row for row in cell_rows if row["arm"] == name]
                pa_c, _, pa_c_se, _, _ = _mean_ci([r["corr_pa"] for r in sub])
                pa_s, _, pa_s_se, _, _ = _mean_ci([r["slope_pa"] for r in sub])
                gibbs_rows = [r for r in sub if np.isfinite(r["corr_gibbs"])]
                if gibbs_rows:
                    g_c = np.mean([r["corr_gibbs"] for r in gibbs_rows])
                    g_s = np.mean([r["slope_gibbs"] for r in gibbs_rows])
                    pg = np.mean([r["corr_pa_gibbs"] for r in gibbs_rows])
                    print(f"  {name:22s} {pa_c:7.4f} ± {pa_c_se:.4f} "
                          f"{pa_s:7.4f} ± {pa_s_se:.4f} {g_c:12.4f} "
                          f"{g_s:12.4f} {pg:14.4f}")
                else:
                    print(f"  {name:22s} {pa_c:7.4f} ± {pa_c_se:.4f} "
                          f"{pa_s:7.4f} ± {pa_s_se:.4f} {'-':>12s} "
                          f"{'-':>12s} {'-':>14s}")
            print()

    contrast_rows = _contrast_rows(replicate_rows)
    print("paired contrasts (mixture - no-mixture per replicate; "
          "mean ± SE, t-based 95% CI half-width)")
    for row in contrast_rows:
        print(f"  {row['onset_model']}/{row['regime']} {row['comparison']}: "
              f"Δcorr={row['delta_corr_pa']:+.5f} ± {row['delta_corr_pa_se']:.5f} "
              f"(CI95 ±{row['delta_corr_pa_ci95']:.5f}); "
              f"Δslope={row['delta_slope_pa']:+.5f} ± {row['delta_slope_pa_se']:.5f} "
              f"(CI95 ±{row['delta_slope_pa_ci95']:.5f})")

    csv_path = write_csv(replicate_rows + contrast_rows, args.output_prefix)
    print(f"\nwrote {os.path.basename(csv_path)}")
    print(f"runtime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
