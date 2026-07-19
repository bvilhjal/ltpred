"""Generative validation of the PA-FGRS age-censored-control mixture.

The PA-FGRS censoring mixture (`use_mixture=True`; Dybdahl Krebs et al. 2024)
ships with unit tests but no generative benchmark (ROADMAP: "mixture validation
remains deferred"; benchmarks/README over-billed PA-FGRS coverage). This is the
deferred benchmark: simulate families under the liability-threshold model with
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

Run:  conda run -n ltpred python benchmarks/bench_pafgrs_mixture.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ltpred.covariance import construct_covmat_single, correct_positive_definite  # noqa: E402
from ltpred.estimate import estimate_liability  # noqa: E402
from ltpred.family import Family, Member  # noqa: E402
from ltpred.thresholds import (liability_threshold, convert_liability_to_aoo,  # noqa: E402
                               convert_age_to_thresh)

SEED = 20260719
H2 = 0.5
K_POP = 0.10
MID_POINT = 60.0
SLOPE = 1.0 / 8.0
FAM_VEC = ["m", "f", "s1"]
N_FAM = 20_000
REPS = 5
GIBBS_N = 2_000          # smaller subset for the sampling reference
AGE_RANGES = {           # (low, high) current-age ranges per regime
    "mid": {"o": (35, 55), "s": (35, 55), "m": (55, 75), "f": (55, 75)},
    "old": {"o": (60, 85), "s": (60, 85), "m": (75, 95), "f": (75, 95)},
}


def cip(age):
    """Logistic cumulative incidence proportion at ``age``."""
    return K_POP / (1.0 + np.exp((MID_POINT - np.asarray(age, dtype=float)) * SLOPE))


def simulate_cohort(rng, regime, n_fam, onset_model="crossing"):
    """Liabilities + honest censoring; returns truth and per-member observations.

    onset_model='crossing': onset age is the deterministic threshold-crossing
    map. 'stochastic': case status is l > T_pop and onset age is drawn from the
    CIP (inverse-CDF of K(age)/K_pop) independent of liability."""
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

    obs = {r: [] for r in roles[1:]}
    for i in range(n_fam):
        for r in roles[1:]:
            col = roles.index(r)
            l = liab[i, col]
            age = ages[r][i]
            if onset_model == "crossing":
                aoo = convert_liability_to_aoo(l, pop_prev=K_POP,
                                               mid_point=MID_POINT, slope=SLOPE)
                aoo = 0.0 if not np.isfinite(aoo) else float(aoo)
                is_case = (l > t_pop) and (aoo <= age)
            else:  # stochastic onset for lifetime cases only
                if l > t_pop:
                    u = float(rng.uniform())
                    aoo = MID_POINT + np.log(u / (1.0 - u)) / SLOPE
                    aoo = max(0.0, aoo)
                    is_case = aoo <= age
                else:
                    is_case = False
                    aoo = age
            obs[r].append((is_case, aoo if is_case else age))
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


def main():
    t0 = time.time()
    print("PA-FGRS censoring-mixture generative validation")
    print(f"seed={SEED}  h2={H2}  K_pop={K_POP}  CIP logistic(mid={MID_POINT},"
          f" slope={SLOPE})  fams={N_FAM} reps={REPS}  fam_vec={FAM_VEC}")
    print("truth = simulated genetic liability; slope = regress(true ~ est)"
          " (1.0 = calibrated)\n")

    grids = [
        ("CROSSING", "crossing",
         [("base + no-mixture", "base", False),
          ("base + mixture   ", "base", True),
          ("interval + mixture", "interval", True),
          ("pinned + no-mixture", "pinned", False),
          ("pinned + mixture", "pinned", True)]),
        ("STOCHASTIC-ONSET", "stochastic",
         [("base + no-mixture", "base", False),
          ("base + mixture   ", "base", True)]),
    ]
    for model_label, onset_model, arms in grids:
        print(f"===== observation model: {model_label} =====")
        for regime in ("mid", "old"):
            results = {name: {"pa": [], "gibbs": [], "pa_gibbs_corr": []}
                       for name, _, _ in arms}
            for rep in range(REPS):
                rng = np.random.default_rng(np.random.PCG64(SEED + 31 * rep))
                liab, roles, obs = simulate_cohort(rng, regime, N_FAM,
                                                   onset_model=onset_model)
                true_g = liab[:, roles.index("g")]
                for name, case_mode, use_mix in arms:
                    fams = build_families(obs, roles, N_FAM, case_mode)
                    res = run_arm(fams, true_g, use_mix,
                                  gibbs_subset=GIBBS_N if name.startswith("base + no")
                                  else None)
                    for k in results[name]:
                        if k in res:
                            results[name][k].append(res[k])
            print(f"--- {regime.upper()} regime "
                  f"({'heavy' if regime == 'mid' else 'light'} censoring) ---")
            print(f"  {'arm':22s} {'corr(PA)':>9s} {'slope(PA)':>10s} "
                  f"{'corr(Gibbs)':>12s} {'slope(Gibbs)':>12s} {'corr(PA,Gibbs)':>14s}")
            for name, _, _ in arms:
                r = results[name]
                pa_c = np.mean([x[0] for x in r["pa"]])
                pa_s = np.mean([x[1] for x in r["pa"]])
                if r["gibbs"]:
                    g_c = np.mean([x[0] for x in r["gibbs"]])
                    g_s = np.mean([x[1] for x in r["gibbs"]])
                    pg = np.mean(r["pa_gibbs_corr"])
                    print(f"  {name:22s} {pa_c:9.4f} {pa_s:10.4f} {g_c:12.4f} "
                          f"{g_s:12.4f} {pg:14.4f}")
                else:
                    print(f"  {name:22s} {pa_c:9.4f} {pa_s:10.4f} {'-':>12s} "
                          f"{'-':>12s} {'-':>14s}")
            print()
    print(f"runtime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
