"""CIP estimation benchmark: recover a known incidence curve, then use it.

`ltpred.cip` estimates cumulative-incidence curves from follow-up records.
This benchmark simulates a registry cohort with a KNOWN true CIP (the repo's
logistic family: lifetime prevalence 0.10, mid-point 60, slope 1/8), plus
mortality (a Gompertz hazard), administrative censoring, and -- in one arm --
delayed entry (register starts mid-life). It then asks:

  Part 1 (no mortality): Kaplan-Meier recovers the true curve -- max abs
         error and the coverage of the Greenwood SE bands (~95% expected).
  Part 2 (mortality):  Aalen-Johansen recovers the true curve while
         Kaplan-Meier (death as independent censoring) OVERESTIMATES
         incidence -- the classic competing-risks bias, quantified.
  Part 3 (delayed entry): the same with left truncation added.
  Part 4 (end-to-end): the estimated curve feeds `thresholds_from_cip` ->
         `estimate_liability` on a family cohort; calibration slope vs the
         oracle-CIP arm (both should sit near 1; cf. the wrong-CIP effect in
         bench_misspecification.py).

Run:  conda run -n ltpred python benchmarks/bench_cip_estimation.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ltpred.cip import aalen_johansen_cip, kaplan_meier_cip  # noqa: E402
from ltpred.covariance import construct_covmat_single, correct_positive_definite  # noqa: E402
from ltpred.estimate import estimate_liability  # noqa: E402
from ltpred.family import Family, Member  # noqa: E402
from ltpred.thresholds import thresholds_from_cip  # noqa: E402

SEED = 20260719
N = 50_000
K_POP = 0.10
MID = 60.0
SLOPE = 1.0 / 8.0
N_PERSONS_E2E = 20_000
FAM_VEC = ["m", "f", "s1"]
H2 = 0.5
REPS_E2E = 3


def true_cip(age):
    return K_POP / (1.0 + np.exp((MID - np.asarray(age, float)) * SLOPE))


def true_crude_cif(age):
    """Cumulative incidence of diagnosis in the PRESENCE of mortality.

    F_crude(t) = integral_0^t f_diag(a) S_death(a) da, with f_diag the logistic
    density of the onset process and S_death the Gompertz survival. This is the
    estimand a health registry measures (the dead cannot be diagnosed), as
    opposed to the marginal no-death-world CIP `true_cip`."""
    a = np.linspace(0, 120, 12001)
    z = np.exp((MID - a) * SLOPE)
    f_diag = K_POP * SLOPE * z / (1.0 + z) ** 2
    haz = np.exp(-9.0 + 0.085 * a)
    s_death = np.exp(-np.cumsum(haz) * (a[1] - a[0]))
    crude = np.cumsum(f_diag * s_death) * (a[1] - a[0])
    return np.interp(age, a, crude)


def draw_onset(rng, n):
    """Onset ages: case with prob K_POP, onset from the CIP inverse CDF."""
    u = rng.uniform(size=n)
    is_case = u < K_POP
    aoo = np.full(n, np.inf)
    v = rng.uniform(size=int(is_case.sum()))
    aoo[is_case] = MID + np.log(v / (1 - v)) / SLOPE
    return is_case, np.maximum(aoo, 0.0)


def draw_death(rng, n):
    """Gompertz mortality: hazard h(a) = exp(-9 + 0.085 a)."""
    u = rng.uniform(size=n)
    return np.log(1.0 + 0.085 * (-np.log(u)) / np.exp(-9.0)) / 0.085


def simulate_registry(rng, n, mortality, register_start_year=None):
    """Per-person follow-up: entry age, exit age, event (0 censor, 1 diag, 2 death)."""
    birth_year = rng.uniform(1900, 2000, n)
    admin_end = 2015.0
    entry_age = np.zeros(n)
    if register_start_year is not None:
        entry_age = np.maximum(0.0, register_start_year - birth_year)
    is_case, aoo = draw_onset(rng, n)
    death = draw_death(rng, n) if mortality else np.full(n, np.inf)
    admin = np.maximum(entry_age, admin_end - birth_year)
    exit_age = np.minimum(np.minimum(aoo, death), admin)
    event = np.zeros(n, dtype=int)
    event[(aoo <= exit_age) & np.isfinite(aoo)] = 1
    event[(death <= exit_age) & (aoo > death)] = 2
    ok = exit_age > entry_age
    return entry_age[ok], exit_age[ok], event[ok]


def curve_error(curve, grid_eval):
    est = np.interp(grid_eval, curve.ages, curve.values)
    truth = true_cip(grid_eval)
    return np.max(np.abs(est - truth)), est, truth


def main():
    t0 = time.time()
    print("CIP estimation benchmark (known true logistic CIP)")
    print(f"seed={SEED}  N={N}  K_pop={K_POP}  mid={MID} slope={SLOPE}")
    grid_eval = np.arange(20, 81, 5, dtype=float)

    # ---- Part 1: KM without mortality ---------------------------------------
    rng = np.random.default_rng(np.random.PCG64(SEED))
    en, ex, ev = simulate_registry(rng, N, mortality=False)
    km = kaplan_meier_cip(en, ex, ev == 1)
    err, est, truth = curve_error(km, grid_eval)
    cover = np.mean(np.abs(est - truth) < 2 * np.interp(grid_eval, km.ages, km.se))
    print(f"\nPart 1 (no mortality, n={km.n_entered}, events={km.n_events}): "
          f"max|est-true| {err:.4f}, Greenwood 2xSE coverage {cover:.2f}")

    # ---- Part 2: AJ vs KM with mortality ------------------------------------
    # Two different estimands: the marginal no-death-world CIP (true_cip) and
    # the crude cumulative incidence in the presence of death (true_crude_cif).
    rng = np.random.default_rng(np.random.PCG64(SEED + 1))
    en, ex, ev = simulate_registry(rng, N, mortality=True)
    aj = aalen_johansen_cip(en, ex, ev, n_boot=100, seed=1)
    km_bad = kaplan_meier_cip(en, ex, ev == 1)
    crude_true = true_crude_cif(grid_eval)
    err_aj = np.max(np.abs(np.interp(grid_eval, aj.ages, aj.values) - crude_true))
    err_km_crude = np.max(np.abs(np.interp(grid_eval, km_bad.ages, km_bad.values)
                                 - crude_true))
    err_km_marg = np.max(np.abs(np.interp(grid_eval, km_bad.ages, km_bad.values)
                                - true_cip(grid_eval)))
    death_frac = np.mean(ev == 2)
    print(f"Part 2 (mortality, death share {death_frac:.2f}):")
    print(f"  AJ vs true crude CIF:       max err {err_aj:.4f}  (its estimand)")
    print(f"  KM vs true crude CIF:       max err {err_km_crude:.4f}  "
          f"<- competing-risks overestimation")
    print(f"  KM vs marginal no-death CIP: max err {err_km_marg:.4f}  "
          f"(KM is consistent for its own, different estimand)")

    # ---- Part 3: delayed entry ----------------------------------------------
    rng = np.random.default_rng(np.random.PCG64(SEED + 2))
    en, ex, ev = simulate_registry(rng, N, mortality=True,
                                   register_start_year=1995)
    aj_lt = aalen_johansen_cip(en, ex, ev, n_boot=100, seed=1)
    err_lt = np.max(np.abs(np.interp(grid_eval, aj_lt.ages, aj_lt.values)
                           - crude_true))
    print(f"Part 3 (delayed entry from 1995, n={aj_lt.n_entered}): "
          f"AJ vs true crude CIF max err {err_lt:.4f}")

    # ---- Part 4: end-to-end into estimate_liability --------------------------
    print("\nPart 4 (end-to-end): estimated CIP -> thresholds_from_cip ->"
          " estimate_liability")
    from scipy.stats import norm
    cov_obj = construct_covmat_single(fam_vec=FAM_VEC, add_ind=True, h2=H2)
    cov, _ = correct_positive_definite(cov_obj.matrix)
    roles = cov_obj.roles
    age_grid = np.arange(0, 121, 1.0)
    true_curve = true_cip(age_grid)
    # the crossing-model families are generated WITHOUT mortality, so the
    # estimand-consistent estimated curve is the Part-1 KM curve; its horizon
    # value wobbles around the truth by sampling error, so use the curve's own
    # horizon as k_pop (the thresholds_from_cip default)
    curves = {"oracle": (age_grid, true_curve, K_POP),
            "estimated": (km.ages, km.values, None)}
    for label, (c_ages, c_vals, kpop) in curves.items():
        slopes, corrs = [], []
        for rep in range(REPS_E2E):
            rng = np.random.default_rng(np.random.PCG64(SEED + 900 + rep))
            liab = rng.multivariate_normal(np.zeros(len(roles)), cov,
                                           size=N_PERSONS_E2E)
            fams = []
            for i in range(N_PERSONS_E2E):
                members = []
                for k, r in enumerate(roles[1:]):
                    l = liab[i, k + 1]
                    age = int(rng.integers(35, 80))
                    # crossing model: onset when the true CIP reaches the liability
                    need = 1.0 - norm.cdf(l)          # = Phi(-l); case iff < K_POP
                    aoo = float(np.interp(need, true_curve, age_grid))
                    is_case = (need < K_POP) and (aoo <= age)
                    lo, hi, _, _ = thresholds_from_cip(
                        np.array([is_case]),
                        np.array([aoo if is_case else float(age)]),
                        c_ages, c_vals, k_pop=kpop)
                    members.append(Member(role=r, lower=float(lo[0]),
                                          upper=float(hi[0])))
                fams.append(Family(fam_id=i, members=members))
            est = estimate_liability(fams, h2=H2)
            est = np.asarray(est.est["genetic"] if hasattr(est, "est") else est.genetic)
            g = liab[:, 0]
            slopes.append(float(np.polyfit(est, g, 1)[0]))
            corrs.append(float(np.corrcoef(est, g)[0, 1]))
        print(f"  {label:10s} slope {np.mean(slopes):.4f}  corr {np.mean(corrs):.4f}")

    print(f"\nruntime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
