"""Stress PA scoring under one model violation at a time.

Against a Gaussian control, arms introduce heavy-tailed residual environment,
assortative mating, unmodelled sibship environment and wrong prevalence.
Independent cohorts report correlation with true genetic liability and
calibration slope. The default design uses h2=0.5, parents plus one sib,
8,000 families and five replicates. Results are diagnostics, not acceptance
gates. Writes `bench_misspecification.csv`.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ltpred.estimate import estimate_liability  # noqa: E402
from ltpred.family import Family, Member  # noqa: E402
from ltpred.thresholds import liability_threshold  # noqa: E402

SEED = 20260719
H2 = 0.5
PREV_TRUE = 0.10
N_FAM = 8_000
REPS = 5
ROLES = ["o", "m", "f", "s1"]
# additive relationship matrix among [o, m, f, s1] (mates unrelated)
A_BASE = np.array([
    [1.0, 0.5, 0.5, 0.5],
    [0.5, 1.0, 0.0, 0.5],
    [0.5, 0.0, 1.0, 0.5],
    [0.5, 0.5, 0.5, 1.0],
])


def simulate(rng, arm, n_fam, prev):
    """Return (true genetic liability of proband, full liabilities, status)."""
    A = A_BASE.copy()
    if arm == "assortative":
        A[1, 2] = A[2, 1] = 0.3           # mate genetic correlation
    G = rng.multivariate_normal(np.zeros(4), H2 * A, size=n_fam)

    if arm == "heavy-tail env":
        # t_5 scaled to unit variance, independent per member
        e = rng.standard_t(df=5, size=(n_fam, 4)) / np.sqrt(5.0 / 3.0)
        e *= np.sqrt(1.0 - H2)
    elif arm == "assortative":
        # mates' environmental parts correlated at rho_e = 0.3
        z = rng.standard_normal((n_fam, 2))
        e = rng.standard_normal((n_fam, 4)) * np.sqrt(1.0 - H2)
        e[:, 1] = z[:, 0] * np.sqrt(1 - H2)
        e[:, 2] = (0.3 * z[:, 0] + np.sqrt(1 - 0.3 ** 2) * z[:, 1]) * np.sqrt(1 - H2)
    else:
        e = rng.standard_normal((n_fam, 4)) * np.sqrt(1.0 - H2)

    L = G + e
    if arm == "sibship env":
        c2 = 0.15
        c = rng.standard_normal(n_fam) * np.sqrt(c2)
        L[:, 0] += c
        L[:, 3] += c
        # keep Var(l) = 1 for o and s1 by shrinking their independent env
        L[:, [0, 3]] = G[:, [0, 3]] + c[:, None] + e[:, [0, 3]] * np.sqrt(
            (1 - H2 - c2) / (1 - H2))

    t = float(liability_threshold(prev))
    status = L > t
    return G[:, 0], L, status


def estimate(true_g, L, status, prev_assumed):
    t = float(liability_threshold(prev_assumed))
    families = []
    n_fam = L.shape[0]
    for i in range(n_fam):
        members = []
        for k, role in enumerate(ROLES):
            lo, hi = (t, np.inf) if status[i, k] else (-np.inf, t)
            members.append(Member(role=role, lower=lo, upper=hi))
        families.append(Family(fam_id=i, members=members))
    est = estimate_liability(families, h2=H2)
    est = np.asarray(est.est["genetic"])
    corr = float(np.corrcoef(est, true_g)[0, 1])
    slope = float(np.polyfit(est, true_g, 1)[0])
    return corr, slope


def main():
    # This benchmark takes no options. Parse anyway, so that `--help`
    # prints the docstring and a stray argument is refused instead of
    # silently ignored while the run reports numbers for a configuration it
    # did not use.
    argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    t0 = time.time()
    print("Model-misspecification stress (PA estimator, classic bounds)")
    print(f"seed={SEED}  h2={H2}  true prevalence={PREV_TRUE}  fams={N_FAM}"
          f"  reps={REPS}  roles={ROLES}")
    arms = ["control", "heavy-tail env", "assortative", "sibship env",
            "wrong prevalence"]
    print(f"  {'arm':24s} {'corr (mean ± SE)':>16s} "
          f"{'slope (mean ± SE)':>16s}")
    for arm in arms:
        if arm == "wrong prevalence":
            for assumed in (0.05, 0.20):
                cs, ss = [], []
                for rep in range(REPS):
                    rng = np.random.default_rng(np.random.PCG64(SEED + 7 * rep))
                    g, L, st = simulate(rng, "control", N_FAM, PREV_TRUE)
                    c, s = estimate(g, L, st, assumed)
                    cs.append(c); ss.append(s)
                print(f"  {arm + ' (' + str(assumed) + ')':24s} "
                      f"{np.mean(cs):6.4f} ± "
                      f"{np.std(cs, ddof=1)/np.sqrt(REPS):.4f}  "
                      f"{np.mean(ss):6.4f} ± "
                      f"{np.std(ss, ddof=1)/np.sqrt(REPS):.4f}")
        else:
            cs, ss = [], []
            for rep in range(REPS):
                rng = np.random.default_rng(np.random.PCG64(SEED + 7 * rep))
                g, L, st = simulate(rng, arm, N_FAM, PREV_TRUE)
                c, s = estimate(g, L, st, PREV_TRUE)
                cs.append(c); ss.append(s)
            print(f"  {arm:24s} {np.mean(cs):6.4f} ± "
                  f"{np.std(cs, ddof=1)/np.sqrt(REPS):.4f}  "
                  f"{np.mean(ss):6.4f} ± "
                  f"{np.std(ss, ddof=1)/np.sqrt(REPS):.4f}")
    print(f"\nruntime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
