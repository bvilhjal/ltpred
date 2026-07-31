"""End-to-end register-pipeline benchmark.

`research.pipeline.estimate_liabilities` chains trio records -> pedigree
discovery -> CIP thresholds -> per-proband genetic-liability scores. This
benchmark builds a synthetic register (3-generation population, remarriages;
a known logistic CIP; liabilities from the extracted pedigrees; onsets by
threshold crossing) and measures the pipeline end to end:

  Part 1 (accuracy):    corr(est, true g) and calibration slope at degree 3
         (first cousins included) vs degree 1 (first-degree only) -- the
         "which relatives you include matters" effect, through the driver.
  Part 2 (CIP from data): the same pipeline with the CIP ESTIMATED from the
         register itself (aalen_johansen_cip on the follow-up records) vs the
         oracle curve.
  Part 3 (prospective): prospective prediction of the proband's future
         diagnosis after index age 40 -- (a) the correct familywise-censored
         score, (b) a leaky arm that also conditions on post-index relative
         events, (c) a leaky arm that conditions on the proband's own future
         outcome. Shows both the accuracy cost of honest censoring and the
         inflation leakage buys.
  Part 4 (throughput):  probands/s for the full pipeline.

Run:  conda run -n ltpred python benchmarks/bench_register_pipeline.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))

from bench_pedigree_inference import simulate_population  # noqa: E402
from ltpred.cip import aalen_johansen_cip  # noqa: E402
from ltpred.covariance import kinship_from_pedigree  # noqa: E402
from research.pipeline import estimate_liabilities  # noqa: E402
from scipy.stats import norm  # noqa: E402

SEED = 20260719
H2 = 0.5
K_POP = 0.10
MID = 60.0
SLOPE = 1.0 / 8.0
N_PROBANDS = 400
INDEX_AGE = 40.0
EVAL_AGE = 70.0

AGE_GRID = np.arange(0, 121, 1.0)
TRUE_CIP = K_POP / (1.0 + np.exp((MID - AGE_GRID) * SLOPE))


def build_register(rng, ids, father, mother, probands):
    """One CONSISTENT liability field for the whole population, then records.

    Draws genetic liabilities G ~ N(0, h2 * A_full) once and full liabilities
    L = G + E with independent E, so every proband-relative correlation is
    exact. (Drawing each proband's pedigree independently would fix a shared
    person's status from one draw and the proband's g from another, silently
    decorrelating them.) Onsets follow the threshold-crossing model with the
    true CIP; records are (status, age, onset, true_g) aligned to ids."""
    _, A_full = kinship_from_pedigree(ids, father, mother)
    n = len(ids)
    G = rng.multivariate_normal(np.zeros(n), H2 * A_full)
    E = rng.standard_normal(n) * np.sqrt(1.0 - H2)
    L = G + E
    status = np.zeros(n, dtype=bool)
    age = np.zeros(n)
    onset = np.full(n, np.nan)
    for k in range(n):
        age_now = float(rng.integers(20, 80))
        need = 1.0 - norm.cdf(L[k])
        aoo = float(np.interp(need, TRUE_CIP, AGE_GRID))
        onset[k] = aoo
        status[k] = (need < K_POP) and (aoo <= age_now)
        age[k] = aoo if status[k] else age_now
    pos = {p: i for i, p in enumerate(ids)}
    true_g = np.array([G[pos[p]] for p in probands])
    return status, age, onset, true_g


def main():
    t0 = time.time()
    print("Register-pipeline benchmark (trio records -> scores)")
    rng = np.random.default_rng(np.random.PCG64(SEED))
    ids, father, mother = simulate_population(rng)
    probands = [ids[i] for i in rng.choice(len(ids), size=N_PROBANDS,
                                           replace=False)]
    status, age, onset, true_g = build_register(rng, ids, father, mother,
                                                probands)
    pos = {p: i for i, p in enumerate(ids)}
    g = true_g                                   # already aligned to probands
    print(f"population {len(ids)}, probands {N_PROBANDS}, h2={H2}")

    def summarize(est, tag):
        c = np.corrcoef(g, est)[0, 1]
        s = np.polyfit(est, g, 1)[0]
        print(f"  {tag:34s} corr {c:.4f}  slope {s:.4f}")
        return c, s

    # ---- Part 1: oracle CIP, degree 3 vs degree 1 -----------------------------
    print("\nPart 1: accuracy, oracle CIP")
    out3 = estimate_liabilities(ids, father, mother, probands=probands,
                                status=status, age=age, cip_ages=AGE_GRID,
                                cip_values=TRUE_CIP, h2=H2, max_degree=3)
    summarize(out3.est, "degree 3 (cousins)")
    out1 = estimate_liabilities(ids, father, mother, probands=probands,
                                status=status, age=age, cip_ages=AGE_GRID,
                                cip_values=TRUE_CIP, h2=H2, max_degree=1)
    summarize(out1.est, "degree 1 (first-degree only)")

    # ---- Part 2: estimated CIP --------------------------------------------------
    print("\nPart 2: CIP estimated from the register vs oracle")
    entry = np.zeros(len(ids))
    exit_age = np.where(status, age, age)
    event = status.astype(int)
    curve = aalen_johansen_cip(entry, exit_age, event)
    out_est = estimate_liabilities(ids, father, mother, probands=probands,
                                   status=status, age=age,
                                   cip_ages=curve.ages, cip_values=curve.values,
                                   h2=H2, max_degree=3)
    summarize(out_est.est, "estimated CIP (degree 3)")
    summarize(out3.est, "oracle CIP (degree 3)")

    # ---- Part 3: prospective prediction ---------------------------------------
    print("\nPart 3: prospective prediction of diagnosis after index age",
          INDEX_AGE)
    future = np.array([(INDEX_AGE < onset[pos[p]] <= EVAL_AGE)
                       for p in probands])
    ia = np.full(N_PROBANDS, INDEX_AGE)
    out_pro = estimate_liabilities(ids, father, mother, probands=probands,
                                   status=status, age=age, cip_ages=AGE_GRID,
                                   cip_values=TRUE_CIP, h2=H2, index_age=ia)
    # leaky arm (b): relatives' post-index events kept (no effective
    # censoring: index age set to +inf) but proband bound uninformative
    out_leak_rel = estimate_liabilities(ids, father, mother, probands=probands,
                                        status=status, age=age,
                                        cip_ages=AGE_GRID, cip_values=TRUE_CIP,
                                        h2=H2, index_age=np.full(N_PROBANDS, 1e9))
    # leaky arm (c): the proband's own future outcome included (full history
    # including own diagnosis at any age)
    out_leak_own = estimate_liabilities(ids, father, mother, probands=probands,
                                        status=status, age=age,
                                        cip_ages=AGE_GRID, cip_values=TRUE_CIP,
                                        h2=H2, max_degree=3)
    for tag, est in (("(a) familywise-censored (honest)", out_pro.est),
                     ("(b) + relatives' post-index events", out_leak_rel.est),
                     ("(c) + proband's own future outcome", out_leak_own.est)):
        c = np.corrcoef(future.astype(float), est)[0, 1]
        print(f"  {tag:34s} corr(score, future case) {c:.4f}")

    # ---- Part 4: throughput -----------------------------------------------------
    print("\nPart 4: throughput")
    t1 = time.time()
    estimate_liabilities(ids, father, mother, probands=probands,
                         status=status, age=age, cip_ages=AGE_GRID,
                         cip_values=TRUE_CIP, h2=H2, max_degree=3)
    dt = time.time() - t1
    print(f"  {N_PROBANDS} probands in {dt:.2f}s "
          f"({N_PROBANDS / dt:,.0f} probands/s)")

    print(f"\nruntime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
