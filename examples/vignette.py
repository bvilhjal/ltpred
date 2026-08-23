"""Numbers quoted in docs/vignette.md (seed 1). How to run ltpred.

Pipeline (same numbering as the vignette)::

    0. heritability / covariances
    1. pedigree
    2. CIP or lifetime prevalence
    3. family-history records
    4. estimate_liability
    5. hand the score to a GWAS

This is a teaching script, not a production analysis. The opening block
simulates a nuclear cohort so the later calls have input; on real data,
skip it and start from your table. Swap the logistic CIP helpers for
``thresholds_from_cip`` before a real GWAS.

Run from the repository root::

    python examples/vignette.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ltpred import (  # noqa: E402
    Family, estimate_liability, estimate_liability_from_kinship,
    families_from_columns, fit_heritability, kinship_from_pedigree,
    observed_to_liability_h2, prevalence_thresholds, simulate_under_LTM_single,
    tetrachoric,
)

H2 = 0.5
K = 0.05
N_FAM = 800
SEED = 1


def _corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def _drop_role(families, role):
    return [Family(fam.fam_id, [m for m in fam.members if m.role != role])
            for fam in families]


def _keep_role(families, role):
    return [Family(fam.fam_id, [m for m in fam.members if m.role == role])
            for fam in families]


def main():
    print("== Stand-in cohort (simulated; skip on real data) ==")
    sim = simulate_under_LTM_single(
        fam_vec=["m", "f", "s1"], h2=H2, pop_prev=K, n_sim=N_FAM,
        use_age=False, seed=SEED,
    )
    status = sim.status["o"].astype(float)
    true_g = sim.genetic
    print(f"families {N_FAM}; liability-scale h2 {H2}; prevalence {K}")
    print(f"proband case rate {status.mean():.3f}")

    print("\n== 0. Heritability and covariance ==")
    po = tetrachoric(sim.status["o"].astype(int), sim.status["m"].astype(int))
    print(f"parent-offspring tetrachoric {po.rho:.3f}  "
          f"(Falconer 2*rho = {2 * po.rho:.3f}; truth {H2})")
    obs = float(np.asarray(observed_to_liability_h2(0.20, pop_prev=K)))
    print(f"Lee: observed-scale 0.20 at K={K} -> liability-scale {obs:.3f}")
    print("pass the study case fraction as prop_cases when the sample is ascertained")
    fit = fit_heritability(
        sim.families, n_iter=250, burn_in=80, seed=SEED, sampling="population",
    )
    print(f"fitted h2 {fit.h2:.3f}  (truth {H2}; within-dataset MC se {fit.h2_se:.4f})")
    print("the MC se is not a sampling interval — a few hundred nuclear families")
    print("at K=0.05 leave a much larger across-cohort SD. Use an external h2")
    print("unless the sampling contract in docs/inference.md holds.")

    print("\n== 1. Pedigree ==")
    ids = ["o", "m", "f", "s1"]
    _, A = kinship_from_pedigree(
        ids, father=["f", None, None, "f"], mother=["m", None, None, "m"],
    )
    print(f"role grammar: o, m, f, s1  |  kinship A shape {A.shape}")
    print(f"A[o,m]={A[0, 1]:.2f}  A[o,s1]={A[0, 3]:.2f}  A[m,f]={A[1, 2]:.2f}")
    print("cousins / inbreeding / messy half-sibs: kinship_from_pedigree "
          "or extract_pedigree, not extra role labels")

    print("\n== 2. CIP / lifetime prevalence ==")
    fam_id, role, st, lower_l, upper_l = [], [], [], [], []
    for i, fam in enumerate(sim.families):
        for m in fam.members:
            fam_id.append(f"fam{i}")
            role.append(m.role)
            is_case = np.isfinite(m.lower)
            st.append(int(is_case))
            lower_l.append(m.lower)
            upper_l.append(m.upper)
    lower, upper = prevalence_thresholds(np.array(st), pop_prev=K)
    print(f"classic LT-FH: one K={K} -> T = Phi^{{-1}}(1-K); "
          f"{len(st)} person-rows")
    print("LT-FH++ / ADuLT: thresholds_from_cip with a population CIP,")
    print("stratified by sex / birth year / ancestry; not this one-K map")

    print("\n== 3. Family-history records ==")
    rebuilt = families_from_columns(fam_id, role, lower, upper)
    print(f"families_from_columns: {len(rebuilt)} families, "
          f"roles {sorted({r for r in role})}")
    print("include role o when mu is a GWAS phenotype of this diagnosis;")
    print("omit o (or unbind it) when the same diagnosis is the prediction target")

    print("\n== 4. Estimate mu ==")
    pa = estimate_liability(sim.families, h2=H2)
    score = pa.genetic
    print(f"score mean {score.mean():.3f}  sd {score.std():.3f}")
    print(f"PA se is identically 0 (deterministic): {pa.se['genetic'][0]:.1f}")
    i_case = int(np.flatnonzero(status)[0])
    i_ctrl = int(np.flatnonzero(1.0 - status)[0])
    print(f"one case    score {score[i_case]:+.3f}  posterior var {pa.var['genetic'][i_case]:.3f}")
    print(f"one control score {score[i_ctrl]:+.3f}  posterior var {pa.var['genetic'][i_ctrl]:.3f}")
    r_status = _corr(status, true_g)
    r_pa = _corr(score, true_g)
    print(f"corr(case/control, true g) {r_status:.3f}")
    print(f"corr(LT-FH PA,     true g) {r_pa:.3f}")
    print(f"squared-corr eff-N proxy   {(r_pa / r_status) ** 2:.2f}x")
    rel = estimate_liability(_drop_role(sim.families, "o"), h2=H2)
    print(f"corr(relatives-only, true g) {_corr(rel.genetic, true_g):.3f}")
    adult = estimate_liability(_keep_role(sim.families, "o"), h2=H2)
    print(f"corr(ADuLT, true g) {_corr(adult.genetic, true_g):.3f}")
    print("(classic LT-FH ADuLT is just a case/control-to-liability map)")
    n_check = 80
    gibbs = estimate_liability(
        sim.families[:n_check], h2=H2, method="gibbs",
        tol=0.03, n_sim=8_000, burn_in=400, seed=SEED,
    )
    print(f"corr(PA, Gibbs) on {n_check} families "
          f"{_corr(score[:n_check], gibbs.genetic):.4f}")
    print(f"Gibbs MC se (median) {np.median(gibbs.se['genetic']):.4f}")
    rebuilt_score = estimate_liability(rebuilt, h2=H2).genetic
    print(f"max |rebuilt - original| {np.max(np.abs(rebuilt_score - score)):.2e}")
    lo = np.empty((N_FAM, 4))
    hi = np.empty((N_FAM, 4))
    for i, fam in enumerate(sim.families):
        by = {m.role: m for m in fam.members}
        for j, r in enumerate(ids):
            lo[i, j] = by[r].lower
            hi[i, j] = by[r].upper
    kin, kin_se, _kin_var = estimate_liability_from_kinship(A, lo, hi, h2=H2)
    print(f"max |kinship PA - role PA| {np.max(np.abs(kin - score)):.2e}  "
          "(PA fold order can differ; this is not Gibbs Monte-Carlo error)")
    print(f"kinship PA se is 0: {kin_se[0]:.1f}")

    print("\n== 5. Ready for GWAS ==")
    print(f"{len(pa.pids)} pids; join to genotyped IDs, residualize, GWAS elsewhere")
    print("ltpred does not run the association scan")

    print("\n== On real data, step 2 is a CIP, not one K ==")
    sim_age = simulate_under_LTM_single(
        fam_vec=["m", "f", "s1"], h2=H2, pop_prev=K, n_sim=N_FAM,
        use_age=True, seed=SEED,
    )
    pa_age = estimate_liability(sim_age.families, h2=H2)
    r_age = _corr(pa_age.genetic, sim_age.genetic)
    r_age_cc = _corr(sim_age.status["o"].astype(float), sim_age.genetic)
    print(f"corr(age-aware PA, true g) {r_age:.3f}  "
          f"(case/control {r_age_cc:.3f})")
    print("real LT-FH++ needs thresholds_from_cip with stratified population CIPs")
    print("done.")


if __name__ == "__main__":
    main()
