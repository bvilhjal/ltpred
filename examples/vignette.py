"""Numbers quoted in docs/vignette.md (seed 1). How to run ltpred.

Pipeline (same numbering as the vignette)::

    0. heritability / covariances
    1. pedigree
    2. CIP or lifetime prevalence
    3. family-history records
    4. estimate_liability  (+ the "did it work?" checks)
    5. prediction (I) and/or GWAS (II); aetiology (III) may stop at 0/2

This is a teaching script, not a production analysis. The opening block
simulates a nuclear cohort so the later calls have input; on real data,
skip it and start from your table. Swap the logistic CIP helpers for
``thresholds_from_cip`` before a real GWAS.

The key outputs are quoted in docs/vignette.md. If you change the script,
re-read the vignette. A final six-person register example demonstrates the
public driver and calendar-time prediction without a performance claim.

Run from the repository root::

    python examples/vignette.py
"""
from __future__ import annotations

import os
import sys
from dataclasses import replace

import numpy as np
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ltpred import (  # noqa: E402
    Family, estimate_liabilities, estimate_liability, estimate_liability_from_kinship,
    families_from_columns, fit_heritability, kinship_from_pedigree,
    observed_to_liability_h2, prevalence_thresholds, simulate_under_LTM_single,
    tetrachoric,
)

H2 = 0.5
K = 0.05
N_FAM = 800
SEED = 1
N_REP = 10


def _corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def _drop_role(families, role):
    return [Family(fam.fam_id, [m for m in fam.members if m.role != role])
            for fam in families]


def _keep_role(families, role):
    return [Family(fam.fam_id, [m for m in fam.members if m.role == role])
            for fam in families]


def _unbind_role(families, role):
    """Use-I encoding: keep the row, drop the observation.

    Scores identically to removing the row, but ``pids`` still comes from the
    role-``o`` record instead of falling back to ``fam_id``.
    """
    out = []
    for fam in families:
        members = []
        for m in fam.members:
            if m.role == role:
                m = replace(m, lower=-np.inf, upper=np.inf)
            members.append(m)
        out.append(Family(fam.fam_id, members))
    return out


def _incident_risk(mean, variance, *, cip_at_index, cip_at_horizon, h2):
    """Toy non-inbred, no-shared-environment Gaussian-moment calculation.

    The family-only posterior does not include the proband's survival to the
    landmark. Condition on that survival here, rather than treating cumulative
    risk by the horizon as incident risk among unaffected people. This simple
    threshold-crossing model has no competing-event or alive/resident risk-set
    conditioning; a disease CIP alone does not supply those processes.
    """
    sd = np.sqrt(np.asarray(variance) + 1.0 - h2)
    f_index = norm.sf((norm.isf(cip_at_index) - np.asarray(mean)) / sd)
    f_horizon = norm.sf((norm.isf(cip_at_horizon) - np.asarray(mean)) / sd)
    return (f_horizon - f_index) / (1.0 - f_index)


def register_example():
    """Six-person API example; the supplied CIP is illustrative, not empirical."""
    ids = ["p", "m", "f", "s", "gm", "gf"]
    father = ["f", "gf", None, "f", None, None]
    mother = ["m", "gm", None, "m", None, None]
    birth = np.array([1980., 1955., 1950., 1985., 1930., 1925.])
    status = np.array([1, 1, 0, 0, 1, 0])
    age = np.array([48., 72., 80., 40., 65., 90.])
    # Replace this toy grid with externally estimated stratum-specific curves.
    cip_ages = np.array([0., 20., 40., 60., 80., 100.])
    cip_values = np.array([0., 0.005, 0.02, 0.05, 0.08, 0.10])
    common = dict(ids=ids, father=father, mother=mother, probands=["p"],
                  cip_ages=cip_ages, cip_values=cip_values, k_pop=0.10,
                  h2=H2, max_degree=1)
    gwas = estimate_liabilities(status=status, age=age, use="gwas", **common)
    prediction = estimate_liabilities(
        status=status, age=age, use="prediction", birth_time=birth,
        index_time=[2020.], **common)

    # p and m differ only after the landmark; gm and gf are closure-only.
    # None of these changed diagnoses belongs in the prediction observation set.
    changed = estimate_liabilities(
        status=np.array([0, 0, 0, 0, 0, 1]),
        age=np.array([80., 85., 80., 40., 95., 70.]),
        use="prediction", birth_time=birth, index_time=[2020.], **common)
    np.testing.assert_allclose(prediction.est, changed.est, rtol=0, atol=0)
    np.testing.assert_allclose(prediction.var, changed.var, rtol=0, atol=0)

    # Input diagnostics: every non-null parent resolves here, and p was
    # disease-free and followed at the 2020 landmark.
    assert gwas.frac_records_with_unresolved_parents == 0.0
    np.testing.assert_array_equal(prediction.proband_state,
                                  ["disease_free_and_followed"])

    # p is unaffected at age 40 in 2020. The caller defines that at-risk cohort;
    # use="prediction" leaves own status uninformative rather than selecting it.
    cip_index, cip_horizon = np.interp([40., 60.], cip_ages, cip_values)
    risk = _incident_risk(
        prediction.est, prediction.var, cip_at_index=cip_index,
        cip_at_horizon=cip_horizon, h2=H2)
    prior_risk = _incident_risk(
        np.array([0.]), np.array([H2]), cip_at_index=cip_index,
        cip_at_horizon=cip_horizon, h2=H2)
    np.testing.assert_allclose(
        prior_risk, [(cip_horizon - cip_index) / (1.0 - cip_index)],
        rtol=1e-14, atol=0)
    return gwas, prediction, changed, risk, prior_risk


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
    print("the SIB tetrachoric is not interchangeable: 2*rho_sib estimates")
    print("h2 + 2*c2, because full sibs also share the sibship kernel")
    obs = float(np.asarray(observed_to_liability_h2(0.20, pop_prev=K)))
    print(f"Lee: observed-scale 0.20 at K={K} -> liability-scale {obs:.3f}")
    print("pass the study case fraction as prop_cases when the sample is ascertained")
    fit = fit_heritability(
        sim.families, n_iter=250, burn_in=80, seed=SEED, sampling="population",
    )
    print(f"fitted h2 {fit.h2:.3f}  (truth {H2}; within-dataset MC se {fit.h2_se:.4f})")
    print("the MC se is not a sampling interval — a few hundred nuclear families")
    print("at K=0.05 leave a much larger across-cohort SD. Use an external h2")
    print("unless the sampling contract in docs/inference.md holds. Note also")
    print("that fit_heritability REJECTS the personalised bounds of steps 2-3:")
    print("it needs one common case/control threshold per trait.")

    print("\n== 1. Pedigree ==")
    ids = ["o", "m", "f", "s1"]
    _, A = kinship_from_pedigree(
        ids, father=["f", None, None, "f"], mother=["m", None, None, "m"],
    )
    print(f"role grammar: o, m, f, s1  |  kinship A shape {A.shape}")
    print(f"A[o,m]={A[0, 1]:.2f}  A[o,s1]={A[0, 3]:.2f}  A[m,f]={A[1, 2]:.2f}")
    print("cousins / inbreeding / messy half-sibs: build_parent_graph then")
    print("extract_pedigree (choose max_degree for the design), then kinship_from_pedigree")

    print("\n== 2. CIP / lifetime prevalence ==")
    fam_id, role, st, lower_l, upper_l = [], [], [], [], []
    for i, fam in enumerate(sim.families):
        for m in fam.members:
            fam_id.append(f"fam{i}")
            role.append(m.role)
            # status comes from the simulator, not from which bound is finite
            is_case = bool(sim.status[m.role][i])
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
    print("use II (GWAS): include role o; use I (prediction): keep o but")
    print("unbind it — dropping the row makes pids fall back to fam_id")
    print("use III (aetiology): this step is optional")

    print("\n== 4. Estimate mu ==")
    pa = estimate_liability(sim.families, h2=H2)
    score = pa.genetic
    var = pa.var["genetic"]
    print(f"score mean {score.mean():.3f}  sd {score.std():.3f}")
    print(f"PA se is identically 0 (deterministic, NOT approximation-free): "
          f"{pa.se['genetic'][0]:.1f}")
    i_case = int(np.flatnonzero(status)[0])
    i_ctrl = int(np.flatnonzero(1.0 - status)[0])
    print(f"one case    score {score[i_case]:+.3f}  posterior var {var[i_case]:.3f}")
    print(f"one control score {score[i_ctrl]:+.3f}  posterior var {var[i_ctrl]:.3f}")
    r_status = _corr(status, true_g)
    r_pa = _corr(score, true_g)
    print(f"corr(case/control, true g) {r_status:.3f}")
    print(f"corr(LT-FH PA,     true g) {r_pa:.3f}")
    print(f"squared-corr eff-N proxy   {(r_pa / r_status) ** 2:.2f}x")
    rel = estimate_liability(_drop_role(sim.families, "o"), h2=H2)
    print(f"corr(relatives-only, true g) {_corr(rel.genetic, true_g):.3f}")
    adult = estimate_liability(_keep_role(sim.families, "o"), h2=H2)
    print(f"corr(ADuLT, true g) {_corr(adult.genetic, true_g):.3f}")
    print("(with no relatives and one lifetime T, ADuLT is a monotone")
    print(" relabelling of the 0/1 status, so this ties by construction)")

    print("\n-- did it work? --")
    print(f"Var(mu) {score.var():.3f} + mean posterior var {var.mean():.3f} "
          f"= {score.var() + var.mean():.3f}  (law of total variance; h2 = {H2})")
    print(f"max posterior var {var.max():.3f} <= h2 {H2}: "
          f"{bool(var.max() <= H2 + 1e-8)}")
    print(f"mu mean {score.mean():+.3f} (should sit near 0)")
    print(f"cases mean {score[status == 1].mean():+.2f}  "
          f"controls mean {score[status == 0].mean():+.2f}")
    assert len(pa.pids) == N_FAM
    np.testing.assert_allclose(score.var() + var.mean(), H2, atol=0.02)
    assert var.max() <= H2 + 1e-8
    assert abs(score.mean()) < 0.05
    assert score[status == 1].mean() > score[status == 0].mean()
    print("all checks pass")

    print("\n-- engine and API identity checks --")
    n_check = 80
    gibbs = estimate_liability(
        sim.families[:n_check], h2=H2, method="gibbs",
        tol=0.03, n_sim=8_000, burn_in=400, seed=SEED,
    )
    print(f"corr(PA, Gibbs) on {n_check} families "
          f"{_corr(score[:n_check], gibbs.genetic):.4f}")
    print(f"Gibbs MC se (median) {np.median(gibbs.se['genetic']):.4f}")
    rebuilt_score = estimate_liability(rebuilt, h2=H2).genetic
    print(f"max |rebuilt - original| {np.max(np.abs(rebuilt_score - score)):.2e}"
          "  (exact)")
    lo = np.empty((N_FAM, 4))
    hi = np.empty((N_FAM, 4))
    for i, fam in enumerate(sim.families):
        by = {m.role: m for m in fam.members}
        for j, r in enumerate(ids):
            lo[i, j] = by[r].lower
            hi[i, j] = by[r].upper
    kin, kin_se, _kin_var = estimate_liability_from_kinship(A, lo, hi, h2=H2)
    print(f"max |kinship PA - role PA| {np.max(np.abs(kin - score)):.2e}  "
          "(PA fold order differs between row layouts: approximation error,")
    print("                                          not round-off, not Monte Carlo)")
    print(f"kinship PA se is 0: {kin_se[0]:.1f}")

    print("\n== 5. What you do with mu ==")
    print(f"{len(pa.pids)} pids")
    print("I  prediction: own status out; optional PGS is downstream")
    print("II GWAS: own status in; join pids, residualize, scan elsewhere")
    print("III aetiology: h2, r_g, CIP can stand alone (steps 0 and/or 2)")

    print("\n-- use I only: mu on the liability scale -> an implied risk --")
    T = float(norm.isf(K))
    cells = []
    for seed in range(1, N_REP + 1):
        sim_p = simulate_under_LTM_single(
            fam_vec=["m", "f", "s1"], h2=H2, pop_prev=K, n_sim=4_000,
            use_age=False, seed=seed,
        )
        # use-I encoding: keep role o, give it no observation
        fh = estimate_liability(_unbind_role(sim_p.families, "o"), h2=H2)
        risk = norm.sf((T - fh.genetic) / np.sqrt(fh.var["genetic"] + 1 - H2))
        obs_status = sim_p.status["o"].astype(float)
        order = np.argsort(risk)
        top, bot = order[-400:], order[:400]
        cells.append((risk.mean(), obs_status.mean(),
                      risk[top].mean(), obs_status[top].mean(),
                      risk[bot].mean(), obs_status[bot].mean()))
    c = np.asarray(cells)

    def _ms(col):
        return c[:, col].mean(), c[:, col].std(ddof=1) / np.sqrt(len(c))

    print(f"T = {T:.4f};  {N_REP} replicates of 4,000 relatives-only families")
    for label, pc, oc in (("overall", 0, 1), ("top decile", 2, 3),
                          ("bottom decile", 4, 5)):
        pm, pse = _ms(pc)
        om, ose = _ms(oc)
        d = c[:, oc] - c[:, pc]
        dse = d.std(ddof=1) / np.sqrt(len(d))
        print(f"{label:14} predicted {pm:.4f} +/- {pse:.4f}   "
              f"observed {om:.4f} +/- {ose:.4f}   "
              f"gap {d.mean():+.4f} +/- {dse:.4f} ({abs(d.mean()) / dse:.1f} SE)")
    print("compatible with calibration overall and in both tails in this simulation;")
    print("a single seed can look off by")
    print("2-3 SE, so do not read one replicate as a bias")
    print("not valid for use II: there the proband's own status is already in D_F")

    print("\n== On real data, step 2 is a CIP, not one K ==")
    sim_age = simulate_under_LTM_single(
        fam_vec=["m", "f", "s1"], h2=H2, pop_prev=K, n_sim=N_FAM,
        use_age=True, seed=SEED,
    )
    pa_age = estimate_liability(sim_age.families, h2=H2)
    r_age = _corr(pa_age.genetic, sim_age.genetic)
    r_age_cc = _corr(sim_age.status["o"].astype(float), sim_age.genetic)

    # Control: the SAME censored cohort scored with classic one-K bounds, so the
    # age term is the only thing that differs from the run above. Without this
    # arm you cannot attribute the gain over the 0/1 label to age-awareness --
    # nearly all of it is family history.
    fid, rol, sta = [], [], []
    for i, fam in enumerate(sim_age.families):
        for m in fam.members:
            fid.append(f"fam{i}")
            rol.append(m.role)
            sta.append(int(sim_age.status[m.role][i]))
    lo_c, hi_c = prevalence_thresholds(np.array(sta), pop_prev=K)
    classic = estimate_liability(families_from_columns(fid, rol, lo_c, hi_c), h2=H2)
    r_classic = _corr(classic.genetic, sim_age.genetic)

    print(f"observed proband case rate {sim_age.status['o'].astype(float).mean():.4f}"
          f"  (against {status.mean():.4f} without ages)")
    print(f"corr(proband 0/1 label,       true g) {r_age_cc:.3f}")
    print(f"corr(classic one-K LT-FH,     true g) {r_classic:.3f}   "
          f"-> {(r_classic / r_age_cc) ** 2:.2f}x eff-N over the label")
    print(f"corr(age-aware LT-FH++,       true g) {r_age:.3f}   "
          f"-> {(r_age / r_classic) ** 2:.2f}x eff-N over classic")
    print("NOT comparable with the 0.419 above — different cohort, 15x fewer")
    print("observed cases. And note the split: family history does nearly all")
    print("the work here; the age term adds the ~1.0-1.1x that RESULTS section")
    print("10 also reports. Do not credit family history's gain to age.")
    print("real LT-FH++ needs thresholds_from_cip with stratified population CIPs")

    print("\n== Public register driver: a six-person calendar example ==")
    gwas, prediction, changed, risk, prior_risk = register_example()
    print("toy CIP supplied explicitly; not empirical data or a benchmark")
    print("max_degree=1 deliberately includes parents and sibling; two maternal")
    print("grandparents remain in exact kinship but not in the observation set")
    print(f"GWAS: proband {gwas.probands[0]}, score {gwas.est[0]:+.6f}, "
          f"posterior var {gwas.var[0]:.6f}, conditioned {gwas.n_conditioned[0]}")
    print(f"Prediction at 2020: score {prediction.est[0]:+.6f}, "
          f"posterior var {prediction.var[0]:.6f}")
    print(f"relatives {prediction.n_relatives[0]}, "
          f"closure-only {prediction.n_closure_only[0]}, "
          f"conditioned {prediction.n_conditioned[0]}")
    print("2020 attained ages: mother 65, father 70, sibling 35; own status out")
    print(f"post-index / closure diagnosis changes: max score difference "
          f"{np.max(np.abs(prediction.est - changed.est)):.1f}")
    print(f"incident risk age 40 to 60, given unaffected at 40: {risk[0]:.6f}")
    print(f"no-history prior check: (0.05 - 0.02) / (1 - 0.02) = {prior_risk[0]:.6f}")
    print("risk uses a non-inbred, no-C/M, Gaussian posterior approximation")
    print("without competing-event or alive/resident risk-set conditioning;")
    print("it is not a clinical calibration result")
    print("done.")


if __name__ == "__main__":
    main()
