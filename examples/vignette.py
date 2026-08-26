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

Every number printed here is quoted in docs/vignette.md, and
``tests/test_vignette_numbers.py`` fails if the two drift apart. ``main``
returns them as a dict for that test; if you change the script, re-run it
and update the page.

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
    Family, estimate_liability, estimate_liability_from_kinship,
    families_from_columns, fit_heritability, kinship_from_pedigree,
    observed_to_liability_h2, prevalence_thresholds, simulate_under_LTM_single,
)
# `from ltpred import tetrachoric` is the form the vignette shows and it is
# correct in a script like this one -- but the package exports a *function*
# under the same name as its module, so in a process that has already imported
# `ltpred.tetrachoric` (a test suite, say) the package attribute is the module
# and the call raises `TypeError: 'module' object is not callable`. Import from
# the owning module, which api.md documents and which is never ambiguous.
from ltpred.tetrachoric import tetrachoric  # noqa: E402

H2 = 0.5
K = 0.05
N_FAM = 800
SEED = 1
N_REP = 10
# A parent-offspring tetrachoric at K=0.05 needs concordant case PAIRS, which
# N_FAM families do not supply: see the comment in step 0.
N_TET = 25_000
# Under age censoring only a few per 1,000 probands are observed cases, so the
# own-status baseline needs a much larger cohort than N_FAM to mean anything.
N_AGE = 20_000


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


def _status_columns(families):
    """Long-format (fam_id, role, status) columns, as a register would hold them."""
    fam_id, role, status = [], [], []
    for i, fam in enumerate(families):
        for m in fam.members:
            fam_id.append(f"fam{i}")
            role.append(m.role)
            status.append(int(np.isfinite(m.lower)))
    return fam_id, role, status


def main():
    f = {}
    print("== Stand-in cohort (simulated; skip on real data) ==")
    sim = simulate_under_LTM_single(
        fam_vec=["m", "f", "s1"], h2=H2, pop_prev=K, n_sim=N_FAM,
        use_age=False, seed=SEED,
    )
    status = sim.status["o"].astype(float)
    true_g = sim.genetic
    f["n_fam"], f["h2"], f["K"] = N_FAM, H2, K
    f["case_rate"] = float(status.mean())
    print(f"families {N_FAM}; liability-scale h2 {H2}; prevalence {K}")
    print(f"proband case rate {f['case_rate']:.3f}")

    print("\n== 0. Heritability and covariance ==")
    o_case = sim.status["o"].astype(int)
    m_case = sim.status["m"].astype(int)
    po = tetrachoric(o_case, m_case)
    f["tet_rho_small"], f["tet_se_small"] = float(po.rho), float(po.se)
    f["tet_pairs_small"] = int(np.sum(o_case & m_case))
    print(f"parent-offspring tetrachoric on these {N_FAM} families: "
          f"{po.rho:+.3f} +/- {po.se:.3f},")
    print(f"  so Falconer 2*rho = {2 * po.rho:+.3f} +/- {2 * po.se:.3f} against a truth "
          f"of {H2}.")
    print(f"  The whole estimate rests on the {f['tet_pairs_small']} case-case pair(s) "
          f"in {N_FAM} rows,")
    print("  so it carries no usable information here. Do NOT read a close hit")
    print("  on a cohort this size as validation.")
    big = simulate_under_LTM_single(
        fam_vec=["m", "f", "s1"], h2=H2, pop_prev=K, n_sim=N_TET,
        use_age=False, seed=SEED,
    )
    po_big = tetrachoric(big.status["o"].astype(int), big.status["m"].astype(int))
    sib_big = tetrachoric(big.status["o"].astype(int), big.status["s1"].astype(int))
    f["tet_rho_big"], f["tet_se_big"] = float(po_big.rho), float(po_big.se)
    f["tet_2rho_big"] = float(2 * po_big.rho)
    f["tet_sib_rho_big"], f["tet_sib_se_big"] = float(sib_big.rho), float(sib_big.se)
    print(f"the same estimator on {N_TET:,} families: {po_big.rho:.3f} +/- {po_big.se:.3f},")
    print(f"  so 2*rho = {2 * po_big.rho:.3f} +/- {2 * po_big.se:.3f} against a truth of {H2}")
    print(f"the SIB tetrachoric there is {sib_big.rho:.3f} +/- {sib_big.se:.3f}. It agrees")
    print("only because this simulator has no sibship kernel: 2*rho_sib estimates")
    print("h2 + 2*c2, so with c2 > 0 the two routes separate")
    f["lee"] = float(np.asarray(observed_to_liability_h2(0.20, pop_prev=K)))
    print(f"Lee: observed-scale 0.20 at K={K} -> liability-scale {f['lee']:.3f}")
    print("pass the study case fraction as prop_cases when the sample is ascertained")
    fit = fit_heritability(
        sim.families, n_iter=250, burn_in=80, seed=SEED, sampling="population",
    )
    f["fit_h2"], f["fit_h2_se"] = float(fit.h2), float(fit.h2_se)
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
    print("extract_pedigree (start at max_degree=2), then kinship_from_pedigree")

    print("\n== 2. CIP / lifetime prevalence ==")
    fam_id, role, st = _status_columns(sim.families)
    lower, upper = prevalence_thresholds(np.array(st), pop_prev=K)
    f["n_rows"] = len(st)
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
    f["mu_mean"], f["mu_sd"] = float(score.mean()), float(score.std())
    print(f"score mean {score.mean():.3f}  sd {score.std():.3f}")
    print(f"PA se is identically 0 (deterministic, NOT approximation-free): "
          f"{pa.se['genetic'][0]:.1f}")
    i_case = int(np.flatnonzero(status)[0])
    i_ctrl = int(np.flatnonzero(1.0 - status)[0])
    f["case_mu"], f["case_var"] = float(score[i_case]), float(var[i_case])
    f["ctrl_mu"], f["ctrl_var"] = float(score[i_ctrl]), float(var[i_ctrl])
    print(f"one case    score {score[i_case]:+.3f}  posterior var {var[i_case]:.3f}")
    print(f"one control score {score[i_ctrl]:+.3f}  posterior var {var[i_ctrl]:.3f}")
    r_status = _corr(status, true_g)
    r_pa = _corr(score, true_g)
    f["corr_status"], f["corr_pa"] = r_status, r_pa
    f["eff_n"] = (r_pa / r_status) ** 2
    print(f"corr(case/control, true g) {r_status:.3f}")
    print(f"corr(LT-FH PA,     true g) {r_pa:.3f}")
    print(f"squared-corr eff-N proxy   {f['eff_n']:.2f}x")
    rel = estimate_liability(_drop_role(sim.families, "o"), h2=H2)
    f["corr_rel"] = _corr(rel.genetic, true_g)
    print(f"corr(relatives-only, true g) {f['corr_rel']:.3f}")
    adult = estimate_liability(_keep_role(sim.families, "o"), h2=H2)
    f["corr_adult"] = _corr(adult.genetic, true_g)
    print(f"corr(ADuLT, true g) {f['corr_adult']:.3f}")
    print("(with no relatives and one lifetime T, ADuLT is a monotone")
    print(" relabelling of the 0/1 status, so this ties by construction)")

    print("\n-- did it work? --")
    f["var_mu"], f["mean_post_var"] = float(score.var()), float(var.mean())
    f["lotv_sum"] = f["var_mu"] + f["mean_post_var"]
    f["max_post_var"] = float(var.max())
    f["cases_mean"] = float(score[status == 1].mean())
    f["ctrls_mean"] = float(score[status == 0].mean())
    print(f"Var(mu) {score.var():.3f} + mean posterior var {var.mean():.3f} "
          f"= {f['lotv_sum']:.3f}  (law of total variance; h2 = {H2})")
    print(f"max posterior var {var.max():.3f} <= h2 {H2}: "
          f"{bool(var.max() <= H2 + 1e-8)}")
    print(f"mu mean {score.mean():+.3f} (should sit near 0)")
    print(f"cases mean {f['cases_mean']:+.2f}  controls mean {f['ctrls_mean']:+.2f}")
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
    f["corr_pa_gibbs"] = _corr(score[:n_check], gibbs.genetic)
    f["gibbs_se_median"] = float(np.median(gibbs.se["genetic"]))
    print(f"corr(PA, Gibbs) on {n_check} families {f['corr_pa_gibbs']:.4f}")
    print(f"Gibbs MC se (median) {f['gibbs_se_median']:.4f}")
    rebuilt_score = estimate_liability(rebuilt, h2=H2).genetic
    f["rebuilt_max_diff"] = float(np.max(np.abs(rebuilt_score - score)))
    print(f"max |rebuilt - original| {f['rebuilt_max_diff']:.2e}  (exact)")
    lo = np.empty((N_FAM, 4))
    hi = np.empty((N_FAM, 4))
    for i, fam in enumerate(sim.families):
        by = {m.role: m for m in fam.members}
        for j, r in enumerate(ids):
            lo[i, j] = by[r].lower
            hi[i, j] = by[r].upper
    kin, kin_se, _kin_var = estimate_liability_from_kinship(A, lo, hi, h2=H2)
    f["kinship_max_diff"] = float(np.max(np.abs(kin - score)))
    print(f"max |kinship PA - role PA| {f['kinship_max_diff']:.1e}  "
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
    f["T"] = T
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
    for label, key, pc, oc in (("overall", "all", 0, 1),
                               ("top decile", "top", 2, 3),
                               ("bottom decile", "bot", 4, 5)):
        pm, pse = _ms(pc)
        om, ose = _ms(oc)
        d = c[:, oc] - c[:, pc]
        dse = d.std(ddof=1) / np.sqrt(len(d))
        f[f"risk_pred_{key}"], f[f"risk_pred_{key}_se"] = pm, pse
        f[f"risk_obs_{key}"], f[f"risk_obs_{key}_se"] = om, ose
        f[f"risk_gap_{key}_se_units"] = abs(d.mean()) / dse
        print(f"{label:14} predicted {pm:.4f} +/- {pse:.4f}   "
              f"observed {om:.4f} +/- {ose:.4f}   "
              f"gap {d.mean():+.4f} +/- {dse:.4f} ({abs(d.mean()) / dse:.1f} SE)")
    print("calibrated overall and in both tails; a single seed can look off by")
    print("2-3 SE, so do not read one replicate as a bias")
    print("not valid for use II: there the proband's own status is already in D_F")

    print("\n== On real data, step 2 is a CIP, not one K ==")
    sim_age = simulate_under_LTM_single(
        fam_vec=["m", "f", "s1"], h2=H2, pop_prev=K, n_sim=N_AGE,
        use_age=True, seed=SEED,
    )
    age_status = sim_age.status["o"].astype(float)
    pa_age = estimate_liability(sim_age.families, h2=H2)
    r_age = _corr(pa_age.genetic, sim_age.genetic)
    r_age_cc = _corr(age_status, sim_age.genetic)

    # Control: the SAME censored cohort scored with classic one-K bounds, so the
    # age term is the only thing that differs from the run above. Without this
    # arm you cannot attribute the gain over the 0/1 label to age-awareness --
    # nearly all of it is family history.
    fid, rol, sta = _status_columns(sim_age.families)
    lo_c, hi_c = prevalence_thresholds(np.array(sta), pop_prev=K)
    classic = estimate_liability(families_from_columns(fid, rol, lo_c, hi_c), h2=H2)
    r_classic = _corr(classic.genetic, sim_age.genetic)

    f["age_n_fam"] = N_AGE
    f["age_cases"] = int(age_status.sum())
    f["age_case_rate"] = float(age_status.mean())
    f["age_corr_label"] = r_age_cc
    f["age_corr_classic"] = r_classic
    f["age_corr_aware"] = r_age
    f["age_fh_gain"] = (r_classic / r_age_cc) ** 2
    f["age_gain"] = (r_age / r_classic) ** 2
    print(f"{N_AGE:,} families; observed proband case rate {f['age_case_rate']:.4f} "
          f"({f['age_cases']} cases), against {f['case_rate']:.4f} without ages")
    print(f"corr(proband 0/1 label,       true g) {r_age_cc:.3f}")
    print(f"corr(classic one-K LT-FH,     true g) {r_classic:.3f}   "
          f"-> {f['age_fh_gain']:.2f}x eff-N over the label")
    print(f"corr(age-aware LT-FH++,       true g) {r_age:.3f}   "
          f"-> {f['age_gain']:.2f}x eff-N over classic")
    print(f"NOT comparable with the {r_pa:.3f} above — different cohort, "
          f"{f['case_rate'] / f['age_case_rate']:.0f}x fewer observed cases.")
    print("And note the split: family history does nearly all the work here; the")
    print(f"age term adds {f['age_gain']:.2f}x on top, inside the 1.02-1.05x that")
    print("RESULTS section 10 reports. Do not credit family history's gain to age.")
    print("real LT-FH++ needs thresholds_from_cip with stratified population CIPs")
    print("done.")
    return f


if __name__ == "__main__":
    main()
