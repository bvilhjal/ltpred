"""LT-FH++ thresholds in an independent-SNP marginal-association simulation.

The main panel combines age-, sex-, and birth-cohort-dependent cumulative
incidence (CIP), coherent family follow-up/onset, competing mortality,
ascertainment, and demographically stratified null SNPs.  It compares

  case/control -> ADuLT (full personalised bounds, no family history) ->
  classic LT-FH -> LT-FH++ component ablations -> full LT-FH++.

The matched ``ADuLT -> LT-FH++`` contrast holds every proband's personalised
age/sex/cohort bounds fixed and adds only their relatives.  It therefore isolates
the family-history contribution that distinguishes LT-FH++ from ADuLT.

A separate, prespecified sex-isolation panel asks the narrower question that the
integrated panel cannot answer cleanly: does adding the correct sex-specific
lifetime prevalence help beyond age alone?  It uses five paired replicates by
default (N=3,000, M=600, 30 causal SNPs), sex RR=2, equal onset midpoints, an
age-dependent CIP, and no cohort or sex-dependent-mortality effect.

Pearson-Aitken (PA) is the primary single-trait engine.  Gibbs is run only for the
first ``--gibbs-reps`` main-panel replicates and is used as an agreement diagnostic,
not as a second headline analysis. Raw and sex/birth-year-adjusted score/truth and
marginal-association metrics are both retained. Paired contrasts use t-based 95%
confidence intervals across independent simulated cohorts. This is not a
real-LD, related-sample mixed-model GWAS.

Run:
    python benchmarks/bench_ltfhpp_personalization.py
    python benchmarks/bench_ltfhpp_personalization.py --n-fam 8000 --reps 10
    python benchmarks/bench_ltfhpp_personalization.py --output-prefix /tmp/ltfhpp

The default output prefix is ``benchmarks/bench_ltfhpp_personalization``.
"""

from __future__ import annotations

import argparse
import csv
import os
import warnings

import numpy as np
from scipy import stats
from scipy.special import ndtr

from _common import get_plt, gwas_chisq, lambda_gc
from ltpred.covariance import construct_covmat_single, correct_positive_definite
from ltpred.estimate import (estimate_liability_gibbs_arrays,
                             estimate_liability_pa_arrays)


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_PREFIX = os.path.join(HERE, "bench_ltfhpp_personalization")
CAL_NOW = 2020.0
BY_REF = 1965.0
STRUCT = ["m", "f", "s1", "s2"]
GW = float(stats.chi2.isf(5e-8, 1))
DEATH_MEAN_MALE = 80.0
DEATH_SD = 10.0

MAIN_PANEL = "main_ablation"
MAIN_SCENARIO = "integrated_age_sex_cohort"
SEX_PANEL = "sex_isolation"
SEX_SCENARIO = "prespecified_sex_rr_only"

CASE_CONTROL = "case/control"
ADULT = "ADuLT (no family history)"
LTFH = "LT-FH (single K)"
AGE = "FH + age CIP"
AGE_SEX = "FH + age + sex CIP"
AGE_COHORT = "FH + age + cohort CIP"
FULL = "LT-FH++ (full CIP)"
ORACLE = "oracle (true g)"

POLICY_LABELS = {
    "adult": ADULT,
    "ltfh": LTFH,
    "age": AGE,
    "age_sex": AGE_SEX,
    "age_cohort": AGE_COHORT,
    "full": FULL,
}
MAIN_POLICIES = tuple(POLICY_LABELS)
SEX_POLICIES = ("age", "age_sex")

MAIN_CONTRASTS = (
    (CASE_CONTROL, ADULT),
    (ADULT, FULL),
    (CASE_CONTROL, LTFH),
    (LTFH, FULL),
    (LTFH, AGE),
    (AGE, AGE_SEX),
    (AGE, AGE_COHORT),
    (AGE, FULL),
    (AGE_SEX, FULL),
    (AGE_COHORT, FULL),
)
SEX_CONTRASTS = ((AGE, AGE_SEX),)

CONTRAST_METRICS = (
    "corr_g_raw", "corr_g_adjusted",
    "calibration_slope_raw", "calibration_slope_adjusted",
    "mean_error_raw", "mean_error_female_raw", "mean_error_male_raw",
    "mean_error_sex_gap_raw",
    "mean_chi2_causal_raw", "mean_chi2_causal_adjusted",
    "effN_vs_cc_raw", "effN_vs_cc_adjusted",
)

_AGE_GAP = {"o": 0.0, "s": 0.0, "m": 29.0, "f": 31.0,
            "mgm": 56.0, "mgf": 58.0, "pgm": 56.0, "pgf": 58.0}


def _stem(role):
    return role.rstrip("0123456789")


def _demography(non_g, n, rng, args):
    """Return sex (female=1), birth year, and observed age for every member."""
    by_o = rng.uniform(1945.0, 1990.0, n)
    sex = np.empty((n, len(non_g)))
    birth = np.empty_like(sex)
    observed_age = np.empty_like(sex)
    for j, role in enumerate(non_g):
        stem = _stem(role)
        if stem in {"m", "mgm", "pgm"}:
            sx = np.ones(n)
        elif stem in {"f", "mgf", "pgf"}:
            sx = np.zeros(n)
        else:
            sx = rng.integers(0, 2, n).astype(float)
        gap = _AGE_GAP.get(stem, 0.0)
        sd = 0.0 if stem == "o" else (4.0 if gap else 3.0)
        by = by_o - gap + rng.normal(0.0, sd, n)
        age_now = CAL_NOW - by
        if stem == "o":
            obs = age_now                       # living genotyped proband
        else:
            death_mean = DEATH_MEAN_MALE + args.death_sex_diff * sx
            death = np.clip(rng.normal(death_mean, DEATH_SD), 1.0, 110.0)
            obs = np.minimum(age_now, death)
        sex[:, j] = sx
        birth[:, j] = by
        observed_age[:, j] = np.clip(obs, 1.0, 110.0)
    return sex, birth, observed_age


def _cip_parameters(sex, birth, args, *, use_sex, use_cohort):
    """Lifetime prevalence and onset midpoint under one ablation policy."""
    sex_center = sex - 0.5 if use_sex else 0.0
    cohort = (birth - BY_REF) / 30.0 if use_cohort else 0.0
    K = args.K * np.power(args.sex_rr, sex_center) * np.power(args.trend_R, cohort)
    K = np.clip(K, 1e-4, 0.8)
    mid = args.mid + args.sex_mid_diff * sex_center + args.cohort_mid_shift * cohort
    return K, mid


def _cip(age, K, mid, slope):
    z = np.clip((mid - age) * slope, -700.0, 700.0)
    return K / (1.0 + np.exp(z))


def _onset_from_liability(liability, K, mid, slope):
    """Invert the personalized CIP threshold; infinity means never affected."""
    K = np.broadcast_to(K, liability.shape)
    mid = np.broadcast_to(mid, liability.shape)
    tail = np.clip(1.0 - ndtr(liability), 1e-15, 1.0)
    onset = np.full(liability.shape, np.inf)
    affected = tail < K
    onset[affected] = (mid[affected]
                       - np.log(K[affected] / tail[affected] - 1.0) / slope)
    return np.maximum(onset, 0.0)


def _genetic_model(args, rng):
    maf = rng.uniform(0.08, 0.48, args.m_snps)
    causal = rng.choice(args.m_snps, args.n_causal, replace=False)
    causal_mask = np.zeros(args.m_snps, dtype=bool)
    causal_mask[causal] = True
    null = np.flatnonzero(~causal_mask)
    strat = rng.choice(null, args.n_strat, replace=False)
    strat_mask = np.zeros(args.m_snps, dtype=bool)
    strat_mask[strat] = True

    beta = rng.normal(size=args.n_causal)
    beta *= np.sqrt(args.h2) / np.linalg.norm(beta)
    cohort_coef = np.zeros(args.m_snps)
    sex_coef = np.zeros(args.m_snps)
    if args.cohort_genotype_stratification:
        cohort_coef[strat] = (rng.uniform(0.008, 0.022, args.n_strat)
                              * rng.choice([-1, 1], args.n_strat))
    sex_coef[strat] = (rng.uniform(0.008, 0.025, args.n_strat)
                       * rng.choice([-1, 1], args.n_strat))
    return dict(maf=maf, causal=causal, causal_mask=causal_mask,
                strat_mask=strat_mask, beta=beta,
                cohort_coef=cohort_coef, sex_coef=sex_coef)


def _draw_batch(n, non_g, cond_cov, cross, args, model, rng):
    sex, birth, obs_age = _demography(non_g, n, rng, args)
    o = non_g.index("o")
    cohort_z = (birth[:, o] - BY_REF) / 30.0
    sex_z = sex[:, o] - 0.5
    P = (model["maf"][None, :]
         + cohort_z[:, None] * model["cohort_coef"][None, :]
         + sex_z[:, None] * model["sex_coef"][None, :])
    P = np.clip(P, 0.02, 0.98)
    X = rng.binomial(2, P).astype(np.float64)

    maf = model["maf"]
    scale = np.sqrt(2.0 * maf * (1.0 - maf))
    X_pop = (X - 2.0 * maf) / scale
    g = X_pop[:, model["causal"]] @ model["beta"]
    mean = np.outer(g, cross / args.h2)
    liability = mean + rng.multivariate_normal(
        np.zeros(len(non_g)), cond_cov, size=n
    )

    K, mid = _cip_parameters(
        sex, birth, args,
        use_sex=args.truth_use_sex,
        use_cohort=args.truth_use_cohort,
    )
    onset = _onset_from_liability(liability, K, mid, args.slope)
    status = onset <= obs_age
    onset_observed = np.where(status, np.round(onset), np.inf)
    return dict(X=X, g=g, liability=liability, sex=sex, birth=birth,
                obs_age=obs_age, onset=onset_observed, status=status)


def simulate_ascertained(args, seed):
    """Generate exactly ``n_fam`` probands at the requested observed case fraction."""
    rng = np.random.default_rng(seed)
    cov_obj = construct_covmat_single(args.fam, add_ind=True, h2=args.h2)
    cov, _ = correct_positive_definite(cov_obj.matrix)
    non_g = cov_obj.roles[1:]
    rest = np.arange(1, len(cov_obj.roles))
    cross = cov[rest, 0]
    cond_cov = cov[np.ix_(rest, rest)] - np.outer(cross, cross) / args.h2
    cond_cov = correct_positive_definite(cond_cov)[0]
    model = _genetic_model(args, rng)
    o = non_g.index("o")

    need_case = int(round(args.case_frac * args.n_fam))
    need_ctrl = args.n_fam - need_case
    blocks = []
    got_case = got_ctrl = 0
    attempts = 0
    while got_case < need_case or got_ctrl < need_ctrl:
        attempts += 1
        if attempts > 100:
            raise RuntimeError(
                "could not obtain the requested case fraction; increase K or "
                "reduce --case-frac"
            )
        raw = _draw_batch(max(2000, args.n_fam // 2), non_g, cond_cov,
                          cross, args, model, rng)
        proband_case = raw["status"][:, o]
        ci = np.flatnonzero(proband_case)[:max(0, need_case - got_case)]
        ni = np.flatnonzero(~proband_case)[:max(0, need_ctrl - got_ctrl)]
        take = np.concatenate([ci, ni])
        if take.size:
            blocks.append({k: v[take] for k, v in raw.items()})
        got_case += len(ci)
        got_ctrl += len(ni)

    data = {k: np.concatenate([b[k] for b in blocks]) for k in blocks[0]}
    perm = rng.permutation(args.n_fam)
    data = {k: v[perm] for k, v in data.items()}
    X = data["X"]
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    data["Xs"] = (X - X.mean(axis=0)) / sd
    data.update(model)
    data["roles"] = non_g

    # Frisch-Waugh-Lovell adjustment.  Residualizing only the phenotype gives the
    # right numerator but the wrong SNP variance for sex/cohort-correlated SNPs.
    z = _standardize(data["birth"][:, o])
    C = np.column_stack([np.ones(args.n_fam), data["sex"][:, o], z])
    X_adj = data["Xs"] - C @ np.linalg.lstsq(C, data["Xs"], rcond=None)[0]
    xsd = X_adj.std(axis=0)
    xsd[xsd == 0] = 1.0
    data["Xs_adjusted"] = X_adj / xsd
    return data


def build_bounds(data, args, policy):
    status = data["status"]
    if policy == "ltfh":
        t = float(stats.norm.isf(args.K))
        return np.where(status, t, -np.inf), np.where(status, np.inf, t)

    use_sex = policy in {"age_sex", "full", "adult"}
    use_cohort = policy in {"age_cohort", "full", "adult"}
    K, mid = _cip_parameters(data["sex"], data["birth"], args,
                             use_sex=use_sex, use_cohort=use_cohort)
    age = np.where(status, data["onset"], data["obs_age"])
    cip = np.clip(_cip(age, K, mid, args.slope), 1e-6, 1.0 - 1e-8)
    threshold = stats.norm.isf(cip)
    lower = np.where(status, threshold, -np.inf)
    upper = threshold                  # onset-pinned case; control ends at current-age threshold
    return lower, upper


def _standardize(x):
    x = np.asarray(x, dtype=float)
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else np.zeros_like(x)


def _residualize(y, sex, birth):
    z = _standardize(birth)
    C = np.column_stack([np.ones(len(y)), sex, z])
    return np.asarray(y, float) - C @ np.linalg.lstsq(C, y, rcond=None)[0]


def _calibration_slope(score, truth):
    variance = np.var(score, ddof=1)
    if variance <= 0:
        return np.nan
    return float(np.cov(truth, score, ddof=1)[0, 1] / variance)


def _safe_corr(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _metadata(args, panel, scenario):
    return dict(
        row_type="replicate",
        panel=panel,
        scenario=scenario,
        n_fam=args.n_fam,
        m_snps=args.m_snps,
        n_causal=args.n_causal,
        n_strat=args.n_strat,
        fam="+".join(args.fam),
        h2=args.h2,
        K=args.K,
        case_frac_target=args.case_frac,
        mid=args.mid,
        slope=args.slope,
        sex_rr=args.sex_rr,
        sex_mid_diff=args.sex_mid_diff,
        trend_R=args.trend_R,
        cohort_mid_shift=args.cohort_mid_shift,
        death_mean_male=DEATH_MEAN_MALE,
        death_sex_diff=args.death_sex_diff,
        death_sd=DEATH_SD,
        truth_use_sex=int(args.truth_use_sex),
        truth_use_cohort=int(args.truth_use_cohort),
        truth_age_dependent=1,
        genotype_stratification=("sex+cohort" if args.cohort_genotype_stratification
                                 else "sex_only"),
        panel_reps=args.reps,
        seed=args.seed,
        gibbs_reps=args.gibbs_reps,
        gibbs_n=args.gibbs_n,
        gibbs_n_sim=args.n_sim,
        gibbs_burn_in=args.gibbs_burn_in,
        gibbs_tol=args.gibbs_tol,
        gibbs_max_rounds=args.gibbs_max_rounds,
        calendar_now=CAL_NOW,
        birth_year_reference=BY_REF,
    )


def _score(name, y, data, args, panel, scenario, rep):
    o = data["roles"].index("o")
    score = np.asarray(y, dtype=float)
    truth = np.asarray(data["g"], dtype=float)
    sex = data["sex"][:, o]
    birth = data["birth"][:, o]
    score_adj = _residualize(score, sex, birth)
    truth_adj = _residualize(truth, sex, birth)

    chi2_raw = gwas_chisq(data["Xs"], score)
    chi2_adjusted = gwas_chisq(data["Xs_adjusted"], score_adj)
    causal = data["causal_mask"]
    strat = data["strat_mask"]
    independent = ~(causal | strat)
    female = sex == 1
    male = ~female
    raw_error = score - truth
    female_error = float(raw_error[female].mean())
    male_error = float(raw_error[male].mean())

    row = _metadata(args, panel, scenario)
    row.update(
        rep=rep,
        phenotype=name,
        comparison="",
        corr_g_raw=_safe_corr(score, truth),
        corr_g_adjusted=_safe_corr(score_adj, truth_adj),
        calibration_slope_raw=_calibration_slope(score, truth),
        calibration_slope_adjusted=_calibration_slope(score_adj, truth_adj),
        mean_error_raw=float(raw_error.mean()),
        mean_error_female_raw=female_error,
        mean_error_male_raw=male_error,
        mean_error_sex_gap_raw=female_error - male_error,
        mean_chi2_causal_raw=float(chi2_raw[causal].mean()),
        mean_chi2_causal_adjusted=float(chi2_adjusted[causal].mean()),
        power_gw_raw=float((chi2_raw[causal] > GW).mean()),
        power_gw_adjusted=float((chi2_adjusted[causal] > GW).mean()),
        lambda_strat_raw=lambda_gc(chi2_raw[strat]),
        lambda_strat_adjusted=lambda_gc(chi2_adjusted[strat]),
        lambda_independent_raw=lambda_gc(chi2_raw[independent]),
        lambda_independent_adjusted=lambda_gc(chi2_adjusted[independent]),
        case_fraction_observed=float(data["status"][:, o].mean()),
        effN_vs_cc_raw=np.nan,
        effN_vs_cc_adjusted=np.nan,
        gibbs_n_used=0,
        gibbs_agreement=np.nan,
        gibbs_mcse_mean=np.nan,
        gibbs_mcse_max=np.nan,
        gibbs_nonfinite_n=np.nan,
        gibbs_unconverged_n=np.nan,
        gibbs_minus_pa_mean=np.nan,
        gibbs_rmse_vs_pa=np.nan,
        gibbs_nrmse_score_sd=np.nan,
        gibbs_vs_pa_slope=np.nan,
        gibbs_vs_pa_intercept=np.nan,
    )
    return row


def _add_gibbs_diagnostics(row, pa, gibbs, mcse, tol):
    diff = gibbs - pa
    finite = np.isfinite(mcse)
    nonfinite = int((~finite).sum())
    unconverged = int(np.count_nonzero(mcse[finite] > tol)) + nonfinite
    pa_sd = float(np.std(pa, ddof=1))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    slope = _calibration_slope(pa, gibbs)  # regression Gibbs on PA
    intercept = float(gibbs.mean() - slope * pa.mean()) if np.isfinite(slope) else np.nan
    row.update(
        gibbs_n_used=len(pa),
        gibbs_agreement=_safe_corr(pa, gibbs),
        gibbs_mcse_mean=float(mcse[finite].mean()) if finite.any() else np.nan,
        gibbs_mcse_max=float(mcse[finite].max()) if finite.any() else np.nan,
        gibbs_nonfinite_n=nonfinite,
        gibbs_unconverged_n=unconverged,
        gibbs_minus_pa_mean=float(diff.mean()),
        gibbs_rmse_vs_pa=rmse,
        gibbs_nrmse_score_sd=rmse / pa_sd if pa_sd > 0 else np.nan,
        gibbs_vs_pa_slope=slope,
        gibbs_vs_pa_intercept=intercept,
    )
    if unconverged:
        warnings.warn(
            f"{unconverged} of {len(pa)} Gibbs cross-check estimates did not reach "
            f"tol={tol}; inspect gibbs_mcse_max/unconverged_n in the CSV",
            RuntimeWarning,
            stacklevel=2,
        )


def _run_panel_rep(args, rep, panel, scenario, policies, *, include_oracle):
    data = simulate_ascertained(args, args.seed + rep)
    roles = data["roles"]
    o = roles.index("o")
    estimates = {CASE_CONTROL: data["status"][:, o].astype(float)}
    bounds = {}
    for policy in policies:
        lo, hi = build_bounds(data, args, policy)
        bounds[policy] = (lo, hi)
        # ADuLT uses the identical full personalised proband bounds as LT-FH++,
        # but deliberately omits every relative.  The contrast with FULL is
        # therefore a clean family-history increment.
        policy_roles = ["o"] if policy == "adult" else roles
        policy_lo = lo[:, [o]] if policy == "adult" else lo
        policy_hi = hi[:, [o]] if policy == "adult" else hi
        estimates[POLICY_LABELS[policy]] = estimate_liability_pa_arrays(
            policy_roles, policy_lo, policy_hi, h2=args.h2, out="genetic"
        )[0]
    if include_oracle:
        estimates[ORACLE] = data["g"]

    rows = [_score(name, value, data, args, panel, scenario, rep)
            for name, value in estimates.items()]

    if "full" in bounds and rep < args.gibbs_reps:
        n = min(args.gibbs_n, args.n_fam)
        lo, hi = bounds["full"]
        gibbs, mcse = estimate_liability_gibbs_arrays(
            roles, lo[:n], hi[:n], h2=args.h2, out="genetic",
            n_sim=args.n_sim, burn_in=args.gibbs_burn_in,
            tol=args.gibbs_tol, max_rounds=args.gibbs_max_rounds,
            seed=args.seed + 100_000 + rep,
        )
        full_row = next(row for row in rows if row["phenotype"] == FULL)
        _add_gibbs_diagnostics(
            full_row, estimates[FULL][:n], gibbs, mcse, args.gibbs_tol
        )

    for suffix in ("raw", "adjusted"):
        key = f"mean_chi2_causal_{suffix}"
        base = next(row[key] for row in rows if row["phenotype"] == CASE_CONTROL)
        denominator = base - 1.0
        for row in rows:
            row[f"effN_vs_cc_{suffix}"] = (
                (row[key] - 1.0) / denominator if denominator > 0 else np.nan
            )
    return rows


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


def _contrast_rows(replicate_rows, args, panel, scenario, contrasts):
    panel_rows = [row for row in replicate_rows if row["panel"] == panel]
    by_name = {}
    for row in panel_rows:
        by_name.setdefault(row["phenotype"], {})[row["rep"]] = row

    out = []
    for left, right in contrasts:
        paired_reps = sorted(set(by_name[left]) & set(by_name[right]))
        metadata = _metadata(args, panel, scenario)
        metadata.update(
            row_type="paired_contrast",
            rep="",
            phenotype="",
            comparison=f"{right} - {left}",
            contrast_left=left,
            contrast_right=right,
            contrast_n=len(paired_reps),
        )
        for metric in CONTRAST_METRICS:
            delta = [by_name[right][rep][metric] - by_name[left][rep][metric]
                     for rep in paired_reps]
            mean, sd, se, ci95, _ = _mean_ci(delta)
            metadata[f"delta_{metric}"] = mean
            metadata[f"delta_{metric}_sd"] = sd
            metadata[f"delta_{metric}_se"] = se
            metadata[f"delta_{metric}_ci95"] = ci95
        out.append(metadata)
    return out


def _cip_curve_rows(args):
    ages = np.linspace(20.0, 100.0, 41)
    sex = np.column_stack([np.zeros_like(ages), np.ones_like(ages)])
    birth = np.full_like(sex, BY_REF)
    K, mid = _cip_parameters(sex, birth, args, use_sex=True, use_cohort=False)
    curves = _cip(ages[:, None], K, mid, args.slope)
    rows = []
    for age, male, female in zip(ages, curves[:, 0], curves[:, 1]):
        metadata = _metadata(args, SEX_PANEL, SEX_SCENARIO)
        metadata.update(
            row_type="cip_curve",
            rep="",
            phenotype="",
            comparison="female - male",
            age=float(age),
            cip_male=float(male),
            cip_female=float(female),
            cip_separation=float(female - male),
        )
        rows.append(metadata)
    return rows


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


def _replicate_rows(rows, panel=None):
    selected = [row for row in rows if row["row_type"] == "replicate"]
    return selected if panel is None else [row for row in selected if row["panel"] == panel]


def _contrast_lookup(rows, panel):
    return {row["comparison"]: row for row in rows
            if row["row_type"] == "paired_contrast" and row["panel"] == panel}


def summarize(rows):
    for panel, title in ((MAIN_PANEL, "MAIN INTEGRATED PANEL"),
                         (SEX_PANEL, "PRESPECIFIED SEX-ISOLATION PANEL")):
        selected = _replicate_rows(rows, panel)
        if not selected:
            continue
        print(f"\n{title}")
        print("phenotype                      corr(adj) effN/cc  meanChi2  "
              "lambda strat raw->adj")
        for name in dict.fromkeys(row["phenotype"] for row in selected):
            sub = [row for row in selected if row["phenotype"] == name]
            mean = lambda key: float(np.nanmean([row[key] for row in sub]))
            print(f"{name:30s} {mean('corr_g_adjusted'):8.4f} "
                  f"{mean('effN_vs_cc_adjusted'):7.3f}  "
                  f"{mean('mean_chi2_causal_adjusted'):8.2f}  "
                  f"{mean('lambda_strat_raw'):6.2f}->{mean('lambda_strat_adjusted'):.2f}")

        print("paired contrasts (right - left; t-based 95% CI)")
        for contrast in [row for row in rows
                         if row["row_type"] == "paired_contrast"
                         and row["panel"] == panel]:
            corr = contrast["delta_corr_g_adjusted"]
            corr_ci = contrast["delta_corr_g_adjusted_ci95"]
            effn = contrast["delta_effN_vs_cc_adjusted"]
            effn_ci = contrast["delta_effN_vs_cc_adjusted_ci95"]
            chi2 = contrast["delta_mean_chi2_causal_adjusted"]
            chi2_ci = contrast["delta_mean_chi2_causal_adjusted_ci95"]
            print(f"  {contrast['comparison']}: Δcorr={corr:+.5f} ± {corr_ci:.5f}; "
                  f"ΔeffN={effn:+.4f} ± {effn_ci:.4f}; "
                  f"ΔmeanChi2={chi2:+.3f} ± {chi2_ci:.3f}")
            if panel == SEX_PANEL:
                gap = contrast["delta_mean_error_sex_gap_raw"]
                gap_ci = contrast["delta_mean_error_sex_gap_raw_ci95"]
                print(f"    Δ(female-male mean-error gap)={gap:+.5f} ± {gap_ci:.5f}")

    full = [row for row in _replicate_rows(rows, MAIN_PANEL)
            if row["phenotype"] == FULL and row["gibbs_n_used"]]
    if full:
        print("\nGibbs cross-check diagnostics (first main replicates only)")
        for key in ("gibbs_agreement", "gibbs_mcse_max", "gibbs_unconverged_n",
                    "gibbs_minus_pa_mean", "gibbs_nrmse_score_sd",
                    "gibbs_vs_pa_slope", "gibbs_vs_pa_intercept"):
            print(f"  {key}: {np.mean([row[key] for row in full]):.6g}")


def _means_and_errors(rows, names, key, *, ci=False):
    means, errors = [], []
    for name in names:
        values = [row[key] for row in rows if row["phenotype"] == name]
        mean, _, se, ci95, _ = _mean_ci(values)
        means.append(mean)
        errors.append(ci95 if ci else se)
    return np.asarray(means), np.nan_to_num(errors)


def _short(name):
    return {
        CASE_CONTROL: "case/control",
        ADULT: "ADuLT",
        LTFH: "single K",
        AGE: "age",
        AGE_SEX: "age+sex",
        AGE_COHORT: "age+cohort",
        FULL: "full",
        ORACLE: "oracle",
    }.get(name, name)


def plot(rows, output_prefix):
    plt = get_plt()
    if plt is None:
        return None

    main = _replicate_rows(rows, MAIN_PANEL)
    sex = _replicate_rows(rows, SEX_PANEL)
    main_names = [name for name in dict.fromkeys(row["phenotype"] for row in main)
                  if name != ORACLE]
    fig, axes = plt.subplots(2, 3, figsize=(17, 10.4))
    blue, orange, green, grey = "#3B4A9C", "#D97706", "#15803D", "#6B7280"

    # Main means: useful context, while the adjacent panels foreground paired deltas.
    ax = axes[0, 0]
    means, errors = _means_and_errors(main, main_names, "corr_g_adjusted")
    ax.bar(range(len(main_names)), means, yerr=errors, capsize=3, color=blue)
    ax.set_xticks(range(len(main_names)), [_short(name) for name in main_names],
                  rotation=28, ha="right", fontsize=8)
    ax.set_ylabel("corr(adjusted score, adjusted g)")
    ax.set_title("A  Main-panel adjusted accuracy")

    main_contrasts = [row for row in rows
                      if row["row_type"] == "paired_contrast"
                      and row["panel"] == MAIN_PANEL]
    compact_labels = [f"{_short(row['contrast_right'])} - {_short(row['contrast_left'])}"
                      for row in main_contrasts]

    ax = axes[0, 1]
    y = np.arange(len(main_contrasts))
    ax.errorbar([row["delta_corr_g_adjusted"] for row in main_contrasts], y,
                xerr=np.nan_to_num([row["delta_corr_g_adjusted_ci95"]
                                    for row in main_contrasts]),
                fmt="o", capsize=3, color=green)
    ax.axvline(0.0, color="black", ls=":", lw=1)
    ax.set_yticks(y, compact_labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("paired Δ adjusted correlation")
    ax.set_title("B  Main ablations (95% CI)")

    ax = axes[0, 2]
    effn = [row["delta_effN_vs_cc_adjusted"] for row in main_contrasts]
    effn_ci = np.nan_to_num([row["delta_effN_vs_cc_adjusted_ci95"]
                             for row in main_contrasts])
    ax.errorbar(effn, y, xerr=effn_ci, fmt="o", capsize=3, color=orange)
    ax.axvline(0.0, color="black", ls=":", lw=1)
    ax.set_yticks(y, compact_labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("paired Δ effective-N ratio vs case/control")
    ax.set_title("C  Main causal-SNP signal (95% CI)")

    if sex:
        paired_names = (AGE, AGE_SEX)
        by_rep = {name: {row["rep"]: row for row in sex
                         if row["phenotype"] == name} for name in paired_names}
        reps = sorted(set(by_rep[AGE]) & set(by_rep[AGE_SEX]))

        ax = axes[1, 0]
        curve = [row for row in rows if row["row_type"] == "cip_curve"]
        ages = np.asarray([row["age"] for row in curve])
        male = np.asarray([row["cip_male"] for row in curve])
        female = np.asarray([row["cip_female"] for row in curve])
        ax.plot(ages, male, label="male", color=blue)
        ax.plot(ages, female, label="female", color=orange)
        ax.fill_between(ages, male, female, color=orange, alpha=0.15)
        max_sep = np.max(np.abs(female - male))
        ax.text(0.03, 0.96, f"max |CIP separation|, ages 20–100={max_sep:.3f}",
                transform=ax.transAxes, va="top", fontsize=8)
        ax.set_xlabel("age")
        ax.set_ylabel("cumulative incidence")
        ax.set_title("D  Prespecified sex-specific CIP")
        ax.legend(fontsize=8)

        ax = axes[1, 1]
        for rep in reps:
            ax.plot([0, 1], [by_rep[name][rep]["corr_g_adjusted"]
                             for name in paired_names], color="#9CA3AF", alpha=0.65)
        means, cis = _means_and_errors(sex, paired_names, "corr_g_adjusted", ci=True)
        ax.errorbar([0, 1], means, yerr=cis, fmt="o-", color=green,
                    lw=2.2, capsize=4, label="mean ± 95% CI")
        ax.set_xticks([0, 1], ["age only", "age + sex"])
        ax.set_ylabel("corr(adjusted score, adjusted g)")
        ax.set_title("E  Sex isolation: paired accuracy")
        sex_contrast = _contrast_lookup(rows, SEX_PANEL)[f"{AGE_SEX} - {AGE}"]
        ax.text(0.03, 0.96,
                f"Δcorr={sex_contrast['delta_corr_g_adjusted']:+.5f} "
                f"±{sex_contrast['delta_corr_g_adjusted_ci95']:.5f}\n"
                f"ΔeffN={sex_contrast['delta_effN_vs_cc_adjusted']:+.4f} "
                f"±{sex_contrast['delta_effN_vs_cc_adjusted_ci95']:.4f}\n"
                f"Δmean χ²={sex_contrast['delta_mean_chi2_causal_adjusted']:+.2f} "
                f"±{sex_contrast['delta_mean_chi2_causal_adjusted_ci95']:.2f}",
                transform=ax.transAxes, va="top", fontsize=8)
        ax.legend(fontsize=8, loc="lower right")

        ax = axes[1, 2]
        error_keys = ("mean_error_female_raw", "mean_error_male_raw")
        labels = ("female", "male")
        x = np.arange(2)
        width = 0.36
        for offset, name, color in ((-width / 2, AGE, grey),
                                    (width / 2, AGE_SEX, blue)):
            subset = [row for row in sex if row["phenotype"] == name]
            values = [np.mean([row[key] for row in subset]) for key in error_keys]
            errors = [_mean_ci([row[key] for row in subset])[2] for key in error_keys]
            ax.bar(x + offset, values, width, yerr=np.nan_to_num(errors), capsize=3,
                   label=_short(name), color=color)
        ax.axhline(0.0, color="black", ls=":", lw=1)
        ax.set_xticks(x, labels)
        ax.set_ylabel("mean(score - true g)")
        ax.set_title("F  Sex-specific mean error (raw score scale)")
        ax.text(
            0.03, 0.04,
            "Δ(female−male error gap)="
            f"{sex_contrast['delta_mean_error_sex_gap_raw']:+.4f} "
            f"±{sex_contrast['delta_mean_error_sex_gap_raw_ci95']:.4f}",
            transform=ax.transAxes, va="bottom", fontsize=8,
        )
        ax.legend(fontsize=8)
    else:
        for ax in axes[1]:
            ax.axis("off")
            ax.text(0.5, 0.5, "sex-isolation panel disabled",
                    ha="center", va="center")

    main_cfg = main[0] if main else {}
    sex_cfg = sex[0] if sex else {}
    fig.suptitle("LT-FH++ decomposition, matched ADuLT contrast, and sex isolation",
                 fontsize=15, y=0.995)
    fig.text(
        0.5, 0.012,
        "Mean bars show ±SE; paired contrast points and thick sex-isolation means "
        "show t-based 95% CIs. Thin lines are independent paired replicates.  "
        f"Main: N={main_cfg.get('n_fam', '—')}, M={main_cfg.get('m_snps', '—')}, "
        f"R={main_cfg.get('panel_reps', '—')}.  Sex isolation: "
        f"N={sex_cfg.get('n_fam', '—')}, M={sex_cfg.get('m_snps', '—')}, "
        f"R={sex_cfg.get('panel_reps', '—')}, RR={sex_cfg.get('sex_rr', '—')}, "
        "cohort effects off, equal onset midpoints.",
        ha="center", va="bottom", fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.965))
    path = f"{output_prefix}.png"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _sex_isolation_args(args):
    values = vars(args).copy()
    values.update(
        n_fam=args.sex_isolation_n_fam,
        m_snps=args.sex_isolation_m_snps,
        n_causal=args.sex_isolation_n_causal,
        n_strat=args.sex_isolation_n_strat,
        reps=args.sex_isolation_reps,
        seed=args.sex_isolation_seed,
        sex_rr=2.0,
        sex_mid_diff=0.0,
        trend_R=1.0,
        cohort_mid_shift=0.0,
        death_sex_diff=0.0,
        truth_use_sex=True,
        truth_use_cohort=False,
        cohort_genotype_stratification=False,
        gibbs_reps=0,
    )
    return argparse.Namespace(**values)


def _validate_design(parser, args, prefix=""):
    label = f"{prefix} " if prefix else ""
    if args.n_fam < 20:
        parser.error(f"{label}--n-fam must be at least 20")
    if args.m_snps < 3:
        parser.error(f"{label}--m-snps must be at least 3")
    if args.n_causal < 1:
        parser.error(f"{label}--n-causal must be positive")
    if args.n_strat < 2:
        parser.error(f"{label}--n-strat must be at least 2 for lambda GC")
    if args.n_causal + args.n_strat >= args.m_snps:
        parser.error(f"{label}need at least one independent null SNP beyond causal "
                     "+ stratified SNPs")
    if args.reps < 1:
        parser.error(f"{label}--reps must be positive")
    n_cases = int(round(args.case_frac * args.n_fam))
    if not 1 <= n_cases < args.n_fam:
        parser.error(f"{label}--case-frac and --n-fam must imply at least one "
                     "case and one control")


def _validate_args(parser, args):
    numeric = (args.h2, args.K, args.case_frac, args.mid, args.slope,
               args.sex_rr, args.sex_mid_diff, args.trend_R,
               args.cohort_mid_shift, args.death_sex_diff, args.gibbs_tol)
    if not np.all(np.isfinite(numeric)):
        parser.error("all model parameters must be finite")
    if not 0 < args.h2 < 1:
        parser.error("--h2 must be strictly between 0 and 1")
    if not 0 < args.K < 0.8:
        parser.error("--K must be strictly between 0 and 0.8")
    if not 0 < args.case_frac < 1:
        parser.error("--case-frac must be strictly between 0 and 1")
    if args.slope <= 0:
        parser.error("--slope must be positive")
    if args.sex_rr <= 0 or args.trend_R <= 0:
        parser.error("--sex-rr and --trend-R must be positive")
    if args.death_sex_diff < -DEATH_MEAN_MALE:
        parser.error("--death-sex-diff makes the female death-age mean nonpositive")
    if len(args.fam) != len(set(args.fam)):
        parser.error("--fam roles must be unique")
    if any(role in {"g", "o"} for role in args.fam):
        parser.error("--fam lists relatives only; g and o are added automatically")
    try:
        construct_covmat_single(args.fam, add_ind=True, h2=args.h2)
    except (TypeError, ValueError) as exc:
        parser.error(f"invalid --fam structure: {exc}")

    _validate_design(parser, args)
    if args.gibbs_reps < 0 or args.gibbs_reps > args.reps:
        parser.error("--gibbs-reps must be between 0 and --reps")
    if args.gibbs_reps and args.gibbs_n < 2:
        parser.error("--gibbs-n must be at least 2 when --gibbs-reps is positive")
    if args.gibbs_n < 0:
        parser.error("--gibbs-n must be nonnegative")
    if args.n_sim < 16:
        parser.error("--n-sim must be at least 16 for batch-means MCSE")
    if args.gibbs_burn_in < 0:
        parser.error("--gibbs-burn-in must be nonnegative")
    if args.gibbs_tol <= 0:
        parser.error("--gibbs-tol must be positive")
    if args.gibbs_max_rounds < 1:
        parser.error("--gibbs-max-rounds must be positive")

    if not args.no_sex_isolation:
        sex_args = _sex_isolation_args(args)
        _validate_design(parser, sex_args, "sex-isolation")
        if args.sex_isolation_seed < 0:
            parser.error("--sex-isolation-seed must be nonnegative")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if not args.output_prefix.strip():
        parser.error("--output-prefix must not be empty")
    if args.output_prefix.endswith((".csv", ".png")):
        parser.error("--output-prefix is a path stem; omit .csv/.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-fam", type=int, default=4000)
    parser.add_argument("--m-snps", type=int, default=1200)
    parser.add_argument("--n-causal", type=int, default=30)
    parser.add_argument("--n-strat", type=int, default=300,
                        help="null SNPs correlated with sex and birth cohort")
    parser.add_argument("--fam", nargs="+", default=STRUCT)
    parser.add_argument("--h2", type=float, default=0.5)
    parser.add_argument(
        "--K", type=float, default=0.12,
        help="geometric-center lifetime prevalence at the reference cohort",
    )
    parser.add_argument("--case-frac", type=float, default=0.25)
    parser.add_argument("--mid", type=float, default=60.0)
    parser.add_argument("--slope", type=float, default=1.0 / 8.0)
    parser.add_argument("--sex-rr", type=float, default=2.0,
                        help="female:male lifetime-prevalence ratio")
    parser.add_argument("--sex-mid-diff", type=float, default=6.0,
                        help="female-minus-male onset midpoint in years")
    parser.add_argument("--trend-R", type=float, default=2.0,
                        help="lifetime-prevalence ratio per 30 birth years")
    parser.add_argument("--cohort-mid-shift", type=float, default=-3.0,
                        help="onset-midpoint shift per 30 birth years")
    parser.add_argument("--death-sex-diff", type=float, default=5.0,
                        help="female-minus-male mean competing-death age")
    parser.add_argument("--reps", type=int, default=10,
                        help="independent PA replicates for the main panel")
    parser.add_argument("--gibbs-reps", type=int, default=2,
                        help="first main-panel replicates receiving a Gibbs cross-check")
    parser.add_argument("--gibbs-n", type=int, default=300)
    parser.add_argument("--n-sim", type=int, default=25_000)
    parser.add_argument("--gibbs-burn-in", type=int, default=800)
    parser.add_argument("--gibbs-tol", type=float, default=0.03)
    parser.add_argument("--gibbs-max-rounds", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)

    sex = parser.add_argument_group("prespecified sex-isolation panel")
    sex.add_argument("--no-sex-isolation", action="store_true",
                     help="skip the otherwise-default sex-isolation panel")
    sex.add_argument("--sex-isolation-reps", type=int, default=5)
    sex.add_argument("--sex-isolation-n-fam", type=int, default=3000)
    sex.add_argument("--sex-isolation-m-snps", type=int, default=600)
    sex.add_argument("--sex-isolation-n-causal", type=int, default=30)
    sex.add_argument("--sex-isolation-n-strat", type=int, default=150,
                     help="sex-correlated null SNPs (cohort coefficients are zero)")
    sex.add_argument("--sex-isolation-seed", type=int, default=100_001)

    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX,
                        help="path stem for .csv and .png outputs")
    args = parser.parse_args()
    args.truth_use_sex = True
    args.truth_use_cohort = True
    args.cohort_genotype_stratification = True
    args.output_prefix = os.path.abspath(args.output_prefix)
    _validate_args(parser, args)

    replicate_rows = []
    for rep in range(args.reps):
        print(f"main replicate {rep + 1}/{args.reps}")
        replicate_rows.extend(_run_panel_rep(
            args, rep, MAIN_PANEL, MAIN_SCENARIO, MAIN_POLICIES,
            include_oracle=True,
        ))

    sex_args = None
    if not args.no_sex_isolation:
        sex_args = _sex_isolation_args(args)
        for rep in range(sex_args.reps):
            print(f"sex-isolation replicate {rep + 1}/{sex_args.reps}")
            replicate_rows.extend(_run_panel_rep(
                sex_args, rep, SEX_PANEL, SEX_SCENARIO, SEX_POLICIES,
                include_oracle=False,
            ))

    rows = list(replicate_rows)
    rows.extend(_contrast_rows(
        replicate_rows, args, MAIN_PANEL, MAIN_SCENARIO, MAIN_CONTRASTS
    ))
    if sex_args is not None:
        rows.extend(_contrast_rows(
            replicate_rows, sex_args, SEX_PANEL, SEX_SCENARIO, SEX_CONTRASTS
        ))
        rows.extend(_cip_curve_rows(sex_args))

    summarize(rows)
    csv_path = write_csv(rows, args.output_prefix)
    png_path = plot(rows, args.output_prefix)
    written = os.path.basename(csv_path)
    if png_path is not None:
        written += f" and {os.path.basename(png_path)}"
    print(f"\nwrote {written}")


if __name__ == "__main__":
    main()
