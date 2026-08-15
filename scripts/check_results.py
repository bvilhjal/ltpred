#!/usr/bin/env python3
"""RESULTS.md headline-number integrity checks against the committed artifacts.

Hand-transcribed summary tables drift: the 2026-08-07 independent review
(``docs/EXTERNAL_REVIEW.md``) found that roughly a third of the tables in
``benchmarks/RESULTS.md`` had gone stale against the committed benchmark
artifacts. pytest and mkdocs cannot see that class of error -- the numbers
are prose. This script is the guard: each load-bearing headline value is
re-derived from the committed artifacts (``benchmarks/*.csv`` and
``benchmarks/run_logs/*.stdout.log``) and compared with the value quoted in
RESULTS.md at its quoted precision.

Each guard is data, not prose parsing:

- ``name`` -- what is being guarded;
- ``artifact`` -- repo-relative path of the artifact to recompute from, or
  ``"newest:"`` + a glob to recompute from the newest matching run log;
- ``recompute`` -- a small function (csv-module reads, mean/SE/min/max/ratio)
  returning the recomputed values in anchor-group order;
- ``anchor`` -- a regex whose capture groups quote the value(s) in RESULTS.md.

The default tolerance is exact-at-quoted-precision: a recomputed value passes
when it rounds to the quoted string (within half a unit of its last decimal).
Switching a guard to a different artifact (e.g. a run log to a future CSV) is
a one-line change to its ``artifact`` field.

Run it from the repository root::

    python scripts/check_results.py

Exits non-zero (and prints computed vs quoted) when any guard mismatches.
"""
from __future__ import annotations

import csv
import math
import pathlib
import re
from dataclasses import dataclass
from typing import Callable

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS_MD = ROOT / "benchmarks" / "RESULTS.md"


# --- guard machinery ------------------------------------------------------


@dataclass
class Guard:
    name: str
    artifact: str
    recompute: Callable[[pathlib.Path], tuple]
    anchor: str
    document: str = "benchmarks/RESULTS.md"


def parse_quoted(text: str):
    """Return (value, decimals) for a quoted number like ``71,264`` or ``+0.001``."""
    cleaned = text.replace(",", "").replace("−", "-").strip()
    return float(cleaned), len(cleaned.partition(".")[2])


def within_quoted_precision(computed: float, quoted_text: str) -> bool:
    """True when ``computed`` rounds to the quoted string at its precision."""
    value, decimals = parse_quoted(quoted_text)
    return abs(computed - value) <= 0.5 * 10 ** -decimals + 1e-12


def quoted_tuples(results_text: str, anchor: str) -> list:
    """All distinct quoted-value tuples captured by the anchor regex."""
    found = re.findall(anchor, results_text, flags=re.MULTILINE)
    tuples = [m if isinstance(m, tuple) else (m,) for m in found]
    return sorted(set(tuples))


def resolve_artifact(root: pathlib.Path, artifact: str) -> pathlib.Path:
    if artifact.startswith("newest:"):
        matches = sorted(root.glob(artifact.removeprefix("newest:")))
        if not matches:
            raise FileNotFoundError(f"no run log matches {artifact!r}")
        return matches[-1]
    return root / artifact


def run_guard(root: pathlib.Path, results_text: str, guard: Guard):
    """Return (ok, detail); recomputes and compares one guard."""
    try:
        path = resolve_artifact(root, guard.artifact)
        computed = tuple(guard.recompute(path))
    except (OSError, ValueError, KeyError) as exc:
        return False, f"artifact error [{guard.artifact}]: {exc}"
    quoted = quoted_tuples(results_text, guard.anchor)
    if not quoted:
        return False, f"anchor not found in RESULTS.md: {guard.anchor}"
    if len(quoted) > 1:
        return False, f"anchor matched {len(quoted)} distinct quoted values: {quoted}"
    quoted_tuple = quoted[0]
    if len(quoted_tuple) != len(computed):
        return False, (f"anchor captures {len(quoted_tuple)} values but recompute "
                       f"returns {len(computed)}")
    for value, text in zip(computed, quoted_tuple):
        if not within_quoted_precision(value, text):
            shown = ", ".join(f"{v:.{parse_quoted(t)[1] + 2}f}"
                              for v, t in zip(computed, quoted_tuple))
            return False, (f"computed ({shown}) vs quoted "
                           f"({', '.join(quoted_tuple)}) [{guard.artifact}]")
    return True, ", ".join(quoted_tuple)


def run_guards(root: pathlib.Path):
    cache = {}
    out = []
    for guard in GUARDS:
        doc = getattr(guard, "document", "benchmarks/RESULTS.md")
        if doc not in cache:
            cache[doc] = (root / doc).read_text(encoding="utf-8")
        out.append((guard, *run_guard(root, cache[doc], guard)))
    return out


# --- recomputation helpers -------------------------------------------------


def _rows(path: pathlib.Path) -> list:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _mean_se(values: list):
    n = len(values)
    mean = sum(values) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    return mean, sd / math.sqrt(n)


def _only(rows: list, **criteria) -> dict:
    out = [r for r in rows if all(r[k] == v for k, v in criteria.items())]
    if len(out) != 1:
        raise ValueError(f"expected exactly 1 row for {criteria}, got {len(out)}")
    return out[0]


def _metric_values(path: pathlib.Path, metric: str) -> list:
    """Per-replicate values of one metric in a long-format rep/metric/value CSV
    (rep 0 marks single-run parts and is excluded)."""
    vals = [float(r["value"]) for r in _rows(path)
            if r["metric"] == metric and int(r["rep"]) > 0]
    if not vals:
        raise ValueError(f"{path.name}: no replicate rows for metric {metric!r}")
    return vals


# Two-sided 97.5% t criticals for the paired-contrast 95% CIs (df = R - 1).
T_CRIT_975 = {1: 12.7062, 2: 4.3027, 3: 3.1824, 4: 2.7764, 9: 2.2622, 24: 2.0639}


def _mean_se_ci(vals: list, df: int) -> tuple:
    mean, se = _mean_se(vals)
    half = T_CRIT_975[df] * se
    return (mean, se, mean - half, mean + half)


# --- recomputations: bench_accuracy.csv (RESULTS §1) ------------------------


def accuracy_agreement_range(path):
    vals = [float(r["gibbs_pa_corr"]) for r in _rows(path)]
    return (min(vals), max(vals))


def accuracy_agreement_se_ceiling(path):
    return (max(float(r["se_gibbs_pa_corr"]) for r in _rows(path)),)


def _accuracy_cell(path, structure):
    row = _only(_rows(path), structure=structure, h2="0.5", prevalence="0.05")
    return tuple(float(row[c]) for c in (
        "corr_gibbs", "se_corr_gibbs", "corr_pa", "se_corr_pa",
        "eff_n_proxy_pa", "se_eff_n_proxy_pa", "gibbs_pa_corr"))


def accuracy_cell_parents(path):
    return _accuracy_cell(path, "parents")


def accuracy_cell_parents_sibs(path):
    return _accuracy_cell(path, "parents+sibs")


def accuracy_cell_extended(path):
    return _accuracy_cell(path, "extended")


def accuracy_effn_grid_max(path):
    row = max(_rows(path), key=lambda r: float(r["eff_n_proxy_pa"]))
    return (float(row["eff_n_proxy_pa"]), float(row["se_eff_n_proxy_pa"]))


def accuracy_effn_former_max(path):
    row = _only(_rows(path), structure="parents+sibs", h2="0.2", prevalence="0.05")
    return (float(row["eff_n_proxy_pa"]), float(row["se_eff_n_proxy_pa"]))


# --- recomputations: bench_scaling.csv (RESULTS §2) -------------------------


def scaling_object_speedup_range(path):
    vals = [float(r["speedup"]) for r in _rows(path) if r["axis"] == "family_size"]
    return (min(vals), max(vals))


def scaling_array_throughput_range(path):
    vals = [float(r["fam_per_s_pa_array"]) for r in _rows(path)]
    return (min(vals) / 1e6, max(vals) / 1e6)


def scaling_array_vs_object_range(path):
    vals = [float(r["pa_array_speedup"]) for r in _rows(path)]
    return (min(vals), max(vals))


def scaling_object_pa_rate_k_range(path):
    vals = [float(r["fam_per_s_pa"]) for r in _rows(path)]
    return (min(vals) / 1e3, max(vals) / 1e3)


def scaling_nfam_500_row(path):
    row = _only(_rows(path), axis="n_fam", value="500")
    return (float(row["fam_per_s_gibbs"]), float(row["fam_per_s_pa"]),
            float(row["fam_per_s_pa_array"]) / 1e6, float(row["speedup"]))


# --- recomputations: bench_gwas_power.csv (RESULTS §4) -----------------------


def _gwas_power_ncp(path, phenotype):
    row = _only(_rows(path), phenotype=phenotype)
    return (float(row["effN_vs_cc"]), float(row["se_effN_vs_cc"]))


def gwas_power_ncp_gibbs(path):
    return _gwas_power_ncp(path, "LT-FH (Gibbs)")


def gwas_power_ncp_pa(path):
    return _gwas_power_ncp(path, "LT-FH (PA)")


# --- recomputations: bench_fit_heritability.csv (RESULTS §5) -----------------


def fit_heritability_h02_row(path):
    row = _only(_rows(path), panel="bias_precision", h2="0.2")
    bias, sd, se = (float(row[c]) for c in ("bias", "sd", "reported_se"))
    return (0.2 + bias, bias, sd, se, sd / se)


def fit_heritability_structure_sds(path):
    rows = _rows(path)
    return tuple(float(_only(rows, panel="structure", structure=s)["sd"])
                 for s in ("parents", "parents+2 sibs", "extended"))


# --- recomputations: bench_variance_components.csv (RESULTS §6) ---------------


def variance_components_recovery_row(path):
    row = _only(_rows(path), panel="recovery", a2="0.4", c2="0.2")
    return (0.4 + float(row["A_bias"]), float(row["A_sd"]),
            0.2 + float(row["C_bias"]), float(row["C_sd"]))


def variance_components_scaling_sds(path):
    rows = _rows(path)

    def sds(n_fam):
        row = _only(rows, panel="scaling", n_fam=n_fam)
        return float(row["A_sd"]), float(row["C_sd"])

    a_1000, c_1000 = sds("1000")
    a_8000, c_8000 = sds("8000")
    return (a_1000, a_8000, c_1000, c_8000)


# --- recomputations: bench_genetic_correlation.csv (RESULTS §7) ---------------


def genetic_correlation_null_row(path):
    row = _only(_rows(path), panel="bias_precision", rg="0.0")
    bias = float(row["bias"])
    return (0.0 + bias, bias, float(row["sd"]))


# --- recomputations: bench_calibration.csv (RESULTS §12) ----------------------


def calibration_psibs_k001_row(path):
    row = _only(_rows(path), panel="correct", structure="parents+sibs",
                prevalence="0.01")
    return tuple(float(row[c]) for c in (
        "slope_pa", "se_slope_pa", "corr_pa", "se_corr_pa",
        "top_ratio_pa", "se_top_ratio_pa"))


def calibration_extended_k001_top_ratio(path):
    row = _only(_rows(path), panel="correct", structure="extended",
                prevalence="0.01")
    return (float(row["top_ratio_pa"]), float(row["se_top_ratio_pa"]))


def calibration_misspec_slope_sweep(path):
    rows = _rows(path)
    lo = _only(rows, panel="misspec", assumed_h2="0.2")
    hi = _only(rows, panel="misspec", assumed_h2="0.8")
    return (float(lo["slope_pa"]), float(lo["se_slope_pa"]),
            float(hi["slope_pa"]), float(hi["se_slope_pa"]))


# --- recomputations: bench_ltfhpp_personalization.csv (RESULTS §15) -----------


def _ltfhpp_main_means(path, phenotype, column):
    vals = [float(r[column]) for r in _rows(path)
            if r["row_type"] == "replicate" and r["panel"] == "main_ablation"
            and r["phenotype"] == phenotype]
    return _mean_se(vals)


def ltfhpp_adult_ncp(path):
    return _ltfhpp_main_means(path, "ADuLT (no family history)",
                              "effN_vs_cc_adjusted")


def ltfhpp_full_ncp(path):
    return _ltfhpp_main_means(path, "LT-FH++ (full CIP)", "effN_vs_cc_adjusted")


def _ltfhpp_contrast(path, column):
    rows = [r for r in _rows(path) if r["row_type"] == "paired_contrast"]
    row = _only(rows, comparison="LT-FH++ (full CIP) - ADuLT (no family history)")
    return (float(row[column]), float(row[column + "_ci95"]))


def ltfhpp_ncp_increment(path):
    return _ltfhpp_contrast(path, "delta_effN_vs_cc_adjusted")


def ltfhpp_corr_increment(path):
    return _ltfhpp_contrast(path, "delta_corr_g_adjusted")


# --- recomputations: bench_sex_limitation.csv (RESULTS §26) -------------------


def sex_limitation_replicate_count(path):
    reps = {r["reps"] for r in _rows(path)}
    if len(reps) != 1:
        raise ValueError(f"mixed replicate counts: {sorted(reps)}")
    return (float(reps.pop()),)


# --- recomputations: bench_pedigree_inference.csv (RESULTS §20) -------------
# The §20/§21 replication landed these long-format CSVs; before that the guards
# recomputed from the newest run_logs/*.stdout.log (see resolve_artifact's
# "newest:" support -- switching a guard's artifact back is a one-line change).


def pedigree_payoff(path):
    out = []
    for metric in ("part2_corr_all_relatives", "part2_corr_named_role"):
        out += _mean_se(_metric_values(path, metric))
    return tuple(out)


def pedigree_payoff_contrast(path):
    return _mean_se_ci(_metric_values(path, "part2_delta_corr_all_minus_role"),
                       df=4)


# --- recomputations: bench_register_pipeline.csv (RESULTS §21) --------------


def register_degree_accuracy(path):
    out = []
    for metric in ("part1_corr_degree3", "part1_corr_degree1"):
        out += _mean_se(_metric_values(path, metric))
    return tuple(out)


def register_relatives_events_delta_auc(path):
    return _mean_se_ci(
        _metric_values(path, "part3_delta_auc_leak_rel_minus_honest"), df=4)


def register_own_outcome_delta_corr(path):
    return _mean_se_ci(
        _metric_values(path, "part3_delta_corr_leak_own_minus_honest"), df=4)


# --- the guard list -----------------------------------------------------------

_ACCURACY = "benchmarks/bench_accuracy.csv"
_ACCURACY_ROW = (r"([\d.]+) ± ([\d.]+) \| ([\d.]+) ± ([\d.]+) \| "
                 r"([\d.]+) ± ([\d.]+)× \| ([\d.]+) \|$")


_ASCERT = "benchmarks/bench_ascertainment.csv"
_ASCERT_NULL = "benchmarks/bench_ascertainment_h2null.csv"


def _ascert_rows(path, **where):
    rows = _rows(path)
    return [r for r in rows
            if all(r[k] == v for k, v in where.items())]


def ascert_h2_nuclear_population(path):
    r = _ascert_rows(path, arm="h2", structure="nuclear", scheme="population")[0]
    return (float(r["fitted_mean"]), float(r["sd"]))


def ascert_h2_nuclear_random50(path):
    r = _ascert_rows(path, arm="h2", structure="nuclear", scheme="random_50")[0]
    return (float(r["fitted_mean"]), float(r["sd"]))


def ascert_ac_saturated_residual(path):
    """Every ascertained A+C cell must have exhausted the residual variance."""
    rows = [r for r in _rows(path) if r["arm"] == "AC"
            and r["scheme"] not in ("population", "random_50")]
    return (max(float(r["residual"]) for r in rows),)


def ascert_proband_case_moment(path):
    r = _ascert_rows(path, arm="mechanism", structure="nuclear",
                     scheme="proband_case")[0]
    unc = float(r["he_uncentered"])
    return (unc, unc / float(r["truth"]), float(r["he_centered"]))


def ascert_scale_invariance(path):
    """proband_case bias is constant in N -- the bias-not-noise claim."""
    rows = _ascert_rows(path, arm="scale", scheme="proband_case")
    return tuple(float(r["bias"]) for r in sorted(rows, key=lambda x: int(x["n_fam"])))


def ascert_null_population(path):
    r = _ascert_rows(path, arm="h2", scheme="population")[0]
    return (float(r["fitted_mean"]),)


def ascert_null_random50(path):
    r = _ascert_rows(path, arm="h2", scheme="random_50")[0]
    return (float(r["fitted_mean"]),)


def ascert_ipw_case_control(path):
    r = _ascert_rows(path, arm="ipw", scheme="case_control")[0]
    return (float(r["max_weight"]), float(r["unweighted_mean"]),
            float(r["fitted_mean"]), float(r["sd"]))


def ascert_ipw_enriched(path):
    r = _ascert_rows(path, arm="ipw", scheme="enriched_20")[0]
    return (float(r["max_weight"]), float(r["unweighted_mean"]),
            float(r["fitted_mean"]), float(r["sd"]))


def ascert_ipw_cc_mean(path):
    return (ascert_ipw_case_control(path)[2],)


def ascert_ipw_enr_mean(path):
    return (ascert_ipw_enriched(path)[2],)


# --- recomputations: bench_confounding.csv (RESULTS §13) --------------------


def confounding_r4_row(path):
    row = _only(_rows(path), trend_R="4.0")
    return (
        float(row["lgc_strat_cohort"]), float(row["se_lgc_strat_cohort"]),
        float(row["lgc_strat_single_K"]), float(row["se_lgc_strat_single_K"]),
        float(row["lgc_strat_casecontrol"]), float(row["se_lgc_strat_casecontrol"]),
    )


# --- recomputations: bench_pgs_comparison.csv (RESULTS §28) -----------------


def age_onset_h05_k03(path):
    row = _only(_rows(path), h2="0.5", prevalence="0.3")
    return (
        float(row["corr_ltfh_pa"]), float(row["se_ltfh_pa"]),
        float(row["corr_age_onset_pa"]), float(row["se_age_onset_pa"]),
        float(row["corr_interval_pa"]), float(row["se_interval_pa"]),
        float(row["gain"]), float(row["se_gain"]),
        float(row["gain_interval"]), float(row["se_gain_interval"]),
    )


def mixture_dependent_mid_pin_slope(path):
    vals = [float(r["slope_pa"]) for r in _rows(path)
            if r.get("row_type") == "replicate"
            and r.get("onset_model") == "dependent"
            and r.get("regime") == "mid"
            and r.get("arm") == "pinned + no-mixture"]
    if not vals:
        raise ValueError("no dependent/mid/pinned-no-mixture rows")
    return (_mean_se(vals)[0],)


def mixture_largest_abs_corr_shift(path):
    """Largest absolute paired mixture-minus-no-mixture correlation shift."""
    rows = [r for r in _rows(path) if r.get("row_type") == "paired_contrast"]
    if not rows:
        raise ValueError("no paired-contrast rows")
    row = max(rows, key=lambda r: abs(float(r["delta_corr_pa"])))
    return (float(row["delta_corr_pa"]), float(row["delta_corr_pa_ci95"]))


def tetrachoric_falconer(path):
    row = _only(_rows(path), pair="falconer_h2")
    return (float(row["tetrachoric"]), float(row["se"]))


def ltfhplus_gibbs_corr(path):
    vals = [float(r["corr_gibbs_ltfhplus"]) for r in _rows(path)]
    return (_mean_se(vals)[0],)


def ltfhplus_gibbs_rmse(path):
    return _mean_se([float(r["rmse_gibbs_ltfhplus"]) for r in _rows(path)])


def ltfgrs_pa_lock(path):
    corr = [float(r["corr_pa_ltfgrs"]) for r in _rows(path)]
    rmse = [float(r["rmse_pa_ltfgrs"]) for r in _rows(path)]
    return (_mean_se(corr)[0],) + _mean_se(rmse)


def ltfhplus_seconds(path):
    rows = _rows(path)

    def pair(key):
        return _mean_se([float(row[key]) for row in rows])

    return (
        pair("seconds_ltfhplus")
        + pair("ms_per_fam_ltfhplus")
        + pair("seconds_ltpred_gibbs")
        + pair("ms_per_fam_ltpred_gibbs")
        + pair("seconds_ltfgrs_pa")
        + pair("ms_per_fam_ltfgrs_pa")
        + pair("seconds_ltpred_pa")
        + pair("ms_per_fam_ltpred_pa")
    )


def ltfhplus_folds(path):
    rows = _rows(path)

    def pair(key):
        return _mean_se([float(r[key]) for r in rows])

    return (
        pair("fold_ltfhplus_over_gibbs")
        + pair("fold_ltfgrs_over_pa")
        + pair("fold_ltfhplus_over_pa")
    )


def ltfhplus_peak_mib(path):
    rows = _rows(path)
    mib = 1024.0 * 1024.0

    def pair(key):
        return _mean_se([float(row[key]) / mib for row in rows])

    return (
        pair("peak_rss_bytes_ltfhplus")
        + pair("peak_rss_bytes_ltfgrs_pa")
        + pair("peak_rss_bytes_ltpred_gibbs")
        + pair("peak_rss_bytes_ltpred_pa")
    )


def liability_lee_bridge(path):
    row = _only(_rows(path), route="(c) Lee-2011 bridged")
    return (float(row["mean"]), float(row["se"]))


def pgs_test_r2s(path):
    rows = [r for r in _rows(path) if r["row_type"] == "replicate"]

    def ms(arm):
        vals = [float(r["r2_g"]) for r in rows
                if r["arm"] == arm and r["r2_g"] not in ("", "nan")]
        return _mean_se(vals)

    pgs, ltfh, joint = ms("PGS"), ms("LT-FH (PA)"), ms("PGS + LT-FH joint")
    return (pgs[0], pgs[1], ltfh[0], ltfh[1], joint[0], joint[1])


# --- the guard list -----------------------------------------------------------


def ascert_dose_row(enrich_lo, enrich_hi):
    """The dose-response cell whose realised enrichment falls in a band.

    Keyed on the measured enrichment rather than the swept target, because the
    realised case share is what the table reports and what moves between runs.
    """
    def recompute(path):
        for r in _rows(path):
            if r["arm"] != "dose" or not r.get("enrichment"):
                continue
            if enrich_lo <= float(r["enrichment"]) < enrich_hi:
                return (float(r["fitted_mean"]),)
        raise AssertionError(f"no dose row in [{enrich_lo}, {enrich_hi})")
    return recompute


def ascert_dose_mild_summary(path):
    """Realised rate, assumed rate, enrichment and bias in the mild-dose cell."""
    rows = [r for r in _rows(path)
            if r["arm"] == "dose" and r.get("enrichment")
            and 1.15 <= float(r["enrichment"]) < 1.35]
    row = _only(rows)
    return (100.0 * float(row["case_frac"]), 100.0 * float(row["prev"]),
            float(row["enrichment"]), float(row["bias"]))


def ascert_dose_mild_rates_bias(path):
    case_pct, prev_pct, _, bias = ascert_dose_mild_summary(path)
    return case_pct, prev_pct, bias


def ascert_dose_mild_enrichment_bias(path):
    _, _, enrichment, bias = ascert_dose_mild_summary(path)
    return enrichment, bias


def ascert_specificity_total(path):
    """Total false positives across the specificity grid -- must be 0."""
    rows = [r for r in _rows(path) if r["arm"] == "specificity"]
    assert rows, "no specificity rows"
    return (sum(float(r["fitted_mean"]) for r in rows),)


def ascert_lee(scheme):
    def recompute(path):
        r = _ascert_rows(path, arm="lee", scheme=scheme)[0]
        return (float(r["fitted_mean"]),)
    return recompute


def _couple_env_ignore_rows(path):
    """The bench_couple_env omission panel, ordered by its s² column."""
    rows = sorted((r for r in _rows(path) if r["panel"] == "ignore_bias"),
                  key=lambda r: float(r["s2"]))
    if len(rows) != 3:
        raise ValueError(f"expected 3 ignore_bias rows, found {len(rows)}")
    return rows


def couple_env_omission_cells(path):
    """C- and M-omission biases in Â with SDs, per s² = 0.1/0.2/0.3 cell.

    Returns 12 values in table order: for each cell,
    (C bias, C SD, M bias, M SD)."""
    return tuple(v for r in _couple_env_ignore_rows(path)
                 for v in (float(r["A_ignoreC_bias"]), float(r["A_ignoreC_sd"]),
                           float(r["A_ignoreM_bias"]), float(r["A_ignoreM_sd"])))


def couple_env_omission_bias_prose(path):
    """The six omission-bias values algorithm.md quotes in its prose.

    Prose order: the three C-omission biases, then the three M-omission
    biases, each at s² = 0.1/0.2/0.3."""
    rows = _couple_env_ignore_rows(path)
    return (tuple(float(r["A_ignoreC_bias"]) for r in rows)
            + tuple(float(r["A_ignoreM_bias"]) for r in rows))


GUARDS = [
    Guard("ascertainment-dose-mild-prose", _ASCERT,
          ascert_dose_mild_rates_bias,
          r"A ([\d.]+)% case rate against an assumed ([\d.]+)% inflates h² by ([+]?\d+(?:\.\d+)?)"),
    Guard("inference-ascertainment-dose-mild-prose", _ASCERT,
          ascert_dose_mild_rates_bias,
          r"A ([\d.]+)% case rate against an assumed ([\d.]+)% already inflates `h²` by ([+]?\d+(?:\.\d+)?)",
          document="docs/inference.md"),
    Guard("inference-ascertainment-enrichment-prose", _ASCERT,
          ascert_dose_mild_enrichment_bias,
          r"a ([\d.]+)× enrichment already inflates `h²` by\s*([+]?\d+(?:\.\d+)?)",
          document="docs/inference.md"),
    Guard("assumptions-ascertainment-dose-prose", _ASCERT,
          ascert_dose_mild_enrichment_bias,
          r"case rate only ([\d.]+)×\s*the assumed prevalence already inflates `h²` by ([+]?\d+(?:\.\d+)?)",
          document="docs/assumptions.md"),
    Guard("report-ascertainment-dose-prose", _ASCERT,
          ascert_dose_mild_summary,
          r"case share of \$([\d.]+)\\%\$ against an\s*asserted \$([\d.]+)\\%\$ \(\$([\d.]+)\\times\$\) already inflates \$\\h\$ by \$([+]?[\d.]+)\$",
          document="report/ltpred_methods.tex"),
    Guard("mixture-largest-corr-shift-headline",
          "benchmarks/bench_pafgrs_mixture.csv",
          mixture_largest_abs_corr_shift,
          r"largest shift is\s*only ([−-][\d.]+) \(95% CI ± (\d+(?:\.\d+)?)\)"),
    Guard("mixture-largest-corr-shift-verdict",
          "benchmarks/bench_pafgrs_mixture.csv",
          mixture_largest_abs_corr_shift,
          r"largest shift is\s*([-][\d.]+) with a 95% CI half-width of (\d+(?:\.\d+)?)"),
    Guard("readme-mixture-largest-corr-shift",
          "benchmarks/bench_pafgrs_mixture.csv",
          mixture_largest_abs_corr_shift,
          r"largest correlation shift is\s*only ([−-][\d.]+) \(95% CI ± (\d+(?:\.\d+)?)\)",
          document="README.md"),
    Guard("algorithm-mixture-largest-corr-shift",
          "benchmarks/bench_pafgrs_mixture.csv",
          mixture_largest_abs_corr_shift,
          r"largest shift is only ([−-][\d.]+) \(95% CI ± (\d+(?:\.\d+)?)\)",
          document="docs/algorithm.md"),
    Guard("landing-mixture-largest-corr-shift",
          "benchmarks/bench_pafgrs_mixture.csv",
          mixture_largest_abs_corr_shift,
          r"largest shift is only ([−-][\d.]+) \(95% CI ± (\d+(?:\.\d+)?)\)",
          document="index.html"),
    Guard("landing-public-fold-times",
          "benchmarks/bench_ltfhplus_compare.csv",
          lambda p: (
              _mean_se([float(r["fold_ltfhplus_over_gibbs"])
                        for r in _rows(p)])[0],
              _mean_se([float(r["fold_ltfgrs_over_pa"])
                        for r in _rows(p)])[0],
          ),
          r"same-algorithm fold times of ([\d.]+)× and ([\d.]+)×",
          document="index.html"),
    Guard("report-mixture-largest-corr-shift",
          "benchmarks/bench_pafgrs_mixture.csv",
          mixture_largest_abs_corr_shift,
          r"largest being \$([-][\d.]+)\$ with a 95\\% CI half-width of \$([\d.]+)\$",
          document="report/ltpred_methods.tex"),
    Guard("couple-env-omission-table",
          "benchmarks/bench_couple_env.csv",
          couple_env_omission_cells,
          r"\| 0\.1 \| \+([\d.]+) \(([\d.]+)\) \| ([+-][\d.]+) \(([\d.]+)\) \|"
          r"\n\| 0\.2 \| \+([\d.]+) \(([\d.]+)\) \| ([+-][\d.]+) \(([\d.]+)\) \|"
          r"\n\| 0\.3 \| \+([\d.]+) \(([\d.]+)\) \| ([+-][\d.]+) \(([\d.]+)\) \|"),
    Guard("algorithm-couple-env-omission-prose",
          "benchmarks/bench_couple_env.csv",
          couple_env_omission_bias_prose,
          r"by \+([\d.]+), \+([\d.]+), \+([\d.]+) as `s²` runs 0\.1 → 0\.2 → 0\.3"
          r" when it is\s*`C`, but only ([−-][\d.]+), \+([\d.]+), \+([\d.]+)"
          r" when it is `M`",
          document="docs/algorithm.md"),
    Guard("ascertainment-dose-unenriched", _ASCERT,
          ascert_dose_row(0.9, 1.1),
          r"^\| fitted h² \| ([\d.]+) \| [\d.]+ \| [\d.]+ \| 1\.000 \| 1\.000 \|$"),
    Guard("ascertainment-dose-mild", _ASCERT,
          ascert_dose_row(1.15, 1.35),
          r"^\| fitted h² \| [\d.]+ \| ([\d.]+) \| [\d.]+ \| 1\.000 \| 1\.000 \|$"),
    Guard("ascertainment-dose-strong", _ASCERT,
          ascert_dose_row(1.4, 1.6),
          r"^\| fitted h² \| [\d.]+ \| [\d.]+ \| ([\d.]+) \| 1\.000 \| 1\.000 \|$"),
    Guard("ascertainment-specificity-zero-false-positives", _ASCERT,
          ascert_specificity_total,
          r"K ∈ \{0\.02 … 0\.20\}, \*\*(0) false positives\*\*"),
    Guard("ascertainment-lee-case-control", _ASCERT, ascert_lee("case_control"),
          r"cohorts lands at \*\*([\d.]+)\*\* \(50/50\)"),
    Guard("ascertainment-lee-enriched", _ASCERT, ascert_lee("enriched_20"),
          r"\*\*([\d.]+)\*\* \(20% enriched\)"),
    Guard("ltfhplus-gibbs-agreement",
          "benchmarks/bench_ltfhplus_compare.csv",
          ltfhplus_gibbs_corr,
          r"Against\s+LTFHPlus 2\.2\.0, corr = ([\d.]+)"),
    Guard("ltfhplus-gibbs-rmse",
          "benchmarks/bench_ltfhplus_compare.csv",
          ltfhplus_gibbs_rmse,
          r"Against\s+LTFHPlus 2\.2\.0, corr = [\d.]+ and RMSE = "
          r"([\d.]+) ± ([\d.]+)"),
    Guard("ltfgrs-pa-agreement",
          "benchmarks/bench_ltfhplus_compare.csv",
          ltfgrs_pa_lock,
          r"ltpred PA and LTFGRS 1\.0\.1 `method=\"PA\"` agree at corr = "
          r"([\d.]+)\s*\(RMSE ([\d.]+) ± ([\d.]+)\)"),
    Guard("ltfhplus-wallclock",
          "benchmarks/bench_ltfhplus_compare.csv",
          ltfhplus_seconds,
          r"1-worker LTFHPlus took\s+([\d.]+) ± ([\d.]+) s "
          r"\(([\d.]+) ± ([\d.]+) ms/family\) versus "
          r"([\d.]+) ± ([\d.]+) s\s*\(([\d.]+) ± ([\d.]+) ms/family\) "
          r"for\s*4-thread ltpred Gibbs, "
          r"([\d.]+) ± ([\d.]+) s\s*\(([\d.]+) ± ([\d.]+) ms/family\) "
          r"for LTFGRS PA, and "
          r"([\d.]+) ± ([\d.]+) s\s*\(([\d.]+) ± ([\d.]+) ms/family\) "
          r"for ltpred PA"),
    Guard("ltfhplus-peak-rss",
          "benchmarks/bench_ltfhplus_compare.csv",
          ltfhplus_peak_mib,
          r"Isolated-process peak\s+RSS was ([\d.]+) ± ([\d.]+), "
          r"([\d.]+) ± ([\d.]+), ([\d.]+) ± ([\d.]+) and "
          r"([\d.]+) ± ([\d.]+) MiB"),
    Guard("report-ltfgrs-pa-rmse",
          "benchmarks/bench_ltfhplus_compare.csv",
          lambda p: (_mean_se([float(r["rmse_pa_ltfgrs"])
                               for r in _rows(p)])[0] * 1e5,),
          r"\\mathrm\{RMSE\}=([\d.]+)\\times 10\^\{-5\}",
          document="report/ltpred_methods.tex"),
    Guard("report-ltfhplus-seconds",
          "benchmarks/bench_ltfhplus_compare.csv",
          lambda p: _mean_se([float(r["seconds_ltfhplus"])
                              for r in _rows(p)]),
          r"LTFHPlus Gibbs\s*& --- & ---\s*& \$([\d.]+)\\pm ([\d.]+)\$",
          document="report/ltpred_methods.tex"),
    Guard("ltfhplus-fold-times",
          "benchmarks/bench_ltfhplus_compare.csv",
          ltfhplus_folds,
          r"\*\*([\d.]+) ± ([\d.]+)×\*\* \(LTFHPlus Gibbs / ltpred Gibbs\) and\s*"
          r"\*\*([\d.]+) ± ([\d.]+)×\*\* \(LTFGRS PA / ltpred PA\)\. "
          r"Mixing algorithms, LTFHPlus /\s*ltpred PA is \*\*([\d.]+) ± ([\d.]+)×\*\*"),
    Guard("report-ltfhplus-fold-times",
          "benchmarks/bench_ltfhplus_compare.csv",
          ltfhplus_folds,
          r"LTFHPlus Gibbs is \$([\d.]+)\\pm ([\d.]+)\\times\$ the ltpred\s*"
          r"Gibbs wall-clock; LTFGRS PA is \$([\d.]+)\\pm ([\d.]+)\\times\$ "
          r"the ltpred PA\s*wall-clock[\s\S]*?"
          r"LTFHPlus / ltpred PA is \$([\d.]+)\\pm ([\d.]+)\\times\$",
          document="report/ltpred_methods.tex"),
    Guard("age-onset-h05-k03-encodings",
          "benchmarks/bench_age_onset.csv",
          age_onset_h05_k03,
          r"^\| 0\.5 \| 0\.30 \| ([\d.]+) ± ([\d.]+) \| ([\d.]+) ± ([\d.]+) \| "
          r"([\d.]+) ± ([\d.]+) \| ([\d.]+) ± ([\d.]+)× \| "
          r"([\d.]+) ± ([\d.]+)× \|$"),
    Guard("mixture-dependent-mid-pin-slope",
          "benchmarks/bench_pafgrs_mixture.csv",
          mixture_dependent_mid_pin_slope,
          r"When onset only \*tends\* to track liability \(ρ = 0\.6\), pinning\s*"
          r"over-conditions \(slope ([\d.]+) under heavy censoring\)"),
    Guard("tetrachoric-falconer-h2",
          "benchmarks/bench_tetrachoric.csv",
          tetrachoric_falconer,
          r"gives ([\d.]+) ± ([\d.]+)\s*\(truth 0\.5\)"),
    Guard("liability-scale-lee-bridge",
          "benchmarks/bench_liability_scale.csv",
          liability_lee_bridge,
          r"Lee-2011 bridged to liability \| ([\d.]+) ± ([\d.]+)"),
    Guard("confounding-r4-stratified-lambda",
          "benchmarks/bench_confounding.csv",
          confounding_r4_row,
          r"^\| 4 \| ([\d.]+) ± ([\d.]+) \| ([\d.]+) ± ([\d.]+) \| "
          r"([\d.]+) ± ([\d.]+) \|$"),
    Guard("pgs-test-r2-three-scores",
          "benchmarks/bench_pgs_comparison.csv",
          pgs_test_r2s,
          r"\*\*([\d.]+) ± ([\d.]+)\*\* \(PGS\), \*\*([\d.]+) ± ([\d.]+)\*\* "
          r"\(classic LT-FH\) and\s*\*\*([\d.]+) ± ([\d.]+)\*\*"),
    Guard("report-ipw-case-control", _ASCERT,
          ascert_ipw_cc_mean,
          r"\$50/50\$ case/control\s+&\s+\$0\.499\$\s+&\s+\$1\.000\$\s+&\s+"
          r"\$([\d.]+)\$",
          document="report/ltpred_methods.tex"),
    Guard("report-ipw-enriched", _ASCERT,
          ascert_ipw_enr_mean,
          r"\$20\\%\$ case-enriched\s+&\s+\$0\.199\$\s+&\s+\$1\.000\$\s+&\s+"
          r"\$([\d.]+)\$",
          document="report/ltpred_methods.tex"),
    Guard("ascertainment-ipw-case-control", _ASCERT,
          ascert_ipw_case_control,
          r"^\| case_control \| ([\d.]+) \| \*\*([\d.]+)\*\* \| "
          r"\*\*([\d.]+)\*\* \| −[\d.]+ \| ([\d.]+) \|$"),
    Guard("ascertainment-ipw-enriched", _ASCERT,
          ascert_ipw_enriched,
          r"^\| enriched_20 \| ([\d.]+) \| \*\*([\d.]+)\*\* \| "
          r"\*\*([\d.]+)\*\* \| −[\d.]+ \| ([\d.]+) \|$"),
    Guard("ascertainment-population-nuclear", _ASCERT,
          ascert_h2_nuclear_population,
          r"^\| population \| 0\.050 \| ([\d.]+) \(SD ([\d.]+)\)"),
    Guard("ascertainment-negative-control-nuclear", _ASCERT,
          ascert_h2_nuclear_random50,
          r"^\| random_50 \*\(negative control\)\* \| 0\.049 \| ([\d.]+) \(SD ([\d.]+)\)"),
    Guard("ascertainment-ac-residual-exhausted", _ASCERT,
          ascert_ac_saturated_residual,
          r"^\| proband_case \| 0\.500 \| 0\.500 \| \*\*([\d.]+)\*\* \| 1\.00 \|$"),
    Guard("ascertainment-proband-case-moment", _ASCERT,
          ascert_proband_case_moment,
          r"^\| proband_case \| ([\d.]+) \| \*\*([\d.]+)\*\* \| ([\d.]+) \|$"),
    Guard("ascertainment-bias-constant-in-n", _ASCERT,
          ascert_scale_invariance,
          r"`proband_case` holds at \*\*\+([\d.]+) / \+([\d.]+) / \+([\d.]+)\*\*"),
    Guard("ascertainment-null-population", _ASCERT_NULL,
          ascert_null_population,
          r"population returns ([\d.]+) and `random_50`"),
    Guard("ascertainment-null-negative-control", _ASCERT_NULL,
          ascert_null_random50,
          r"and `random_50` ([\d.]+), while"),
    Guard("accuracy-pa-gibbs-agreement-range", _ACCURACY,
          accuracy_agreement_range,
          r"mean (?:corr\(PA, Gibbs\)|PA–Gibbs agreement) is "
          r"(\d+(?:\.\d+)?)–(\d+(?:\.\d+)?)"),
    Guard("accuracy-pa-gibbs-se-ceiling", _ACCURACY,
          accuracy_agreement_se_ceiling,
          r"across-seed SEs of at most ([\d.]+)"),
    Guard("accuracy-row-parents-h2-05-k-005", _ACCURACY,
          accuracy_cell_parents,
          r"^\| parents \| " + _ACCURACY_ROW),
    Guard("accuracy-row-parents-2-siblings-h2-05-k-005", _ACCURACY,
          accuracy_cell_parents_sibs,
          r"^\| parents \+ 2 siblings \| " + _ACCURACY_ROW),
    Guard("accuracy-row-extended-h2-05-k-005", _ACCURACY,
          accuracy_cell_extended,
          r"^\| extended \| " + _ACCURACY_ROW),
    Guard("accuracy-effn-grid-max", _ACCURACY,
          accuracy_effn_grid_max,
          r"the largest grid mean is ([\d.]+) ± ([\d.]+)×"),
    Guard("accuracy-effn-former-max-replication", _ACCURACY,
          accuracy_effn_former_max,
          r"replicates as ([\d.]+) ± ([\d.]+)×"),
    Guard("scaling-object-speedup-range-4-threads",
          "benchmarks/bench_scaling.csv",
          scaling_object_speedup_range,
          r"\*\*(\d+)–(\d+)×"),
    Guard("scaling-array-throughput-range",
          "benchmarks/bench_scaling.csv",
          scaling_array_throughput_range,
          r"its ([\d.]+)–([\d.]+)\s+million families/s"),
    Guard("scaling-nfam-500-row",
          "benchmarks/bench_scaling.csv",
          scaling_nfam_500_row,
          r"^\| 500 \| (\d+) \| ([\d,]+) \| ([\d.]+) M \| (\d+)×"),
    Guard("report-scaling-object-speedup",
          "benchmarks/bench_scaling.csv",
          scaling_object_speedup_range,
          r"quotes a \$(\d+)\$--\$(\d+)\\times\$ object-path",
          document="report/ltpred_methods.tex"),
    Guard("report-scaling-array-throughput",
          "benchmarks/bench_scaling.csv",
          scaling_array_throughput_range,
          r"ran at \$([\d.]+)\$--\$([\d.]+)\$\s+million families per second",
          document="report/ltpred_methods.tex"),
    Guard("report-scaling-array-vs-object",
          "benchmarks/bench_scaling.csv",
          scaling_array_vs_object_range,
          r"another\s+\$(\d+)\$--\$(\d+)\\times\$ over the object path",
          document="report/ltpred_methods.tex"),
    Guard("estimation-array-speedup",
          "benchmarks/bench_scaling.csv",
          scaling_array_vs_object_range,
          r"(\d+)–(\d+)× faster than the object path",
          document="docs/estimation.md"),
    Guard("estimation-array-throughput",
          "benchmarks/bench_scaling.csv",
          scaling_array_throughput_range,
          r"at\s+([\d.]+)–([\d.]+) million already-aligned families/s",
          document="docs/estimation.md"),
    Guard("landing-object-pa-rate",
          "benchmarks/bench_scaling.csv",
          scaling_object_pa_rate_k_range,
          r"ran ~([\d.]+)k–([\d.]+)k families/second on the object path",
          document="index.html"),
    Guard("landing-array-throughput",
          "benchmarks/bench_scaling.csv",
          scaling_array_throughput_range,
          r"array path reached ([\d.]+)–([\d.]+) M families/s",
          document="index.html"),
    Guard("gwas-power-ncp-ratio-headline",
          "benchmarks/bench_gwas_power.csv",
          gwas_power_ncp_gibbs,
          r"NCP ratio of\s*\*\*([\d.]+) ± ([\d.]+)×\*\*"),
    Guard("gwas-power-ncp-ratio-pa-row",
          "benchmarks/bench_gwas_power.csv",
          gwas_power_ncp_pa,
          r"^\| classic LT-FH \(PA\) \| [\d.]+ ± [\d.]+ \| ([\d.]+) ± ([\d.]+)×"),
    Guard("fit-heritability-h2-02-row",
          "benchmarks/bench_fit_heritability.csv",
          fit_heritability_h02_row,
          r"^\| 0\.2 \| ([\d.]+) \| ([-+][\d.]+) \| ([\d.]+) \| ([\d.]+) \| (\d+)×"),
    Guard("fit-heritability-structure-sd-trio",
          "benchmarks/bench_fit_heritability.csv",
          fit_heritability_structure_sds,
          r"empirical SD is ([\d.]+) with parents only, ([\d.]+) with parents "
          r"\+\s*two siblings, and ([\d.]+) with the extended structure"),
    Guard("variance-components-recovery-a04-c02-row",
          "benchmarks/bench_variance_components.csv",
          variance_components_recovery_row,
          r"^\| 0\.4 \| 0\.2 \| ([\d.]+) \(([\d.]+)\) \| ([\d.]+) \(([\d.]+)\)"),
    Guard("variance-components-scaling-sd-span",
          "benchmarks/bench_variance_components.csv",
          variance_components_scaling_sds,
          r"empirical SD from ([\d.]+) to ([\d.]+) for A and from ([\d.]+) to "
          r"([\d.]+) for C"),
    Guard("genetic-correlation-null-row",
          "benchmarks/bench_genetic_correlation.csv",
          genetic_correlation_null_row,
          r"^\| 0\.0 \| (-?[\d.]+) \| (-?[\d.]+) \| ([\d.]+) \|$"),
    Guard("calibration-psibs-k-001-row",
          "benchmarks/bench_calibration.csv",
          calibration_psibs_k001_row,
          r"^\| parents \+ siblings \| 0\.01 \| ([\d.]+) ± ([\d.]+) \| "
          r"([\d.]+) ± ([\d.]+) \| ([\d.]+) ± ([\d.]+) \|$"),
    Guard("calibration-extended-k-001-top-ratio",
          "benchmarks/bench_calibration.csv",
          calibration_extended_k001_top_ratio,
          r"is ([\d.]+) ± ([\d.]+), about 3 SE below 1"),
    Guard("calibration-misspec-slope-sweep",
          "benchmarks/bench_calibration.csv",
          calibration_misspec_slope_sweep,
          r"the slope sweeps from ([\d.]+) ± ([\d.]+) down to ([\d.]+) ± ([\d.]+)"),
    Guard("ltfhpp-adult-ncp-ratio",
          "benchmarks/bench_ltfhpp_personalization.csv",
          ltfhpp_adult_ncp,
          r"^\| ADuLT \(same full personalised proband CIP, no FH\) \| "
          r"[\d.]+ ± [\d.]+ \| [\d.]+ ± [\d.]+ \| ([\d.]+) ± ([\d.]+)×"),
    Guard("ltfhpp-full-ncp-ratio",
          "benchmarks/bench_ltfhpp_personalization.csv",
          ltfhpp_full_ncp,
          r"^\| \*\*LT-FH\+\+ \(full age \+ sex \+ cohort CIP\)\*\* \| "
          r"\*\*[\d.]+ ± [\d.]+\*\* \| \*\*[\d.]+ ± [\d.]+\*\* \| "
          r"\*\*([\d.]+) ± ([\d.]+)×\*\*"),
    Guard("ltfhpp-ncp-increment-vs-adult",
          "benchmarks/bench_ltfhpp_personalization.csv",
          ltfhpp_ncp_increment,
          r"causal-SNP NCP ratio by\s*\*\*\+([\d.]+) ± ([\d.]+)\*\*"),
    Guard("ltfhpp-corr-increment-vs-adult",
          "benchmarks/bench_ltfhpp_personalization.csv",
          ltfhpp_corr_increment,
          r"improves adjusted\s*correlation by \*\*\+([\d.]+) ± ([\d.]+)\*\*"),
    Guard("sex-limitation-replicate-count",
          "benchmarks/bench_sex_limitation.csv",
          sex_limitation_replicate_count,
          r"(\d+) replicates of 2,000 families"),
    Guard("pedigree-payoff-correlations",
          "benchmarks/bench_pedigree_inference.csv",
          pedigree_payoff,
          r"corr\(est, truth\) is \*\*([\d.]+) ± ([\d.]+)\*\* using all\s*"
          r"relatives up to third degree vs \*\*([\d.]+) ± ([\d.]+)\*\*"),
    Guard("pedigree-payoff-paired-contrast",
          "benchmarks/bench_pedigree_inference.csv",
          pedigree_payoff_contrast,
          r"paired all-minus-named-role contrast is \*\*\+([\d.]+) ± ([\d.]+)"
          r"\*\*, 95% CI\s*\[\+([\d.]+), \+([\d.]+)\]"),
    Guard("register-degree-accuracy",
          "benchmarks/bench_register_pipeline.csv",
          register_degree_accuracy,
          r"corr\(est, true g\) is\s*\*\*([\d.]+) ± ([\d.]+)\*\* at degree 3 "
          r"\(first cousins\) vs\s*\*\*([\d.]+) ± ([\d.]+)\*\* at\s*degree 1"),
    Guard("register-relatives-events-delta-auc",
          "benchmarks/bench_register_pipeline.csv",
          register_relatives_events_delta_auc,
          r"\*\*ΔAUC \+([\d.]+) ± ([\d.]+), 95% CI\s*"
          r"\[\+([\d.]+), \+([\d.]+)\]\*\*"),
    Guard("register-own-outcome-delta-corr",
          "benchmarks/bench_register_pipeline.csv",
          register_own_outcome_delta_corr,
          r"paired\s*\(c\)-\(a\) Δcorr \*\*\+([\d.]+) ± ([\d.]+)\*\*, 95% CI\s*"
          r"\[\+([\d.]+), \+([\d.]+)\]"),
]


def main() -> int:
    results = run_guards(ROOT)
    failures = []
    for guard, ok, detail in results:
        print(f"{guard.name:<46} {'ok' if ok else 'MISMATCH'}"
              + ("" if ok else f" -- {detail}"), flush=True)
        if not ok:
            failures.append((guard.name, detail))

    print(f"\n{len(results)} guards, {len(results) - len(failures)} ok, "
          f"{len(failures)} mismatches")
    if failures:
        print("\nRESULTS.md integrity checks failed:")
        for name, detail in failures:
            print(f"  - {name}: {detail}")
        return 1
    print("\nRESULTS.md headline values all re-derive from the committed artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
