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
    results_text = (root / "benchmarks" / "RESULTS.md").read_text(encoding="utf-8")
    return [(guard, *run_guard(root, results_text, guard)) for guard in GUARDS]


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


GUARDS = [
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
