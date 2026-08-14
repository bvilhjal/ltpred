#!/usr/bin/env python3
"""Render paper-ready tables from the committed benchmark CSVs.

The 2026-08 doc/artifact skew review showed that hand-transcribed tables
drift from the artifacts they summarise. Paper tables are therefore
*generated, not transcribed*: every table in ``paper/tables/`` is rendered
here from the committed long-format CSVs in ``benchmarks/`` (stdlib ``csv``
only, so this runs in the dependency-minimal ``ltpred314`` verification
environment), and ``scripts/check_results.py`` guards the RESULTS.md
headline values against the same artifacts.

One function per table: read CSV rows, filter, aggregate (across-replicate
mean +/- SE), format. Markdown and booktabs-style LaTeX share the same cell
strings; the LaTeX writer only translates the math glyphs.

Run it from the repository root::

    python scripts/make_results.py

Outputs (do not hand-edit): ``paper/tables/*.md`` and ``paper/tables/*.tex``.
"""
from __future__ import annotations

import csv
import math
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
TABLES_DIR = ROOT / "paper" / "tables"
COMMAND = "python scripts/make_results.py"

STRUCTURE_DISPLAY = {
    "parents": "parents",
    "parents+sibs": "parents + 2 siblings",
    "parents+2 sibs": "parents + 2 siblings",
    "extended": "extended",
}

PHENOTYPE_DISPLAY = {
    "case/control": "case/control",
    "ADuLT (no family history)": "ADuLT (full personalised proband CIP, no FH)",
    "LT-FH (single K)": "LT-FH single-K",
    "FH + age CIP": "FH + age CIP (ablation)",
    "FH + age + sex CIP": "FH + age + sex CIP (ablation)",
    "FH + age + cohort CIP": "FH + age + cohort CIP (ablation)",
    "LT-FH++ (full CIP)": "LT-FH++ (full age + sex + cohort CIP)",
}


# --- helpers ------------------------------------------------------------------


def _rows(path: pathlib.Path) -> list:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _mean_se(values: list):
    n = len(values)
    mean = sum(values) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    return mean, sd / math.sqrt(n)


def _pm(mean: float, se: float, ndp: int) -> str:
    return f"{mean:.{ndp}f} ± {se:.{ndp}f}"


def _replicate_means(rows: list, phenotype: str, column: str):
    vals = [float(r[column]) for r in rows if r["phenotype"] == phenotype]
    return _mean_se(vals)


# --- one function per table ----------------------------------------------------
# Each returns (title, source_csv, headers, aligns, body); cells are display
# strings in markdown style ("±", "×"), translated for LaTeX at write time.


def accuracy_grid(rows):
    order = {"parents": 0, "parents+sibs": 1, "extended": 2}
    body = []
    for r in sorted(rows, key=lambda r: (order[r["structure"]], float(r["h2"]),
                                         float(r["prevalence"]))):
        body.append([
            STRUCTURE_DISPLAY[r["structure"]], r["h2"], r["prevalence"],
            _pm(float(r["corr_gibbs"]), float(r["se_corr_gibbs"]), 3),
            _pm(float(r["corr_pa"]), float(r["se_corr_pa"]), 3),
            _pm(float(r["eff_n_proxy_pa"]), float(r["se_eff_n_proxy_pa"]), 2) + "×",
            f'{float(r["gibbs_pa_corr"]):.4f}',
        ])
    return ("PA versus Gibbs accuracy (27-cell grid, five seeds per cell)",
            "bench_accuracy.csv",
            ["family structure", "h²", "K", "corr Gibbs", "corr PA",
             "PA eff-N proxy / case-control", "corr(PA, Gibbs)"],
            ["l", "r", "r", "r", "r", "r", "r"], body)


def fit_heritability(rows):
    body = []
    for r in sorted((r for r in rows if r["panel"] == "bias_precision"),
                    key=lambda r: float(r["h2"])):
        h2, bias = float(r["h2"]), float(r["bias"])
        sd, se = float(r["sd"]), float(r["reported_se"])
        body.append([f"{h2:.1f}", f"{h2 + bias:.3f}", f"{bias:+.3f}",
                     f"{sd:.3f}", f"{se:.4f}", f"{sd / se:.0f}×"])
    return ("Heritability fitting (25 population cohorts per cell)",
            "bench_fit_heritability.csv",
            ["true h²", "fitted mean", "bias", "empirical SD", "reported MC SE",
             "SD / MC SE"],
            ["r", "r", "r", "r", "r", "r"], body)


def variance_components(rows):
    body = []
    for r in sorted((r for r in rows if r["panel"] == "recovery"),
                    key=lambda r: (float(r["a2"]), float(r["c2"]))):
        a2, c2 = float(r["a2"]), float(r["c2"])
        body.append([
            f"{a2:.1f}", f"{c2:.1f}",
            f"{a2 + float(r['A_bias']):.3f} ({float(r['A_sd']):.3f})",
            f"{c2 + float(r['C_bias']):.3f} ({float(r['C_sd']):.3f})",
        ])
    return ("A+C variance-component recovery (25 population cohorts per cell)",
            "bench_variance_components.csv",
            ["true A", "true C", "fitted A (SD)", "fitted C (SD)"],
            ["r", "r", "r", "r"], body)


def genetic_correlation(rows):
    body = []
    for r in sorted((r for r in rows if r["panel"] == "bias_precision"),
                    key=lambda r: float(r["rg"])):
        rg, bias = float(r["rg"]), float(r["bias"])
        body.append([f"{rg:.1f}", f"{rg + bias:.4f}", f"{bias:+.4f}",
                     f"{float(r['sd']):.4f}"])
    return ("Genetic-correlation recovery (25 cohorts per cell)",
            "bench_genetic_correlation.csv",
            ["true r_g", "fitted mean", "bias", "empirical SD"],
            ["r", "r", "r", "r"], body)


def calibration_summary(rows):
    order = {"parents+sibs": 0, "extended": 1}
    body = []
    for r in sorted((r for r in rows if r["panel"] == "correct"),
                    key=lambda r: (order[r["structure"]],
                                   float(r["prevalence"]))):
        body.append([
            STRUCTURE_DISPLAY[r["structure"]], r["prevalence"],
            _pm(float(r["slope_pa"]), float(r["se_slope_pa"]), 3),
            _pm(float(r["corr_pa"]), float(r["se_corr_pa"]), 3),
            _pm(float(r["top_ratio_pa"]), float(r["se_top_ratio_pa"]), 3),
        ])
    return ("Score calibration, correctly specified PA (five seeds per cell)",
            "bench_calibration.csv",
            ["family structure", "K", "slope", "corr",
             "top-decile realised / predicted"],
            ["l", "r", "r", "r", "r"], body)


def gwas_power(rows):
    display = {
        "case_control": "case/control",
        "LT-FH (Gibbs)": "classic LT-FH (Gibbs)",
        "LT-FH (PA)": "classic LT-FH (PA)",
        "oracle (true g)": "oracle true g",
    }
    order = list(display)
    body = []
    for r in sorted(rows, key=lambda r: order.index(r["phenotype"])):
        ncp = float(r["effN_vs_cc"])
        se_ncp = float(r["se_effN_vs_cc"])
        ncp_s = "1.00×" if r["phenotype"] == "case_control" else (
            _pm(ncp, se_ncp, 3) + "×")
        body.append([
            display[r["phenotype"]],
            _pm(float(r["mean_chi2_causal"]), float(r["se_mean_chi2_causal"]), 2),
            ncp_s,
            _pm(100 * float(r["power_gw"]), 100 * float(r["se_power_gw"]), 1) + "%",
            _pm(float(r["lambda_gc"]), float(r["se_lambda_gc"]), 3),
        ])
    return ("Replicated classic-LT-FH genotype GWAS (three replicates)",
            "bench_gwas_power.csv",
            ["phenotype", "mean causal χ²", "causal-SNP NCP ratio / c-c",
             "power at 5e-8", "λ_GC"],
            ["l", "r", "r", "r", "r"], body)


def confounding(rows):
    body = []
    for r in sorted(rows, key=lambda r: float(r["trend_R"])):
        body.append([
            f'{float(r["trend_R"]):.0f}',
            _pm(float(r["lgc_strat_cohort"]), float(r["se_lgc_strat_cohort"]), 3),
            _pm(float(r["lgc_strat_single_K"]),
                float(r["se_lgc_strat_single_K"]), 3),
            _pm(float(r["lgc_strat_casecontrol"]),
                float(r["se_lgc_strat_casecontrol"]), 3),
        ])
    return ("Cohort-component confounding (three replicates per trend)",
            "bench_confounding.csv",
            ["prevalence trend R per 30 y", "FH + cohort-specific K",
             "single-K FH", "case/control"],
            ["r", "r", "r", "r"], body)


def pgs_comparison(rows):
    reps = [r for r in rows if r["row_type"] == "replicate"]

    def ms(arm, column):
        vals = [float(r[column]) for r in reps
                if r["arm"] == arm and r[column] not in ("", "nan")]
        return _mean_se(vals)

    body = [
        ["case/control label",
         _pm(*ms("case/control", "corr_g"), 3),
         _pm(*ms("case/control", "r2_g"), 3)],
        ["PGS",
         _pm(*ms("PGS", "corr_g"), 3),
         _pm(*ms("PGS", "r2_g"), 3)],
        ["LT-FH (PA)",
         _pm(*ms("LT-FH (PA)", "corr_g"), 3),
         _pm(*ms("LT-FH (PA)", "r2_g"), 3)],
        ["PGS + LT-FH joint (OLS)",
         "—",
         _pm(*ms("PGS + LT-FH joint", "r2_g"), 3)],
    ]
    return ("PGS and classic LT-FH on a held-out test half (five replicates)",
            "bench_pgs_comparison.csv",
            ["score", "corr with held-out g", "R²"],
            ["l", "r", "r"], body)


def calibration_misspec(rows):
    body = []
    for r in sorted((r for r in rows if r["panel"] == "misspec"),
                    key=lambda r: float(r["assumed_h2"])):
        body.append([
            r["assumed_h2"],
            _pm(float(r["slope_pa"]), float(r["se_slope_pa"]), 3),
            _pm(float(r["corr_pa"]), float(r["se_corr_pa"]), 3),
            _pm(float(r["top_ratio_pa"]), float(r["se_top_ratio_pa"]), 3),
            _pm(float(r["cal_rmse_pa"]), float(r["se_cal_rmse_pa"]), 3),
        ])
    return ("Score calibration under a wrong assumed h² (true h² = 0.5, K = 0.05)",
            "bench_calibration.csv",
            ["assumed h²", "slope", "corr", "top realised/predicted",
             "calibration RMSE"],
            ["r", "r", "r", "r", "r"], body)


def integrated_panel(rows):
    reps = [r for r in rows
            if r["row_type"] == "replicate" and r["panel"] == "main_ablation"]
    body = []
    for phenotype in PHENOTYPE_DISPLAY:
        corr = _replicate_means(reps, phenotype, "corr_g_adjusted")
        slope = _replicate_means(reps, phenotype, "calibration_slope_adjusted")
        ncp = _replicate_means(reps, phenotype, "effN_vs_cc_adjusted")
        lam_raw = _replicate_means(reps, phenotype, "lambda_strat_raw")
        lam_adj = _replicate_means(reps, phenotype, "lambda_strat_adjusted")
        body.append([
            PHENOTYPE_DISPLAY[phenotype],
            _pm(*corr, 3), _pm(*slope, 3), _pm(*ncp, 3) + "×",
            f"{_pm(*lam_raw, 3)} -> {_pm(*lam_adj, 3)}",
        ])
    return ("Integrated personalised LT-FH++ genotype GWAS "
            "(ten paired replicates)",
            "bench_ltfhpp_personalization.csv",
            ["phenotype", "adjusted corr", "adjusted slope",
             "adjusted causal-SNP NCP ratio / c-c",
             "stratified-null λ raw -> adjusted"],
            ["l", "r", "r", "r", "r"], body)


# --- writers --------------------------------------------------------------------


TABLES = {
    "accuracy_grid": ("bench_accuracy.csv", accuracy_grid),
    "gwas_power": ("bench_gwas_power.csv", gwas_power),
    "fit_heritability": ("bench_fit_heritability.csv", fit_heritability),
    "variance_components": ("bench_variance_components.csv", variance_components),
    "genetic_correlation": ("bench_genetic_correlation.csv", genetic_correlation),
    "calibration_summary": ("bench_calibration.csv", calibration_summary),
    "calibration_misspec": ("bench_calibration.csv", calibration_misspec),
    "integrated_panel": ("bench_ltfhpp_personalization.csv", integrated_panel),
    "confounding": ("bench_confounding.csv", confounding),
    "pgs_comparison": ("bench_pgs_comparison.csv", pgs_comparison),
}

_GENERATED = (f"Generated by scripts/make_results.py from benchmarks/{{source}} "
              f"-- do not hand-edit. Regenerate: {COMMAND}")


def _tex_cell(text: str) -> str:
    """Render one cell as LaTeX.

    Order matters. The math substitutions below *introduce* underscores
    (``$r_g$``, ``$\lambda_{\mathrm{GC}}$``) which must not then be escaped, while
    any underscore or percent sign coming from the CSV must be, or TeX reads
    them as a subscript and a comment respectively -- which silently truncated
    every row of the GWAS-power table. So: stash the math tokens, escape the
    literals, restore the math.
    """
    RG, LGC = "\x00rg\x00", "\x00lgc\x00"
    text = text.replace("λ_GC", LGC).replace("r_g", RG)
    text = text.replace("%", r"\%").replace("_", r"\_")
    text = (text.replace("±", r"$\pm$").replace("×", r"$\times$")
            .replace("->", r"$\to$").replace("h²", r"$h^2$")
            .replace("λ", r"$\lambda$")
            .replace("χ²", r"$\chi^2$").replace("—", "---"))
    return text.replace(RG, r"$r_g$").replace(LGC, r"$\lambda_{\mathrm{GC}}$")


def write_markdown(path, title, source, headers, aligns, body):
    lines = [f"<!-- {_GENERATED.format(source=source)} -->", "",
             f"## {title}", "",
             "| " + " | ".join(headers) + " |",
             "|" + "|".join("---:" if a == "r" else "---" for a in aligns) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in body]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex(path, title, source, headers, aligns, body):
    lines = [f"% {_GENERATED.format(source=source)}",
             f"% {title}",
             r"\begin{tabular}{" + "".join(aligns) + "}",
             r"\toprule",
             " & ".join(_tex_cell(h) for h in headers) + r" \\",
             r"\midrule"]
    lines += [" & ".join(_tex_cell(c) for c in row) + r" \\" for row in body]
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    for name, (csv_name, build) in TABLES.items():
        title, source, headers, aligns, body = build(
            _rows(ROOT / "benchmarks" / csv_name))
        write_markdown(TABLES_DIR / f"{name}.md", title, source, headers,
                       aligns, body)
        write_latex(TABLES_DIR / f"{name}.tex", title, source, headers,
                    aligns, body)
        print(f"wrote paper/tables/{name}.md and paper/tables/{name}.tex "
              f"({len(body)} rows)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
