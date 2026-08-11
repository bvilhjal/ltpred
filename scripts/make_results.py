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
    "fit_heritability": ("bench_fit_heritability.csv", fit_heritability),
    "variance_components": ("bench_variance_components.csv", variance_components),
    "genetic_correlation": ("bench_genetic_correlation.csv", genetic_correlation),
    "calibration_summary": ("bench_calibration.csv", calibration_summary),
    "integrated_panel": ("bench_ltfhpp_personalization.csv", integrated_panel),
}

_GENERATED = (f"Generated by scripts/make_results.py from benchmarks/{{source}} "
              f"-- do not hand-edit. Regenerate: {COMMAND}")


def _tex_cell(text: str) -> str:
    return (text.replace("±", r"$\pm$").replace("×", r"$\times$")
            .replace("->", r"$\to$").replace("h²", r"$h^2$")
            .replace("r_g", r"$r_g$").replace("λ", r"$\lambda$"))


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
