#!/usr/bin/env python3
"""Check the few benchmark claims that define the public release snapshot."""

import csv
import hashlib
import json
import math
import re
import statistics
import subprocess
import sys
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "RESULTS.md"
REPORT_TEX = ROOT / "report" / "ltpred_methods.tex"
REPORT_PDF = REPORT_TEX.with_suffix(".pdf")


def read(path):
    return path.read_text(encoding="utf-8")


def csv_rows(name):
    with (ROOT / "benchmarks" / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def one(rows, **wanted):
    found = [row for row in rows
             if all(row[key] == str(value) for key, value in wanted.items())]
    if len(found) != 1:
        raise AssertionError(f"expected one row for {wanted}, found {len(found)}")
    return found[0]


def mean_se(rows, key):
    values = [float(row[key]) for row in rows]
    return statistics.fmean(values), statistics.stdev(values) / math.sqrt(len(values))


def require(path, *snippets):
    text = read(path)
    missing = [snippet for snippet in snippets if snippet not in text]
    if missing:
        raise AssertionError(f"{path.relative_to(ROOT)} missing {missing!r}")


def check_scaling():
    rows = csv_rows("bench_scaling.csv")
    sizes = [row for row in rows if row["axis"] == "family_size"]
    speedups = [float(row["speedup"]) for row in sizes]
    speed_range = f"{min(speedups):.0f}–{max(speedups):.0f}×"
    require(RESULTS, f"**{speed_range} faster**", f"**{speed_range}** object-path")

    counts = [row for row in rows if row["axis"] == "n_fam"]
    for row in counts:
        expected = (
            f'| {int(row["value"]):,} | {float(row["fam_per_s_gibbs"]):,.0f} | '
            f'{float(row["fam_per_s_pa"]):,.0f} | '
            f'{float(row["fam_per_s_pa_array"]) / 1e6:.2f} M | '
            f'{float(row["speedup"]):.0f}× |'
        )
        require(RESULTS, expected)

    labels = {
        "2": "parents", "3": "+ sibling", "5": "+ two grandparents",
        "7": "extended", "10": "extended + aunts",
    }
    for row in sizes:
        t_pa = float(row["t_pa"])
        pa = f"{t_pa:.5f}" if t_pa < 0.01 else f"{t_pa:.4f}"
        expected = (
            f'| {row["value"]} | {labels[row["value"]]} | '
            f'{float(row["t_gibbs"]):.2f} s | {pa} s | '
            f'{float(row["t_pa_array"]):.5f} s | {float(row["speedup"]):.0f}× |'
        )
        require(RESULTS, expected)

    for path in [ROOT / "README.md", ROOT / "docs/estimation.md",
                 ROOT / "docs/algorithm.md", ROOT / "ltpred/estimate.py"]:
        require(path, speed_range)
    plain_range = speed_range[:-1]
    require(ROOT / "paper/tables/headlines.md", plain_range.replace("–", "--") + "×")
    lo, hi = min(speedups), max(speedups)
    require(ROOT / "paper/tables/headlines.tex",
            f"{lo:.0f}--{hi:.0f}$\\times$")
    require(REPORT_TEX, f"${lo:.0f}$--${hi:.0f}\\times$")
    return speed_range, lo, hi


def check_pa_robustness():
    rows = csv_rows("bench_pa_robustness.csv")
    agrees = [float(row["agree"]) for row in rows if row.get("agree")]
    if not agrees:
        raise AssertionError("bench_pa_robustness.csv has no agreement values")
    worst = min(agrees)
    # Four-decimal *floor*, not round: 0.99916 rounds to 0.9992, which is how
    # the methods note previously contradicted its own worst-seed parenthetical.
    floor = f"{int(worst * 10_000) / 10_000:.4f}"
    seed5 = f"{worst:.5f}"
    require(RESULTS, f"at least {floor}", f"worst single-seed {seed5}")
    require(REPORT_TEX, f"$\\ge {floor}$", seed5)
    return floor, seed5


def check_ipw():
    rows = csv_rows("bench_ascertainment.csv")
    case_control = one(rows, arm="ipw", scheme="case_control")
    enriched = one(rows, arm="ipw", scheme="enriched_20")
    if case_control["n_fam"] != enriched["n_fam"]:
        raise AssertionError("IPW headline rows use different sample sizes")
    n_fam = int(case_control["n_fam"])
    cc = f'{float(case_control["fitted_mean"]):.3f}'
    en = f'{float(enriched["fitted_mean"]):.3f}'

    require(RESULTS, f"**{cc}**", f"**{en}**", f"N = {n_fam:,}")
    require(ROOT / "docs/inference.md", f"**{cc}**", f"**{en}**")
    require(ROOT / "ltpred/fit.py", f"``{cc}``", f"{n_fam:,} families")
    require(ROOT / "paper/tables/headlines.md", cc, f"N = {n_fam:,}")
    require(ROOT / "paper/tables/headlines.tex", cc, f"N = {n_fam:,}")
    tex_n = f"{n_fam:,}".replace(",", "{,}")
    require(REPORT_TEX, f"${cc}$", f"${en}$", f"${tex_n}$")
    return cc, en, n_fam


def check_r_provenance(rows, threads):
    provenance = [
        "ltfhplus_seed", "r_version", "r_rng_kind", "python_version",
        "numpy_version", "numba_version", "numba_threads",
        "omp_num_threads", "openblas_num_threads",
    ]
    if not rows or any(not row.get(key) for row in rows for key in provenance):
        raise AssertionError("R-lock rows lack seed, runtime or thread provenance")
    seeds = [row["ltfhplus_seed"] for row in rows]
    if len(set(seeds)) != len(seeds):
        raise AssertionError("R-lock LTFHPlus seeds are not unique by replicate")
    for key in ["r_version", "r_rng_kind", "python_version", "numpy_version",
                "numba_version"]:
        if len({row[key] for row in rows}) != 1:
            raise AssertionError(f"R-lock {key} changes within one comparison")
    expected_threads = (str(threads), str(threads), "1")
    for row in rows:
        observed = (row["numba_threads"], row["omp_num_threads"],
                    row["openblas_num_threads"])
        if observed != expected_threads or row["ltfhplus_workers"] != "1":
            raise AssertionError(
                f"R-lock thread provenance is {observed}, expected {expected_threads}"
            )


def check_r_lock():
    rows = csv_rows("bench_ltfhplus_compare.csv")
    rows_one = csv_rows("bench_ltfhplus_compare_1thread.csv")
    check_r_provenance(rows, 4)
    check_r_provenance(rows_one, 1)

    gibbs, gibbs_se = mean_se(rows, "fold_ltfhplus_over_gibbs")
    pa, pa_se = mean_se(rows, "fold_ltfgrs_over_pa")
    gibbs_one, gibbs_one_se = mean_se(rows_one, "fold_ltfhplus_over_gibbs")
    pa_one, pa_one_se = mean_se(rows_one, "fold_ltfgrs_over_pa")
    detailed = {
        "gibbs": f"{gibbs:.3f} ± {gibbs_se:.3f}×",
        "pa": f"{pa:.0f} ± {pa_se:.0f}×",
        "gibbs_one": f"{gibbs_one:.3f} ± {gibbs_one_se:.3f}×",
        "pa_one": f"{pa_one:.0f} ± {pa_one_se:.0f}×",
    }

    gibbs_rmse, _ = mean_se(rows, "rmse_gibbs_ltfhplus")
    pa_rmse, _ = mean_se(rows, "rmse_pa_ltfhplus")
    ltfgrs_rmse, _ = mean_se(rows, "rmse_pa_ltfgrs")
    require(
        ROOT / "ltpred/estimate.py",
        f"RMSE ``{gibbs_rmse:.4f}`` Gibbs, ``{pa_rmse:.4f}`` PA",
        f"RMSE ``{ltfgrs_rmse:.6f}``",
    )
    for path in [ROOT / "README.md", ROOT / "docs/algorithm.md"]:
        require(path, f"RMSE {gibbs_rmse:.4f} Gibbs, {pa_rmse:.4f} PA",
                f"RMSE {ltfgrs_rmse:.6f}")

    for path in [ROOT / "README.md", ROOT / "docs/estimation.md"]:
        require(path, f"{gibbs:.2f}×", f"{pa:.0f}×",
                f"{gibbs_one:.2f}×", f"{pa_one:.0f}×")
    for path in [ROOT / "docs/algorithm.md", ROOT / "ltpred/estimate.py"]:
        require(path, f"{gibbs:.2f}×", f"{pa:.0f}×")
    require(RESULTS, *detailed.values())
    require(ROOT / "paper/tables/headlines.md", f"{gibbs:.2f} ±", f"{pa:.0f} ±")
    require(REPORT_TEX, f"{gibbs:.2f}", f"{pa:.0f}",
            f"{gibbs_one:.3f}", f"{pa_one:.0f}")
    return {
        "details": detailed, "gibbs": gibbs, "pa": pa,
        "gibbs_one": gibbs_one, "pa_one": pa_one,
    }


def check_pgs():
    rows = csv_rows("bench_pgs_comparison.csv")
    replicates = [row for row in rows if row["row_type"] == "replicate"]
    joint_rows = [row for row in replicates if row["arm"] == "PGS + LT-FH joint"]
    if not joint_rows or any(row["joint_fit_design"] !=
                             "seeded_shuffled_test_kfold_ols" for row in joint_rows):
        raise AssertionError("PGS joint rows are not the expected cross-fitted design")
    if any(int(row["joint_folds"]) < 2 or not row["joint_seed"]
           for row in joint_rows):
        raise AssertionError("PGS joint rows lack folds or seeds")
    if len({row["joint_seed"] for row in joint_rows}) != len(joint_rows):
        raise AssertionError("PGS joint-fold seeds are not independent by replicate")

    joint, joint_se = mean_se(joint_rows, "r2_g")
    pgs, pgs_se = mean_se(
        [row for row in replicates if row["arm"] == "PGS"], "r2_g")
    fh, fh_se = mean_se(
        [row for row in replicates if row["arm"] == "LT-FH (PA)"], "r2_g")
    headline = f"{joint:.3f} / {pgs:.3f} / {fh:.3f}"
    joint_cell = f"{joint:.3f} ± {joint_se:.3f}"
    for path in [RESULTS, ROOT / "paper/tables/pgs_comparison.md"]:
        require(path, joint_cell, "cross-fitted")
    require(ROOT / "paper/tables/headlines.md", headline, "cross-fitted")
    require(ROOT / "paper/tables/headlines.tex", headline, "cross-fitted")
    require(ROOT / "paper/tables/pgs_comparison.tex",
            f"{joint:.3f} $\\pm$ {joint_se:.3f}", "cross-fitted")
    require(REPORT_TEX, f"R^{{2}}={joint:.3f}", "cross-fitted OLS")

    contrasts = [row for row in rows if row["row_type"] == "paired_contrast"]
    over_pgs = one(contrasts, comparison="PGS + LT-FH joint - PGS", metric="r2_g")
    over_fh = one(contrasts, comparison="PGS + LT-FH joint - LT-FH (PA)",
                  metric="r2_g")
    increment_pgs = float(over_pgs["delta_mean"])
    increment_pgs_ci = float(over_pgs["delta_ci95"])
    increment_fh = float(over_fh["delta_mean"])
    increment_fh_ci = float(over_fh["delta_ci95"])
    require(RESULTS, f"+{increment_pgs:.4f} ± {increment_pgs_ci:.4f}",
            f"+{increment_fh:.4f} ± {increment_fh_ci:.4f}")
    require(REPORT_TEX, f"+{increment_pgs:.3f}\\pm {increment_pgs_ci:.3f}",
            f"+{increment_fh:.3f}\\pm {increment_fh_ci:.3f}")
    return {
        "headline": headline, "joint": joint, "pgs": pgs, "fh": fh,
        "increment_pgs": increment_pgs, "increment_fh": increment_fh,
    }


def report_inputs():
    inputs = [REPORT_TEX]
    for raw in re.findall(r"\\input\{([^}]+)\}", read(REPORT_TEX)):
        inputs.append((REPORT_TEX.parent / raw).resolve())
    return inputs


def git_dirty(path):
    relative = str(path.relative_to(ROOT))
    result = subprocess.run(
        ["git", "status", "--porcelain", "--", relative], cwd=ROOT,
        check=True, capture_output=True, text=True,
    )
    return bool(result.stdout.strip())


def last_commit(path):
    result = subprocess.run(
        ["git", "rev-list", "-1", "HEAD", "--", str(path.relative_to(ROOT))],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    return result.stdout.strip()


def check_report(version, release_date, scaling, r_lock, pgs, pa_robust, time_memory):
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise AssertionError("pypdf>=4 is required to check the tracked report") from exc

    text = "\n".join(page.extract_text() or "" for page in PdfReader(REPORT_PDF).pages)
    # TeX math spacing is sometimes extracted as ``1. 751`` rather than
    # ``1.751``; normalise only whitespace inside numeric decimals.
    text = re.sub(r"(?<=\d)\.\s+(?=\d)", ".", text)
    expected_date = f"{release_date.day} {release_date.strftime('%B %Y')}"
    _, speed_lo, speed_hi = scaling
    patterns = [
        rf"v{re.escape(version)}", re.escape(expected_date),
        rf"{speed_lo:.0f}\s*[–-]\s*{speed_hi:.0f}",
        re.escape(f"{r_lock['gibbs']:.2f}"), re.escape(f"{r_lock['pa']:.0f}"),
        re.escape(f"{r_lock['gibbs_one']:.3f}"),
        re.escape(f"{r_lock['pa_one']:.0f}"),
        re.escape(f"{pgs['joint']:.3f}"), re.escape(f"{pgs['pgs']:.3f}"),
        re.escape(f"{pgs['fh']:.3f}"),
        re.escape(pa_robust[0]), re.escape(pa_robust[1]),
    ]
    for pattern in patterns:
        if not re.search(pattern, text):
            raise AssertionError(f"tracked PDF is missing /{pattern}/; rebuild it")
    compact = re.sub(r"\s+", "", text)
    for row in time_memory:
        if re.sub(r"\s+", "", row) not in compact:
            raise AssertionError(f"tracked PDF is missing time/memory row {row!r}; rebuild it")

    if not (ROOT / ".git").exists() or git_dirty(REPORT_PDF):
        return
    pdf_commit = last_commit(REPORT_PDF)
    for source in report_inputs():
        if git_dirty(source):
            raise AssertionError(f"{source.relative_to(ROOT)} changed without rebuilding the PDF")
        source_commit = last_commit(source)
        if not source_commit:
            continue
        current = subprocess.run(
            ["git", "merge-base", "--is-ancestor", source_commit, pdf_commit],
            cwd=ROOT, check=False,
        )
        if current.returncode:
            raise AssertionError(
                f"tracked PDF predates {source.relative_to(ROOT)}; rebuild it"
            )


def version_and_date():
    init = read(ROOT / "ltpred/__init__.py")
    version_match = re.search(r'^__version__\s*=\s*["\']([^"\']+)', init, re.M)
    cff = read(ROOT / "CITATION.cff")
    date_match = re.search(r"^date-released:\s*[\"']?([0-9-]+)", cff, re.M)
    if not version_match or not date_match:
        raise AssertionError("could not read release version/date")
    version = version_match.group(1)
    release_date = date.fromisoformat(date_match.group(1))
    require(ROOT / "CITATION.cff", f"version: {version}")
    expected_date = f"{release_date.day} {release_date.strftime('%B %Y')}"
    require(REPORT_TEX, f"v{version}", expected_date)
    return version, release_date


def check_efficient_inference():
    """Bind the development pilot table to its measured source and raw timings."""
    capsule = ROOT / "benchmarks/results/2026-09-05-efficient-inference"
    artifact = json.loads(read(capsule / "results.json"))
    metadata, checks, results = (artifact[key] for key in ("metadata", "checks", "results"))
    hashes = metadata["source_hashes"]
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    assert metadata["source_stable"] and metadata["source_digest"] == digest
    assert metadata["source_digest_after"] == digest
    for relative, expected in hashes.items():
        assert hashlib.sha256((capsule / "source_snapshot" / relative).read_bytes()).hexdigest() == expected
    assert all(result["source_digest"] == digest for result in results.values())
    assert checks["pipeline_input_equal"] and checks["fitting_inputs_equal"]
    assert checks["pipeline_max_abs_mean_difference"] == checks["pipeline_max_abs_variance_difference"] == 0
    for field in ("est", "var"):
        assert results["pipeline_selected"]["scores"][field] == results["pipeline_full"]["scores"][field]
    for first, second in (("pipeline_selected", "pipeline_full"), ("adult_scalar", "adult_covariance"),
                          ("inference_pa", "inference_quadrature"), ("inference_pa", "inference_gibbs")):
        assert results[first]["input_sha256"] == results[second]["input_sha256"]
    assert [r["input_sha256"] for r in results["fit_pairwise"]["replicates"]] == [
        r["input_sha256"] for r in results["fit_moment"]["replicates"]]
    source = ROOT / "report/efficient_inference.tex"
    require(source, digest[:12])
    for case, label in (
        ("inference_pa", "One family: PA"), ("inference_quadrature", "One family: quadrature"),
        ("inference_gibbs", r"One family: Gibbs, $10^6$ draws"),
        ("adult_scalar", "20,000 ADuLT: scalar"), ("adult_covariance", "20,000 ADuLT: covariance API"),
        ("pipeline_selected", "200 probands: selected relationships"),
        ("pipeline_full", "200 probands: full construction"),
    ):
        result = results[case]
        require(source, f"{label} & {result['first_call_seconds']:.3f} & "
                f"{1000 * statistics.median(result['warm_seconds']):.3f} & "
                f"{result['peak_rss_bytes'] / 2**20:.1f}")


def check_time_memory_rerun():
    """Keep the v0.6.1 measurements tied to their sources and published rows."""
    capsule = ROOT / "benchmarks/results/2026-09-09-time-memory-v061-rerun"
    artifact = json.loads(read(capsule / "results.json"))
    provenance = json.loads(read(capsule / "provenance.json"))
    assert hashlib.sha256((capsule / "results.json").read_bytes()).hexdigest() == provenance["results_sha256"]
    assert artifact["baseline_revision"] == provenance["baseline_revision"] == "52dec5294c101d4730c86d3307091be75dc47a5e"
    assert provenance["candidate_revision"] == provenance["candidate_revision_after"] == "b516271266a9fa0d95f5137954e4a284d1913ccb"
    assert provenance["source_stable"] and provenance["exit_code"] == 0
    assert provenance["source_status_before"] == provenance["source_status_after"] == ""
    assert provenance["source_hashes_before"] == provenance["source_hashes_after"]
    assert all(provenance["source_hashes_before"][path] == digest
               for path, digest in artifact["source_hashes"]["candidate"].items())
    assert artifact["benchmark_sha256"] == provenance["source_hashes_before"]["benchmarks/bench_time_memory.py"]
    assert artifact["power_guard"]["status"] == "passed"
    assert len(artifact["thread_variables"]) == 6 and set(artifact["thread_variables"].values()) == {"1"}
    labels = {
        "graph_200k": "Parent graph, 200,000 records",
        "graph_1m": "Parent graph, 1,000,000 records",
        "pa_mixed": "PA, mixed pin masks",
        "pa_pin": "PA, common pin mask",
        "pa_intervals": "PA, intervals only",
        "pa_mixture": "PA, censoring mixture",
        "pipeline": "Register scoring, 300 probands",
    }
    rows = {(row["case"], row["arm"], row["mode"]): row for row in artifact["results"]}
    expected = {(case, arm, mode) for case in labels
                for arm in ("baseline", "candidate") for mode in ("time", "allocation")}
    assert len(artifact["results"]) == 28 and set(rows) == expected
    assert set(artifact["agreement"]) == set(labels)
    original = json.loads(read(ROOT / "benchmarks/results/2026-09-09-time-memory/results.json"))
    old_rows = {(row["case"], row["arm"], row["mode"]): row for row in original["results"]}
    report_rows = []
    for case, label in labels.items():
        for arm, version in (("baseline", "0.6.0"), ("candidate", "0.6.1")):
            for mode in ("time", "allocation"):
                row = rows[case, arm, mode]
                assert row["source_hashes"] == artifact["source_hashes"][arm]
                assert row["settings"] == rows[case, "baseline", "time"]["settings"]
                assert row["settings"] == old_rows[case, arm, mode]["settings"]
                assert row["environment"]["ltpred"] == version
                assert row["environment"]["numba_threads"] == 1
                for key in ("python", "numpy", "scipy", "numba"):
                    assert row["environment"][key] == artifact["results"][0]["environment"][key]
                values = [row["peak_allocated_bytes"]] if mode == "allocation" else [
                    row["peak_rss_bytes"], row["first_call_seconds"], *row["warm_seconds"]]
                assert all(math.isfinite(value) and value > 0 for value in values)
                if mode == "time":
                    assert len(row["warm_seconds"]) == 5
        if case.startswith("graph_"):
            assert artifact["agreement"][case] == {"identical_graph": True}
            assert rows[case, "baseline", "allocation"]["graph_sha256"] == rows[case, "candidate", "allocation"]["graph_sha256"]
        else:
            fields = ("est", "var") if case.startswith("pa_") else (
                "est", "var", "n_relatives", "n_conditioned", "n_closure_only", "degree_max")
            assert artifact["agreement"][case] == dict.fromkeys(fields, 0.0)
        times = [rows[case, arm, "time"] for arm in ("baseline", "candidate")]
        medians = [statistics.median(row["warm_seconds"]) for row in times]
        first = " / ".join(f"{row['first_call_seconds']:.3f}" for row in times)
        warm = " / ".join(f"{value:.3f}" for value in medians)
        speed = f"{medians[0] / medians[1]:.2f}"
        rss = " / ".join(f"{row['peak_rss_bytes'] / 2**20:.1f}" for row in times)
        alloc = " / ".join(f"{rows[case, arm, 'allocation']['peak_allocated_bytes'] / 2**20:.1f}"
                           for arm in ("baseline", "candidate"))
        old_speed = statistics.median(old_rows[case, "baseline", "time"]["warm_seconds"]) / statistics.median(old_rows[case, "candidate", "time"]["warm_seconds"])
        require(capsule / "README.md", f"| {label} | {first} | {warm} | {speed}x |",
                f"| {label} | {rss} | {alloc} |", f"| {label} | {old_speed:.2f}x | {speed}x |")
        require(RESULTS, f"| {label} | {warm} | {speed}× | {rss} | {alloc} |")
        require(ROOT / "report/efficient_inference.tex",
                f"{label} & {warm} & ${speed}\\times$ & {rss}")
        report_rows.append(f"{label} {warm} {speed}× {rss}")
        if case in ("pa_mixed", "graph_1m"):
            require(ROOT / "README.md", f"**{speed}× faster**")
            for path in (ROOT / "benchmarks/README.md", ROOT / "report/README.md"):
                require(path, f"{speed}× faster")
        if case == "pa_mixed":
            require(ROOT / "docs/estimation.md", f"{speed}× faster", warm.replace(" / ", " → "),
                    rss.replace(" / ", " → "))
    return report_rows


def main():
    version, release_date = version_and_date()
    scaling = check_scaling()
    pa_robust = check_pa_robustness()
    cc, en, n_fam = check_ipw()
    r_lock = check_r_lock()
    pgs = check_pgs()
    check_efficient_inference()
    time_memory = check_time_memory_rerun()
    check_report(version, release_date, scaling, r_lock, pgs, pa_robust, time_memory)
    print(
        f"Evidence artifacts internally consistent: scaling {scaling[0]}; "
        f"PA stress floor {pa_robust[0]} (worst {pa_robust[1]}); "
        f"IPW {cc}/{en} (N={n_fam:,}); "
        f"R locks {r_lock['details']['gibbs']} and {r_lock['details']['pa']}; "
        f"PGS {pgs['headline']}; time/memory {len(time_memory)} matched cases; PDF v{version}."
    )


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, OSError, subprocess.SubprocessError) as exc:
        print(f"evidence check failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
