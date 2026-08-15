import pathlib

from scripts.check_results import (
    GUARDS,
    Guard,
    parse_quoted,
    quoted_tuples,
    run_guard,
    run_guards,
    within_quoted_precision,
)
from scripts.make_results import generated_table_texts

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _synthetic_guard(tmp_path, anchor=r"eff-N proxy ([\d.]+)×"):
    (tmp_path / "synthetic.csv").write_text("cell,eff\na,1.02\nb,1.06\n",
                                            encoding="utf-8")

    def recompute(path):
        import csv
        with open(path, newline="", encoding="utf-8") as handle:
            vals = [float(r["eff"]) for r in csv.DictReader(handle)]
        return (sum(vals) / len(vals),)

    return Guard("synthetic", "synthetic.csv", recompute, anchor)


def test_parse_quoted_strips_commas_signs_and_counts_decimals():
    assert parse_quoted("71,264") == (71264.0, 0)
    assert parse_quoted("+0.001") == (0.001, 3)
    assert parse_quoted("-0.0061") == (-0.0061, 4)


def test_within_quoted_precision_rounds_to_the_quoted_last_decimal():
    assert within_quoted_precision(16.4604, "16.460")
    assert not within_quoted_precision(16.4606, "16.460")
    assert within_quoted_precision(0.5318, "0.532")
    assert not within_quoted_precision(0.05092, "0.05091")
    assert within_quoted_precision(296.49, "296")
    assert not within_quoted_precision(296.51, "296")


def test_quoted_tuples_deduplicates_identical_repeats():
    text = "**203–492× faster** ... the observed **203–492×** object-path"
    assert quoted_tuples(text, r"\*\*(\d+)–(\d+)×") == [("203", "492")]


def test_run_guard_ok_when_recomputation_rounds_to_quoted(tmp_path):
    guard = _synthetic_guard(tmp_path)
    ok, detail = run_guard(tmp_path, "an eff-N proxy 1.04× over case/control",
                           guard)
    assert ok, detail


def test_run_guard_mismatch_reports_computed_and_quoted(tmp_path):
    guard = _synthetic_guard(tmp_path)
    ok, detail = run_guard(tmp_path, "an eff-N proxy 1.05× over case/control",
                           guard)
    assert not ok
    assert "computed (1.0400) vs quoted (1.05)" in detail


def test_run_guard_missing_anchor_fails(tmp_path):
    guard = _synthetic_guard(tmp_path)
    ok, detail = run_guard(tmp_path, "nothing to anchor on", guard)
    assert not ok
    assert "anchor not found" in detail


def test_run_guard_ambiguous_anchor_fails(tmp_path):
    guard = _synthetic_guard(tmp_path)
    text = "eff-N proxy 1.04× here, but eff-N proxy 1.05× there"
    ok, detail = run_guard(tmp_path, text, guard)
    assert not ok
    assert "distinct quoted values" in detail


def test_run_guard_repeated_identical_anchor_is_ok(tmp_path):
    guard = _synthetic_guard(tmp_path)
    text = "eff-N proxy 1.04× in the headline; eff-N proxy 1.04× in the table"
    ok, detail = run_guard(tmp_path, text, guard)
    assert ok, detail


def test_real_guard_list_passes_against_repo_artifacts():
    results = run_guards(ROOT)
    assert len(results) == len(GUARDS)
    failures = [(guard.name, detail) for guard, ok, detail in results if not ok]
    assert failures == []


def test_generated_paper_tables_match_committed_artifacts():
    stale = [str(path) for path, expected in generated_table_texts(ROOT)
             if (ROOT / path).read_text(encoding="utf-8") != expected]
    assert stale == [], "regenerate with: python scripts/make_results.py"


def test_report_consumes_generated_load_bearing_tables():
    report = (ROOT / "report/ltpred_methods.tex").read_text(encoding="utf-8")
    for name in ("headlines", "gwas_power", "integrated_panel", "confounding",
                 "pgs_comparison", "fit_heritability"):
        assert rf"\input{{../paper/tables/{name}.tex}}" in report
