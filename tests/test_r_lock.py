"""Locked R-package reference scores as an ordinary pytest.

The full LTFHPlus/LTFGRS lock (RESULTS.md section 30) is an opt-in benchmark
needing R; a numerical regression in the port would otherwise pass the whole
suite until someone reran it. These tests recompute ltpred on a committed
48-family table and compare against the committed R outputs (LTFHPlus 2.2.0
Gibbs, LTFGRS 1.0.1 PA) — see tests/fixtures/r_lock/README.md.
"""
import csv
import pathlib

import numpy as np
import pytest

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "r_lock"


def _parse_bound(text):
    if text in ("-Inf", "-inf"):
        return -np.inf
    if text in ("Inf", "inf"):
        return np.inf
    return float(text)


def _fixture_families():
    from ltpred import families_from_columns
    with open(FIXTURES / "input_tbl.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return families_from_columns(
        [r["fam_ID"] for r in rows],
        [r["role"] for r in rows],
        [_parse_bound(r["lower"]) for r in rows],
        [_parse_bound(r["upper"]) for r in rows],
        pid=[r["indiv_ID"] for r in rows],
    )


def _r_scores(name):
    with open(FIXTURES / name, newline="", encoding="utf-8") as fh:
        return {r["fam_ID"]: float(r["genetic_est"])
                for r in csv.DictReader(fh)}


def _aligned(families, by_fam):
    return np.array([by_fam[str(f.fam_id)] for f in families])


def _corr_rmse(a, b):
    return (float(np.corrcoef(a, b)[0, 1]),
            float(np.sqrt(np.mean((a - b) ** 2))))


def test_pa_matches_locked_ltfgrs_scores():
    from ltpred import estimate_liability
    families = _fixture_families()
    r_est = _aligned(families, _r_scores("ltfgrs_pa.csv"))
    ours = estimate_liability(families, h2=0.5).est["genetic"]
    corr, rmse = _corr_rmse(ours, r_est)
    # the 200-family benchmark lock is corr 1.0000 / RMSE 0.000087
    assert corr > 0.99999
    assert rmse < 1e-3


@pytest.mark.jit_required
def test_gibbs_matches_locked_ltfhplus_scores():
    from ltpred import estimate_liability
    families = _fixture_families()
    r_est = _aligned(families, _r_scores("ltfhplus_gibbs.csv"))
    ours = estimate_liability(families, h2=0.5, method="gibbs",
                              n_sim=100_000, burn_in=1000, tol=0.01,
                              seed=20260816).est["genetic"]
    corr, rmse = _corr_rmse(ours, r_est)
    # the 200-family benchmark lock is corr 0.9999 / RMSE 0.0041; the looser
    # bands here absorb seed-level Monte-Carlo differences on 48 families
    assert corr > 0.999
    assert rmse < 0.01
