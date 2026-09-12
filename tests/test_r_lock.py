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


def _fixture_families(name="input_tbl.csv"):
    from ltpred import families_from_columns
    with open(FIXTURES / name, newline="", encoding="utf-8") as fh:
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


def test_ltfhplus_fixture_records_seed():
    with open(FIXTURES / "ltfhplus_gibbs.csv", newline="",
              encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert {int(r["seed"]) for r in rows} == {20260816}
    assert all(r["r_version"] and r["rng_kind"] for r in rows)


def test_pa_matches_locked_ltfgrs_scores():
    from ltpred import estimate_liability
    families = _fixture_families()
    r_est = _aligned(families, _r_scores("ltfgrs_pa.csv"))
    ours = estimate_liability(families, h2=0.5).est["genetic"]
    corr, rmse = _corr_rmse(ours, r_est)
    # The 200-family benchmark lock is corr 1.0000 / RMSE 0.000087, and this
    # 48-family fixture measures corr 0.99999999 / RMSE 2.53e-5. PA is
    # deterministic -- no Monte-Carlo slack to absorb -- so the band is set
    # just wide enough for platform float noise. A looser rmse < 1e-3 would
    # have licensed a systematic multiplicative bias in every PA score while
    # still passing, which is exactly what this fixture exists to catch.
    assert corr > 0.9999999
    assert rmse < 1e-4


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


# LT-FH++ (age-of-onset) cohort: the same 48 families written under both case
# encodings LTFHPlus::prepare_LTFHPlus_input can emit -- the point pin
# (use_fixed_case_thr = TRUE; ltpred's default) and the one-sided interval
# (use_fixed_case_thr = FALSE; the R default). The classic fixture above has no
# pinned row and six case rows, so it exercises neither.
AGE_ENCODINGS = ["pin", "interval"]


def test_age_fixtures_share_families_and_differ_only_in_case_upper():
    pin = _fixture_families("input_tbl_age_pin.csv")
    interval = _fixture_families("input_tbl_age_interval.csv")
    n_pinned = 0
    for fp, fi in zip(pin, interval):
        assert fp.fam_id == fi.fam_id
        for mp, mi in zip(fp.members, fi.members):
            assert (mp.role, mp.lower) == (mi.role, mi.lower)
            if mp.lower == mp.upper:
                n_pinned += 1
                assert mi.upper == np.inf
            else:
                assert mp.upper == mi.upper
    assert n_pinned == 16


@pytest.mark.parametrize("encoding", AGE_ENCODINGS)
def test_pa_matches_locked_ltfgrs_scores_with_age_of_onset(encoding):
    from ltpred import estimate_liability
    families = _fixture_families(f"input_tbl_age_{encoding}.csv")
    r_est = _aligned(families, _r_scores(f"ltfgrs_pa_age_{encoding}.csv"))
    ours = estimate_liability(families, h2=0.5).est["genetic"]
    corr, rmse = _corr_rmse(ours, r_est)
    # Measured at generation: interval corr 0.99999985 / RMSE 1.8e-4; pin corr
    # 0.99999796 / RMSE 6.7e-4 (max 3.3e-3). The pin residual is the 0.6.0
    # design difference: ltpred conditions all pins jointly and exactly before
    # the sequential fold, LTFGRS folds them one at a time. The bands sit just
    # above those values so a systematic bias in the case branch still fails.
    assert corr > 0.99999
    assert rmse < (2e-3 if encoding == "pin" else 5e-4)


@pytest.mark.jit_required
@pytest.mark.parametrize("encoding", AGE_ENCODINGS)
def test_gibbs_matches_locked_ltfhplus_scores_with_age_of_onset(encoding):
    from ltpred import estimate_liability
    families = _fixture_families(f"input_tbl_age_{encoding}.csv")
    r_est = _aligned(families, _r_scores(f"ltfhplus_gibbs_age_{encoding}.csv"))
    ours = estimate_liability(families, h2=0.5, method="gibbs",
                              n_sim=100_000, burn_in=1000, tol=0.01,
                              seed=20260912).est["genetic"]
    corr, rmse = _corr_rmse(ours, r_est)
    # measured at generation: corr 0.99991 / RMSE 3.6e-3 (pin) and
    # 0.99994 / 3.5e-3 (interval); same Monte-Carlo bands as the classic lock
    assert corr > 0.999
    assert rmse < 0.01
