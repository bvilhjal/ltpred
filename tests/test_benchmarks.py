"""Regression checks for benchmark code: the PGS joint-model evaluation, the
register benchmark's calendar-time design, and the provenance runner."""

import csv
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from _helpers import load_script
from benchmarks import run_benchmark
from ltpred.covariance import kinship_from_pedigree


PGS = load_script("benchmarks/bench_pgs_comparison.py")
REGISTER = load_script("benchmarks/bench_register_pipeline.py")


# --- bench_pgs_comparison.py ---


def test_joint_crossfit_is_deterministic_and_holds_out_own_target():
    rng = np.random.default_rng(43)
    g = rng.normal(size=40)
    pgs = 0.6 * g + rng.normal(size=40)
    fh = 0.4 * g + rng.normal(size=40)

    pred = PGS._cross_fitted_joint_predictions(g, pgs, fh, 2, seed=17)
    repeated = PGS._cross_fitted_joint_predictions(
        g, pgs, fh, 2, seed=17)
    np.testing.assert_array_equal(pred, repeated)

    changed = g.copy()
    changed[0] += 1000.0
    changed_pred = PGS._cross_fitted_joint_predictions(
        changed, pgs, fh, 2, seed=17)
    assert changed_pred[0] == pred[0]


def test_pgs_smoke_records_crossfit_design(tmp_path, monkeypatch):
    args = SimpleNamespace(
        seed=7, n_fam=40, m_snps=30, n_causal=5, h2=0.5, prev=0.1,
        fam=["m", "f", "s1"], train_frac=0.5, pgs_backend="numpy",
        joint_folds=2,
    )
    rows = PGS.run_rep(args, rep=0)
    monkeypatch.setattr(PGS, "HERE", str(tmp_path))
    path = PGS.write_csv(rows, args)

    with open(path, newline="", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    joint = next(row for row in written if row["arm"] == PGS.JOINT)
    assert joint["joint_fit_design"] == "seeded_shuffled_test_kfold_ols"
    assert joint["joint_folds"] == "2"
    assert joint["joint_seed"] == str(PGS._joint_crossfit_seed(7, 0))
    assert np.isfinite(float(joint["r2_g"]))


# --- bench_register_pipeline.py ---


@pytest.mark.parametrize("self_relationship", [1.0, 1.25])
def test_prior_future_risk_is_cip_increment_conditional_on_survival(self_relationship):
    # At the prior mean, the future risk is the CIP increment given survival to
    # the index age, for an outbred and an inbred (A_xx = 1.25) target alike:
    # the target-specific residual variance keeps the full liability unit-scale.
    # The outbred case uses the default residual variance.
    scale2 = REGISTER.H2 * self_relationship + (1.0 - REGISTER.H2)
    residual_var = (None if self_relationship == 1.0
                    else np.array([(1.0 - REGISTER.H2) / scale2]))
    risk = REGISTER.future_case_prob(
        est_g=np.array([0.0]),
        est_var=np.array([REGISTER.H2 * self_relationship / scale2]),
        residual_var=residual_var)
    cip_index = np.interp(REGISTER.INDEX_AGE, REGISTER.AGE_GRID, REGISTER.TRUE_CIP)
    cip_eval = np.interp(REGISTER.EVAL_AGE, REGISTER.AGE_GRID, REGISTER.TRUE_CIP)
    expected = (cip_eval - cip_index) / (1.0 - cip_index)

    np.testing.assert_allclose(risk, [expected], rtol=1e-14, atol=0.0)


def test_build_register_uses_the_public_inbreeding_scale():
    # x is the child of full siblings, so A_xx = 1.25. Reproduce the two RNG
    # draws and require both the simulated truth and observation process to use
    # the unit-full-liability scale used by construct_covmat_from_kinship.
    ids = ["gm", "gf", "sA", "sB", "x"]
    father = [None, None, "gf", "gf", "sA"]
    mother = [None, None, "gm", "gm", "sB"]
    seed = 913
    observed = REGISTER.build_register(
        np.random.default_rng(seed), ids, father, mother)
    status, age, onset, genetic, _, residual_var = observed

    _, relationship = kinship_from_pedigree(ids, father, mother)
    replay = np.random.default_rng(seed)
    # The same canonical draw the public generator makes: one standard-normal
    # vector through the unique Cholesky factor, not multivariate_normal's
    # sign-arbitrary SVD, which is why a seed now reproduces across builds.
    from ltpred.simulate import _stable_factor
    raw_genetic = replay.standard_normal(len(ids)) @ _stable_factor(
        REGISTER.H2 * relationship).T
    residual = replay.standard_normal(len(ids)) * np.sqrt(1.0 - REGISTER.H2)
    scale = np.sqrt(
        REGISTER.H2 * np.diag(relationship) + (1.0 - REGISTER.H2))
    liability = (raw_genetic + residual) / scale

    assert scale[-1] > 1.0
    np.testing.assert_allclose(genetic, raw_genetic / scale, rtol=0, atol=0)
    np.testing.assert_allclose(
        residual_var, (1.0 - REGISTER.H2) / scale ** 2, rtol=0, atol=0)
    raw_full_cov = (REGISTER.H2 * relationship
                    + (1.0 - REGISTER.H2) * np.eye(len(ids)))
    standardized_cov = raw_full_cov / np.outer(scale, scale)
    np.testing.assert_allclose(
        np.diag(standardized_cov), np.ones(len(ids)), rtol=0, atol=1e-15)

    need = 1.0 - REGISTER.norm.cdf(liability)
    expected_onset = np.full(len(ids), np.inf)
    event = need <= REGISTER.TRUE_CIP[-1]
    expected_onset[event] = np.maximum(
        np.interp(need[event], REGISTER.TRUE_CIP, REGISTER.AGE_GRID), 1e-9)
    expected_status = expected_onset <= REGISTER.EVAL_AGE
    expected_age = np.where(expected_status, expected_onset, REGISTER.EVAL_AGE)
    np.testing.assert_array_equal(status, expected_status)
    np.testing.assert_allclose(age, expected_age, rtol=0, atol=0)
    np.testing.assert_allclose(onset, expected_onset, rtol=0, atol=0)


def test_benchmark_generators_are_the_public_ones():
    """The ledger and the tutorial must not be able to drift apart.

    ``build_register`` and ``pedigree_birth_times`` here are thin bindings of
    this benchmark's constants around :mod:`ltpred.simulate`. If either stops
    delegating, the promoted public generator and the published benchmark would
    be two different simulations producing incomparable numbers -- so pin the
    delegation bit for bit rather than trusting the import line.
    """
    from ltpred.simulate import pedigree_birth_times, simulate_register_liabilities

    ids, father, mother = REGISTER.simulate_population(
        np.random.default_rng(17), n_founder_pairs=25, gens=2)

    np.testing.assert_array_equal(
        REGISTER.pedigree_birth_times(ids, father, mother),
        pedigree_birth_times(ids, father, mother,
                             base_birth_year=REGISTER.BASE_BIRTH_TIME,
                             generation_years=REGISTER.GENERATION_YEARS))

    bound = simulate_register_liabilities(
        np.random.default_rng(23), ids, father, mother, h2=REGISTER.H2,
        cip_ages=REGISTER.AGE_GRID, cip_values=REGISTER.TRUE_CIP,
        eval_age=REGISTER.EVAL_AGE)
    expected = (bound.status, bound.age, bound.onset, bound.genetic,
                bound.birth_time, bound.residual_var)
    observed = REGISTER.build_register(np.random.default_rng(23), ids, father,
                                     mother)
    assert len(observed) == len(expected)
    for got, want in zip(observed, expected):
        # atol=0/rtol=0: same generator, same seed, same constants must give the
        # same bits, not merely the same numbers to within a tolerance
        np.testing.assert_array_equal(got, want)


# --- run_benchmark.py ---


def test_source_state_does_not_exclude_tracked_repo_inputs(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    input_path = root / "data" / "cohort.bed"
    calls = []

    def fake_git(*args):
        if args[:2] == ("rev-parse", "HEAD"):
            return "a" * 40 + "\n"
        if args[:3] == ("ls-files", "--cached", "--"):
            return "data/cohort.bed\n"
        calls.append(args)
        return " M data/cohort.bed\n"

    monkeypatch.setattr(run_benchmark, "ROOT", root)
    monkeypatch.setattr(run_benchmark, "_git", fake_git)
    state = run_benchmark._source_state([input_path])

    assert state == {
        "commit": "a" * 40,
        "dirty": True,
        "status": [" M data/cohort.bed"],
    }
    assert ":(exclude,literal)data/cohort.bed" not in calls[0]


def test_source_state_excludes_declared_untracked_repo_inputs(
        tmp_path, monkeypatch):
    root = tmp_path / "repo"
    input_path = root / "data" / "cohort.bed"
    calls = []

    def fake_git(*args):
        if args[:2] == ("rev-parse", "HEAD"):
            return "a" * 40 + "\n"
        if args[:3] == ("ls-files", "--cached", "--"):
            return ""
        calls.append(args)
        return ""

    monkeypatch.setattr(run_benchmark, "ROOT", root)
    monkeypatch.setattr(run_benchmark, "_git", fake_git)
    state = run_benchmark._source_state([input_path])

    assert state == {"commit": "a" * 40, "dirty": False, "status": []}
    assert ":(exclude,literal)data/cohort.bed" in calls[0]


def test_resolve_script_rejects_nonbenchmark_and_path_escape(tmp_path, monkeypatch):
    here = tmp_path / "benchmarks"
    here.mkdir()
    (here / "bench_ok.py").write_text("", encoding="utf-8")
    (here / "helper.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(run_benchmark, "HERE", here)

    assert run_benchmark._resolve_script("bench_ok.py") == here / "bench_ok.py"
    for invalid in ("helper.py", "../bench_ok.py", "missing.py"):
        with pytest.raises(ValueError, match=r"bench_\*\.py"):
            run_benchmark._resolve_script(invalid)


def test_only_csv_outputs_are_retained_artifacts(tmp_path, monkeypatch):
    # Figures are gitignored, so a manifest row must not hash one as retained.
    monkeypatch.setattr(run_benchmark, "HERE", tmp_path)
    assert run_benchmark._resolve_artifact("bench_ok.csv") == tmp_path / "bench_ok.csv"
    for invalid in ("bench_ok.png", "../bench_ok.csv"):
        with pytest.raises(ValueError, match="CSV files"):
            run_benchmark._resolve_artifact(invalid)


def test_runner_records_commit_environment_and_artifact_hash(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    here = root / "benchmarks"
    here.mkdir(parents=True)
    (root / "ltpred").mkdir()
    (root / "ltpred" / "__init__.py").write_text(
        '__version__ = "9.8.7"\n', encoding="utf-8")
    script = here / "bench_smoke.py"
    script.write_text(
        "from pathlib import Path\n"
        "Path(__file__).with_suffix('.csv').write_text('x\\n1\\n')\n",
        encoding="utf-8",
    )
    (here / "bench_smoke.csv").write_text("x\n1\n", encoding="utf-8")
    os.utime(here / "bench_smoke.csv", ns=(1_000_000_000, 1_000_000_000))
    input_path = root / "input.dat"
    input_path.write_bytes(b"input")
    manifest = here / "manifest.jsonl"
    monkeypatch.setattr(run_benchmark, "ROOT", root)
    monkeypatch.setattr(run_benchmark, "HERE", here)
    monkeypatch.setattr(run_benchmark, "DEFAULT_MANIFEST", manifest)
    monkeypatch.setattr(
        run_benchmark, "_source_state",
        lambda ignored_paths=(): {
            "commit": "a" * 40, "dirty": False, "status": []},
    )

    assert run_benchmark.main(
        ["--artifact", "bench_smoke.csv", "--input", str(input_path),
         "bench_smoke.py"]) == 0
    record = json.loads(manifest.read_text(encoding="utf-8"))
    assert record["source"]["commit"] == "a" * 40
    assert record["source"]["dirty"] is False
    assert record["source"]["status"] == []
    assert record["environment"]["packages"]["ltpred"] == "9.8.7"
    assert record["artifacts"][0]["path"] == "bench_smoke.csv"
    assert record["artifacts"][0]["touched"] is True
    assert len(record["artifacts"][0]["sha256"]) == 64
    assert record["inputs"][0]["sha256"] == (
        "c96c6d5be8d08a12e7b5cdc1b207fa6b2430974c86803d8891675e76fd992c20")
    assert record["source"]["unchanged_after_run"] is True
    assert record["changed_artifacts"] == []
    assert record["wrapper_exit_code"] == 0
    assert record["provenance_errors"] == []


def test_runner_refuses_dirty_source_by_default(tmp_path, monkeypatch):
    here = tmp_path / "benchmarks"
    here.mkdir()
    (here / "bench_smoke.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(run_benchmark, "HERE", here)
    monkeypatch.setattr(
        run_benchmark, "_source_state",
        lambda ignored_paths=(): {
            "commit": "a" * 40, "dirty": True,
            "status": [" M ltpred/estimate.py"]},
    )

    with pytest.raises(SystemExit, match="source tree is dirty"):
        run_benchmark.main(["--artifact", "bench_smoke.csv", "bench_smoke.py"])


def test_runner_rejects_untouched_preexisting_artifact(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    here = root / "benchmarks"
    here.mkdir(parents=True)
    (root / "ltpred").mkdir()
    (root / "ltpred" / "__init__.py").write_text(
        '__version__ = "9.8.7"\n', encoding="utf-8")
    (here / "bench_nop.py").write_text("", encoding="utf-8")
    (here / "bench_stale.csv").write_text("stale\n", encoding="utf-8")
    manifest = here / "manifest.jsonl"
    monkeypatch.setattr(run_benchmark, "ROOT", root)
    monkeypatch.setattr(run_benchmark, "HERE", here)
    monkeypatch.setattr(run_benchmark, "DEFAULT_MANIFEST", manifest)
    monkeypatch.setattr(
        run_benchmark, "_source_state",
        lambda ignored_paths=(): {
            "commit": "a" * 40, "dirty": False, "status": []},
    )

    assert run_benchmark.main(
        ["--artifact", "bench_stale.csv", "bench_nop.py"]) == 2
    record = json.loads(manifest.read_text(encoding="utf-8"))
    assert record["untouched_artifacts"] == ["bench_stale.csv"]
    assert record["artifacts"][0]["touched"] is False
    assert record["wrapper_exit_code"] == 2
