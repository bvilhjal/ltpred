"""Regression checks for the PGS benchmark's joint-model evaluation."""

import csv
import importlib.util
import pathlib
import sys
from types import SimpleNamespace

import numpy as np


BENCHMARKS = pathlib.Path(__file__).parents[1] / "benchmarks"
SPEC = importlib.util.spec_from_file_location(
    "bench_pgs_comparison", BENCHMARKS / "bench_pgs_comparison.py")
MODULE = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(BENCHMARKS))
try:
    SPEC.loader.exec_module(MODULE)
finally:
    sys.path.pop(0)


def test_joint_crossfit_is_deterministic_and_holds_out_own_target():
    rng = np.random.default_rng(43)
    g = rng.normal(size=40)
    pgs = 0.6 * g + rng.normal(size=40)
    fh = 0.4 * g + rng.normal(size=40)

    pred = MODULE._cross_fitted_joint_predictions(g, pgs, fh, 2, seed=17)
    repeated = MODULE._cross_fitted_joint_predictions(
        g, pgs, fh, 2, seed=17)
    np.testing.assert_array_equal(pred, repeated)

    changed = g.copy()
    changed[0] += 1000.0
    changed_pred = MODULE._cross_fitted_joint_predictions(
        changed, pgs, fh, 2, seed=17)
    assert changed_pred[0] == pred[0]


def test_pgs_smoke_records_crossfit_design(tmp_path, monkeypatch):
    args = SimpleNamespace(
        seed=7, n_fam=40, m_snps=30, n_causal=5, h2=0.5, prev=0.1,
        fam=["m", "f", "s1"], train_frac=0.5, pgs_backend="numpy",
        joint_folds=2,
    )
    rows = MODULE.run_rep(args, rep=0)
    monkeypatch.setattr(MODULE, "HERE", str(tmp_path))
    path = MODULE.write_csv(rows, args)

    with open(path, newline="", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    joint = next(row for row in written if row["arm"] == MODULE.JOINT)
    assert joint["joint_fit_design"] == "seeded_shuffled_test_kfold_ols"
    assert joint["joint_folds"] == "2"
    assert joint["joint_seed"] == str(MODULE._joint_crossfit_seed(7, 0))
    assert np.isfinite(float(joint["r2_g"]))
