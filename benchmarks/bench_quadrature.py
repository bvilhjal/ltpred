"""The quadrature engine against PA and Gibbs on its own documented scope.

`ltpred.quadrature.estimate_liability_quadrature_arrays` computes nuclear-
family posterior moments by Gauss--Hermite factor quadrature -- the third
inference engine, and the one no other benchmark measures. This script is
that missing evidence, on the scope its docstring claims: unique ``o``/``m``/
``f``/``s*`` roles, unrelated noninbred parents, no mixture, ``h2 < 1``.

Three questions, each over structures x prevalence x encoding x seeds
(200 families per cell):

  1. **Accuracy.** Quadrature's posterior mean is exact up to its ``atol``
     tolerance, so it is a near-truth reference for PA (a moment
     approximation) and Gibbs (Monte Carlo): report corr and RMSE of each
     against quadrature, and of each against the true genetic liability.
  2. **Encoding coverage.** Classic one-sided case/control bounds and
     onset-pinned cases (the LT-FH++ encodings); censored-control mixtures
     are outside the quadrature model and excluded by design.
  3. **Cost.** Wall-time per family for each engine at the default settings,
     after a warm-up call (the quadrature engine is pure NumPy/SciPy: no
     JIT).

Output: ``bench_quadrature.csv``; one row per cell x engine plus the engine
pairs.::

    python benchmarks/run_benchmark.py --artifact bench_quadrature.csv \\
        bench_quadrature.py
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from _common import simulate_families, write_rows
from ltpred import estimate_liability, estimate_liability_pa_arrays
from ltpred.quadrature import estimate_liability_quadrature_arrays

STRUCTURES = (("o", "m", "f"), ("m", "f", "s1"), ("m", "f", "s1", "s2"))
PREVALENCES = (0.01, 0.05, 0.20)
ENCODINGS = ("classic", "pinned")
SEEDS = (1, 2, 3)
N_FAM = 200
H2 = 0.5
N_SIM = 25_000
BURN_IN = 800
TOL = 0.03


def bounds_from_simulation(sim, encoding):
    """Role-ordered bounds arrays from a simulated family set.

    ``classic`` keeps the simulator's one-sided case/control bounds;
    ``pinned`` narrows every case to a point pin at its threshold (the
    onset-pinned LT-FH++ encoding)."""
    from scipy.stats import norm
    roles = [r for r in sim.roles if r != "g"]
    t = float(norm.isf(sim.pop_prev))    # classic single-K threshold
    lower = np.full((len(sim.families), len(roles)), -np.inf)
    upper = np.full_like(lower, np.inf)
    for j, role in enumerate(roles):
        case = np.asarray(sim.status[role], bool)
        # the simulator's one-sided bounds: case (t, inf), control (-inf, t)
        lower[case, j] = t
        upper[~case, j] = t
    if encoding == "pinned":
        for j in range(len(roles)):
            pin = np.isfinite(lower[:, j]) & (upper[:, j] == np.inf)
            lower[pin, j] = t
            upper[pin, j] = t
    return roles, lower, upper


def run(roles, lower, upper, h2, seed):
    """All three engines on identical bounds; returns (name, est, seconds)."""
    families = _families_from_arrays(roles, lower, upper)
    q = estimate_liability_quadrature_arrays(roles, lower, upper, h2)
    t0 = time.perf_counter()
    q = estimate_liability_quadrature_arrays(roles, lower, upper, h2)
    t_q = time.perf_counter() - t0
    pa, _ = estimate_liability_pa_arrays(roles, lower, upper, h2)
    t0 = time.perf_counter()
    pa, _ = estimate_liability_pa_arrays(roles, lower, upper, h2)
    t_pa = time.perf_counter() - t0
    res = estimate_liability(families, h2=h2, method="gibbs", out=("genetic",),
                             n_sim=N_SIM, burn_in=BURN_IN, tol=TOL, seed=seed)
    t0 = time.perf_counter()
    res = estimate_liability(families, h2=h2, method="gibbs", out=("genetic",),
                             n_sim=N_SIM, burn_in=BURN_IN, tol=TOL, seed=seed)
    t_g = time.perf_counter() - t0
    return (("quadrature", q.est, t_q), ("pa", pa, t_pa),
            ("gibbs", res.est["genetic"], t_g))


def _families_from_arrays(roles, lower, upper):
    from ltpred.family import Family, Member
    families = []
    for f in range(lower.shape[0]):
        members = [Member(role, lower[f, j], upper[f, j])
                   for j, role in enumerate(roles)]
        families.append(Family(fam_id=f, members=members))
    return families


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-fam", type=int, default=N_FAM)
    parser.add_argument("--output-prefix",
                        default=os.path.join(os.path.dirname(
                            os.path.abspath(__file__)), "bench_quadrature"))
    args = parser.parse_args()

    print("Quadrature vs PA vs Gibbs on the quadrature engine's own scope")
    print(f"structures={STRUCTURES}  K={PREVALENCES}  encodings={ENCODINGS}")
    print(f"n_fam={args.n_fam}  seeds={SEEDS}  h2={H2}  Gibbs n_sim={N_SIM} "
          f"burn_in={BURN_IN} tol={TOL}\n")

    # warm the JIT on a small call so engine timings are steady-state
    sim = simulate_families(["m", "f"], H2, 0.05, 8, seed=0)
    roles, lo, hi = bounds_from_simulation(sim, "classic")
    run(roles, lo, hi, H2, 0)

    rows = []
    for structure in STRUCTURES:
        for prev in PREVALENCES:
            for encoding in ENCODINGS:
                for seed in SEEDS:
                    sim = simulate_families(list(structure), H2, prev,
                                            args.n_fam, seed=100 * seed + 7)
                    true_g = np.asarray(sim.genetic, float)
                    roles, lower, upper = bounds_from_simulation(
                        sim, encoding)
                    out = run(roles, lower, upper, H2, seed)
                    est = {name: e for name, e, _ in out}
                    secs = {name: t for name, _, t in out}
                    for name, e, t in out:
                        rows.append(dict(
                            structure="+".join(structure), prevalence=prev,
                            encoding=encoding, seed=seed, engine=name,
                            seconds=t, ms_per_family=1000.0 * t / args.n_fam,
                            corr_truth=float(np.corrcoef(e, true_g)[0, 1])))
                    for a, b in (("pa", "quadrature"), ("gibbs", "quadrature"),
                                 ("pa", "gibbs")):
                        diff = est[a] - est[b]
                        rows.append(dict(
                            structure="+".join(structure), prevalence=prev,
                            encoding=encoding, seed=seed, engine=f"{a}_vs_{b}",
                            seconds="", ms_per_family="",
                            corr_truth=float(np.corrcoef(est[a], est[b])[0, 1]),
                            rmse=float(np.sqrt(np.mean(diff ** 2))),
                            max_abs=float(np.max(np.abs(diff)))))
                    pa_row = next(r for r in rows
                                  if r["engine"] == "pa_vs_quadrature")
                    print(f"{'+'.join(structure):12s} K={prev:4.2f} "
                          f"{encoding:8s} seed {seed}: "
                          f"RMSE(PA,quad)={pa_row['rmse']:.4f}  "
                          f"corr(PA,truth)="
                          f"{next(r['corr_truth'] for r in rows if r['engine'] == 'pa' and r['seed'] == seed and r['structure'] == '+'.join(structure) and r['prevalence'] == prev and r['encoding'] == encoding):.4f}"
                          f"  {secs['quadrature'] / secs['pa']:.0f}x PA time"
                          f"  quad {1000 * secs['quadrature'] / args.n_fam:.2f} ms/fam")

    out_csv = f"{args.output_prefix}.csv"
    fields = ["structure", "prevalence", "encoding", "seed", "engine",
              "seconds", "ms_per_family", "corr_truth", "rmse", "max_abs"]
    for r in rows:
        r.setdefault("rmse", "")
        r.setdefault("max_abs", "")
    write_rows(out_csv, rows, fields=fields)

    def cell_stats(engine, key):
        # a cell whose simulated cohort happens to contain no case anywhere
        # gives constant estimates and a NaN correlation; those cells carry
        # no accuracy information, so summary statistics skip them
        vals = np.array([r[key] for r in rows if r["engine"] == engine
                         and r[key] != ""], float)
        vals = vals[np.isfinite(vals)]
        return float(vals.mean()), float(vals.std(ddof=1) / np.sqrt(len(vals)))

    print("\nAcross all cells (mean ± SE)")
    for engine, label in (
            ("pa_vs_quadrature", "RMSE(PA, quadrature)"),
            ("gibbs_vs_quadrature", "RMSE(Gibbs, quadrature)")):
        mean, se = cell_stats(engine, "rmse")
        print(f"  {label:28s} {mean:.5f} ± {se:.5f}")
    mean, se = cell_stats("pa", "corr_truth")
    print(f"  {'corr(PA, truth)':28s} {mean:.5f} ± {se:.5f}")
    mean, se = cell_stats("quadrature", "corr_truth")
    print(f"  {'corr(quad, truth)':28s} {mean:.5f} ± {se:.5f}")
    for engine in ("quadrature", "pa", "gibbs"):
        mean, se = cell_stats(engine, "ms_per_family")
        print(f"  {engine:28s} {mean:9.3f} ± {se:.3f} ms/family")
    print(f"\nwrote {os.path.basename(out_csv)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
