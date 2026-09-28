"""ltpred's throughput at population scale: the register path and the array PA.

The published comparison point for this problem class is LTFHPlus's scaling
claim ("350 000 individuals in under 25 minutes on 32 cores", Pedersen et al.
2022, AJHG, Table S13). This script measures ltpred's two production paths on
matched-scale synthetic cohorts so any future head-to-head has a same-shape
number to sit against (§30/§34 lock fold-times are per-package wall-times on
200 families; this is the cohort-scale complement):

  * **register** -- `estimate_liabilities` (pinned-onset LT-FH++, deterministic
    PA, ``max_degree=2``) on multi-generation synthetic registers with
    10k / 50k / 200k probands;
  * **array** -- `estimate_liability_pa_chunked` on 350k role-array rows with
    LT-FH++-style bounds (pinned cases, censored controls, absent relatives),
    the input shape of the personalised lock.

Reports probands (families) per second, milliseconds per proband and the
process's peak RSS. Timing is the estimator call only (simulation excluded),
after a warm-up call; the register arm loops probands serially, exactly as
the pipeline does. Pin threads with ``NUMBA_NUM_THREADS`` and record the load
average beside any quoted number -- throughput claims without a thread count
and a load line are not comparable (review 2026-09f, lesson 2).::

    NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
        python benchmarks/run_benchmark.py \\
        --artifact bench_population_scale.csv bench_population_scale.py
"""

from __future__ import annotations

import argparse
import os
import resource
import sys
import time

import numpy as np

from _common import write_rows

REGISTER_SIZES = (10_000, 50_000, 200_000)
ARRAY_ROWS = 350_000
H2 = 0.5
K_POP = 0.10
SEED = 20260928
CIP_AGES = np.arange(0.0, 121.0)
CIP_VALUES = K_POP / (1.0 + np.exp((60.0 - CIP_AGES) / 8.0))


def _peak_rss_mib():
    # ru_maxrss is bytes on macOS and KiB on Linux; it includes the
    # interpreter and Numba, so it is an upper bound on the estimator's
    # own working set
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024.0 * 1024.0) if sys.platform == "darwin" else rss / 1024.0


def _trio_register(rng, n_probands):
    """Vectorised two-generation register: unrelated founder couples, 1-3
    children each, children are the proband pool (parents degree 1, full
    siblings degree 2). ``simulate_pedigree``'s multi-generation construction
    is quadratic in the pool (an ``ids.index`` call inside its mating loop),
    which dominates at these sizes, so this benchmark builds the trios
    directly."""
    children_per = []
    n_children = 0
    while n_children < n_probands:      # 1-3 children per couple, mean 2
        k = int(rng.integers(1, 4))
        children_per.append(k)
        n_children += k
    n_couples = len(children_per)
    n = 2 * n_couples + n_children
    ids = [f"p{i}" for i in range(n)]
    father = [None] * n
    mother = [None] * n
    couples = [(2 * i, 2 * i + 1) for i in range(n_couples)]
    child = 2 * n_couples
    for c, k in enumerate(children_per):
        fa, mo = couples[c]
        for _ in range(int(k)):
            father[child] = ids[fa]
            mother[child] = ids[mo]
            child += 1
    return ids, father, mother


def register_arm(n_probands, seed):
    from ltpred import estimate_liabilities
    rng = np.random.default_rng(seed)
    ids, father, mother = _trio_register(rng, n_probands)
    n = len(ids)
    age = rng.uniform(20, 90, n)
    status = (rng.random(n)
              < K_POP / (1.0 + np.exp((60.0 - age) / 8.0))).astype(int)
    probands = [ids[i] for i in range(len(ids) - n_probands, len(ids))]
    kwargs = dict(ids=ids, father=father, mother=mother, probands=probands,
                  status=status, age=age, use="gwas", cip_ages=CIP_AGES,
                  cip_values=CIP_VALUES, k_pop=K_POP, h2=H2, max_degree=2)
    estimate_liabilities(**dict(kwargs, probands=probands[:64]))   # warm-up
    t0 = time.perf_counter()
    scores = estimate_liabilities(**kwargs)
    seconds = time.perf_counter() - t0
    return dict(arm="register", n=n_probands, seconds=seconds,
                per_second=len(probands) / seconds,
                ms_per_proband=1000.0 * seconds / len(probands),
                population=len(ids), peak_rss_mib=_peak_rss_mib(),
                corr_est_finite=bool(np.all(np.isfinite(scores.est))))


def array_arm(n_rows, seed):
    from ltpred.chunked import estimate_liability_pa_chunked
    rng = np.random.default_rng(seed + 1)
    roles = ["m", "f", "s1", "s2"]
    T = 1.6449
    case = rng.random((n_rows, 4)) < 0.10
    lower = np.where(case, T, -np.inf)
    upper = np.where(case, np.inf, T)
    lower[:, 0], upper[:, 0] = T, np.inf
    absent = rng.random((n_rows, 4)) < 0.12
    absent[:, 0] = False
    lower[absent], upper[absent] = -np.inf, np.inf
    pins = (~absent) & (rng.random((n_rows, 4)) < 0.06)
    pins[:, 0] = False
    lower[pins], upper[pins] = 1.9, 1.9
    estimate_liability_pa_chunked(roles, lower[:1024], upper[:1024], H2)
    t0 = time.perf_counter()
    est, _ = estimate_liability_pa_chunked(roles, lower, upper, H2)
    seconds = time.perf_counter() - t0
    return dict(arm="array_pa", n=n_rows, seconds=seconds,
                per_second=n_rows / seconds,
                ms_per_proband=1000.0 * seconds / n_rows,
                population=n_rows, peak_rss_mib=_peak_rss_mib(),
                corr_est_finite=bool(np.all(np.isfinite(est))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-prefix",
                        default=os.path.join(os.path.dirname(
                            os.path.abspath(__file__)), "bench_population_scale"))
    args = parser.parse_args()

    try:
        from numba import get_num_threads
        threads = get_num_threads()
    except ImportError:  # pragma: no cover
        threads = 1

    print("Population-scale throughput (register pipeline + array PA)")
    print(f"h2={H2}  K={K_POP}  Numba threads={threads}  seed={SEED}")
    print("Estimand: pinned-onset LT-FH++ genetic liability, deterministic PA")
    print("Timing excludes simulation; warm-up call precedes each arm.\n")

    rows = []
    for n in REGISTER_SIZES:
        row = register_arm(n, SEED + n)
        rows.append(row)
        print(f"register  {row['n']:7d} probands (pop {row['population']:7d}): "
              f"{row['seconds']:8.2f} s  {row['per_second']:9.0f}/s  "
              f"{row['ms_per_proband']:7.3f} ms/proband  "
              f"RSS {row['peak_rss_mib']:.0f} MiB")
    row = array_arm(ARRAY_ROWS, SEED)
    rows.append(row)
    print(f"array PA  {row['n']:7d} rows:                 "
          f"{row['seconds']:8.2f} s  {row['per_second']:9.0f}/s  "
          f"{row['ms_per_proband']:7.3f} ms/row      "
          f"RSS {row['peak_rss_mib']:.0f} MiB")

    for r in rows:
        r["numba_threads"] = threads
        r["omp_num_threads"] = os.environ.get("OMP_NUM_THREADS", "")
        r["openblas_num_threads"] = os.environ.get("OPENBLAS_NUM_THREADS", "")
    out_csv = f"{args.output_prefix}.csv"
    write_rows(out_csv, rows, fields=list(rows[0]))
    print("\nQuote with the thread count and the machine's load line.")
    print(f"wrote {os.path.basename(out_csv)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
