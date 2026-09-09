# LTpred time and memory review, 9 September 2026

A subsequent [clean v0.6.1 rerun](../2026-09-09-time-memory-v061-rerun/README.md)
repeats all seven workloads and confirms the main runtime and memory gains.
The measurements below retain their original development-source provenance.

The changes remove observation-mask sorting overhead and duplicate arrays in
PA, and temporary sibling sets in parent-graph construction. All measured
means, posterior variances, and register diagnostics agree exactly with the
baseline; both graph sizes have identical adjacency digests. Table 1 reports
time and Table 2 separates process memory from allocations during a call.

This is a development comparison against commit
`52dec5294c101d4730c86d3307091be75dc47a5e` (v0.6.0), using source snapshots and
file hashes recorded in [results.json](results.json). The measured candidate
still declared v0.6.0; the v0.6.1 release changes only that version string in
`ltpred/__init__.py` among the recorded package sources. All numerical source
files retain their measured hashes. It is a synthetic
single-thread experiment, not a production register or a new statistical
calibration study. Each PA workload contains 200,000 families with 17
coordinates and a shared positive-definite covariance; `pa_mixed` assigns each
non-target coordinate a 3% chance of a pin. These numerical PA workloads use
a general covariance, not a simulated population of related individuals.
The graph workloads have half founders and two children per parent pair.
The register workload uses the public scoring API, a shared paternal chain of
depth 60, and degree-one observation selection.

**Table 1. Runtime in seconds, baseline / candidate.** First calls include any
JIT compilation reached by the call. Warm time is the median of five further
calls in the same process; input generation, imports and garbage collection
are outside these call timings. Each arm has its own initially empty Numba
cache. Source order alternates between cases.

| Workload | First call (s) | Warm median (s) | Warm speedup |
|---|---:|---:|---:|
| Parent graph, 200,000 records | 0.403 / 0.203 | 0.406 / 0.201 | 2.02x |
| Parent graph, 1,000,000 records | 2.016 / 1.157 | 2.024 / 1.170 | 1.73x |
| PA, mixed pin masks | 2.523 / 1.838 | 1.175 / 0.517 | 2.27x |
| PA, common pin mask | 1.811 / 1.728 | 0.453 / 0.380 | 1.19x |
| PA, intervals only | 1.801 / 1.750 | 0.409 / 0.383 | 1.07x |
| PA, censoring mixture | 2.318 / 2.277 | 0.721 / 0.669 | 1.08x |
| Register scoring, 300 probands | 1.088 / 1.091 | 0.092 / 0.093 | 0.99x |

**Table 2. Peak memory in MiB, baseline / candidate.** RSS is the peak of an
isolated timing process, including inputs, imports and JIT work, measured by
`_peak_launcher.py`. Allocations are the separate warmed-call `tracemalloc`
peak: inputs/imports are excluded; Python and NumPy allocations are included,
but this is not a complete measure of native/JIT workspace. Tracing and output
serialization run in separate processes and do not inflate the reported RSS
or runtime. MiB means 2^20 bytes.

| Workload | Peak process RSS (MiB) | Peak call allocations (MiB) |
|---|---:|---:|
| Parent graph, 200,000 records | 306.7 / 234.3 | 109.5 / 68.3 |
| Parent graph, 1,000,000 records | 899.9 / 689.0 | 536.6 / 330.2 |
| PA, mixed pin masks | 505.1 / 326.3 | 179.2 / 50.2 |
| PA, common pin mask | 521.5 / 369.6 | 237.1 / 92.1 |
| PA, intervals only | 339.2 / 269.9 | 77.8 / 12.8 |
| PA, censoring mixture | 475.4 / 359.0 | 132.8 / 42.0 |
| Register scoring, 300 probands | 160.3 / 160.4 | 3.1 / 3.1 |

The small register-scoring timings overlap (baseline 0.0907–0.0948 s,
candidate 0.0917–0.0928 s); no end-to-end speedup is established for that cell.
Its profile still spends most time on selected kinship, pedigree extraction,
and covariance construction/validation. No timing claim is made here for
Gibbs, quadrature, or variance-component fitting. Sibling adjacency lists still
require quadratic space within an unusually large full-sibling group; this
change removes temporary sets without changing the public graph representation.

Measurements used an Apple M2 Pro (10 CPU cores, 16 GB RAM), Python 3.10.20,
NumPy 2.2.6, SciPy 1.15.3, and Numba 0.67.0. Numba used one verified thread;
BLAS/OpenMP environment limits were one. Runtime BLAS thread counts were not
independently queried. The AC-power/Low-Power-Mode guard passed. Recorded
one-minute load averages ranged from 3.40 to 4.73: these are repeated timings
on this machine, not confidence intervals from independent machine sessions.

Validation: the unchanged baseline passed 788 tests; the candidate passed
800 tests in 63.69 s and 146 focused checks with JIT disabled (11
sampling-heavy tests deselected). Ruff, `scripts/check_evidence.py` and
`git diff --check` passed. New checks cover long observation masks, nonzero
targets, float32 centering, read-only inputs, and shuffled pedigree rows.
A separate seeded probe compared 4,000 families across 200 random covariances,
1–39 coordinates, mixed observation types, float32/float64 bounds, and C/Fortran
layouts: both output arrays were byte-identical to the baseline. Existing tests
cover singular pin support, tiny positive conditional variances, impossible
observations, prediction landmarks, and the locked R reference scores.

To rerun the measured cases from the repository root, choose a new output
directory (the script refuses to overwrite one):

```bash
.venv/bin/python benchmarks/bench_time_memory.py \
  --baseline-ref 52dec5294c101d4730c86d3307091be75dc47a5e \
  --output /tmp/ltpred-time-memory-rerun
```

The runner saves baseline/candidate source snapshots, inputs' construction
settings, output arrays, per-process JSON, file hashes and numerical comparisons.
No data downloads are required. Its measured SHA-256 is recorded in
`results.json`; that file's command and snapshot paths refer to the original
local run. Current numerical source hashes must match the recorded candidate to
reproduce this comparison; the release version-string difference is described
above. The final assertion checks that source did not change
during measurement. The retained JSON is a development capsule and does not
replace the historical benchmark campaign or its provenance wrapper.

To repeat the randomized comparison using the snapshots from a rerun:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1 .venv/bin/python \
  benchmarks/results/2026-09-09-time-memory/differential.py \
  /tmp/ltpred-time-memory-rerun/baseline /tmp/ltpred-differential-baseline.npy
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1 .venv/bin/python \
  benchmarks/results/2026-09-09-time-memory/differential.py \
  /tmp/ltpred-time-memory-rerun/candidate /tmp/ltpred-differential-candidate.npy
.venv/bin/python - <<'PYTHON'
import numpy as np
baseline = np.load('/tmp/ltpred-differential-baseline.npy')
candidate = np.load('/tmp/ltpred-differential-candidate.npy')
np.testing.assert_array_equal(baseline, candidate)
PYTHON
```
