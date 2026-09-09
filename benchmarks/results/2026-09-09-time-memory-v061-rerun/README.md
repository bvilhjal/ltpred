# LTpred v0.6.1 benchmark rerun, 9 September 2026

All seven time/memory cases completed successfully from a clean checkout of
`b516271266a9fa0d95f5137954e4a284d1913ccb` (v0.6.1), against
`52dec5294c101d4730c86d3307091be75dc47a5e` (v0.6.0).
The candidate commit, package files, benchmark driver and support scripts were
unchanged throughout measurement. This rerun uses the same inputs and settings
as the [development comparison](../2026-09-09-time-memory/README.md).

The main findings reproduce: mixed-mask PA is 2.25x faster, and constructing a
million-record parent graph is 1.73x faster. Peak memory also falls. The small
register-scoring case improves by about 3% in this run; it remains a much smaller
change than the large-batch gains, and its call-allocation peak is unchanged.

**Table 1. Runtime in seconds, v0.6.0 / v0.6.1.** First calls include reached
JIT compilation. Warm times are medians of five subsequent calls; imports,
input construction and garbage collection are outside the call timings. Each
source/case has a fresh process and initially empty Numba cache. Source order
alternates across cases.

| Workload | First call (s) | Warm median (s) | Warm speedup |
|---|---:|---:|---:|
| Parent graph, 200,000 records | 0.430 / 0.216 | 0.432 / 0.214 | 2.02x |
| Parent graph, 1,000,000 records | 2.091 / 1.216 | 2.101 / 1.213 | 1.73x |
| PA, mixed pin masks | 2.474 / 1.771 | 1.120 / 0.497 | 2.25x |
| PA, common pin mask | 1.684 / 1.609 | 0.429 / 0.366 | 1.17x |
| PA, intervals only | 1.645 / 1.629 | 0.393 / 0.370 | 1.06x |
| PA, censoring mixture | 2.169 / 2.118 | 0.692 / 0.643 | 1.08x |
| Register scoring, 300 probands | 1.065 / 0.993 | 0.087 / 0.085 | 1.03x |

**Table 2. Peak memory in MiB, v0.6.0 / v0.6.1.** Process RSS includes inputs,
imports, JIT work and all timing repetitions. Call allocations are measured
separately with `tracemalloc` after a kernel warmup; inputs/imports are excluded,
while Python and NumPy allocations are included. This second measure does not
capture every native/JIT workspace allocation. Tracing and output serialization
run outside the timing process. MiB means 2^20 bytes.

| Workload | Peak process RSS (MiB) | Peak call allocations (MiB) |
|---|---:|---:|
| Parent graph, 200,000 records | 290.5 / 255.2 | 109.5 / 68.3 |
| Parent graph, 1,000,000 records | 930.1 / 690.5 | 536.6 / 330.2 |
| PA, mixed pin masks | 491.9 / 337.3 | 179.2 / 50.2 |
| PA, common pin mask | 495.6 / 348.0 | 237.1 / 92.1 |
| PA, intervals only | 347.9 / 243.6 | 77.8 / 12.8 |
| PA, censoring mixture | 447.5 / 354.7 | 132.8 / 42.0 |
| Register scoring, 300 probands | 159.3 / 156.8 | 3.1 / 3.1 |

**Table 3. Warm speedup in the original comparison and this independent process
rerun.** These are repeated measurements on one machine, not independent
hardware replications or confidence intervals.

| Workload | Original speedup | Rerun speedup |
|---|---:|---:|
| Parent graph, 200,000 records | 2.02x | 2.02x |
| Parent graph, 1,000,000 records | 1.73x | 1.73x |
| PA, mixed pin masks | 2.27x | 2.25x |
| PA, common pin mask | 1.19x | 1.17x |
| PA, intervals only | 1.07x | 1.06x |
| PA, censoring mixture | 1.08x | 1.08x |
| Register scoring, 300 probands | 0.99x | 1.03x |

Every reported estimate and posterior variance matched exactly between versions
in all four 200,000-family PA cases. All register scores, variances, relative
counts, conditioning counts, closure counts and degree diagnostics matched
exactly. Both graph workloads produced identical adjacency digests. Input
settings and PA input hashes match both across versions and the original run.
All 28 worker records are present: seven cases, two versions, and separate
runtime/RSS and allocation processes. No case was skipped or failed.

The PA inputs use 17 coordinates with a shared general positive-definite
covariance; they are numerical workloads rather than a simulated population.
The mixed-mask case assigns each non-target coordinate a 3% chance of a pin.
The pedigree workloads have half founders and two children per parent pair.
The register case has 300 probands and a shared paternal chain of depth 60,
using the public API with degree-one observation selection. These synthetic
comparisons measure computational efficiency and numerical agreement; they do
not establish new statistical calibration, full-register scaling, or speedups
for Gibbs, quadrature, or variance-component fitting.

The runtime stack was Python 3.10.20, NumPy 2.2.6, SciPy 1.15.3, and Numba
0.67.0 on arm64 macOS with 10 logical CPUs. Each worker verified one Numba
thread. BLAS/OpenMP environment limits were one; runtime BLAS thread counts
were not independently queried. The AC-power/Low-Power-Mode guard passed,
and the same power state was confirmed after the run. Worker-start one-minute
load averages ranged from 2.46 to 3.29. The whole campaign took 109.46 seconds.
Individual warm repetitions and per-worker load readings are in
[results.json](results.json).

The [provenance record](provenance.json) captures the clean source commit and
file hashes before launch and after completion, exact command, timestamps,
exit status and results checksum. The [run log](run.log) records completion of
every case. Full source snapshots, output arrays, per-worker records and JIT
caches remain in `/private/tmp/ltpred-benchmarks-rerun-20260909/measurement`;
this compact capsule retains the aggregate raw measurements and provenance.
The existing benchmark driver and package were not edited for this rerun.

Reproduce from the v0.6.1 checkout with a new output directory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
.venv/bin/python benchmarks/bench_time_memory.py \
  --baseline-ref 52dec5294c101d4730c86d3307091be75dc47a5e \
  --reps 5 --output /tmp/ltpred-v061-rerun
```

The driver saves each measured source snapshot and rejects numerical
mismatches or package-source changes during the run. The additional provenance
record here also verifies the candidate Git revision and support-script hashes.
