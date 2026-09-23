# ltpred v0.7.1 time/memory run, 23 September 2026

All seven time/memory cases completed from `5b42b13` (v0.7.1) against
`ac78ba3` (v0.7.0 plus documentation). Every output matched exactly. No
workload moved by more than 2%: v0.7.1 speeds up paths this driver does not
measure (dense per-pedigree kinship when many relatives are selected,
`kinship_from_pedigree`, simulation, fit guards). Its register case selects
3–4 members per proband from a 60-deep shared chain, so it stays on the
memoized pair recursion. An earlier always-dense draft slowed that case from
0.078 s to 0.115 s, and this run confirms the per-proband switch in `5b42b13`
removed that slowdown. The gain on many-relative pedigrees is in
[RESULTS.md §21](../../RESULTS.md#21-end-to-end-register-pipeline-bench_register_pipelinepy)
(276 to 987 probands/s at degree 3).

**Table 1. Runtime and peak memory, v0.7.0 / v0.7.1.** Warm time is the median
of five calls after the first call; first calls include reached JIT compilation.
RSS covers the whole timing process; call allocations are `tracemalloc` peaks
in a separate warmed worker. MiB = 2^20 bytes.

| Workload | First call (s) | Warm median (s) | Warm speedup | Peak process RSS (MiB) | Peak call allocations (MiB) |
|---|---:|---:|---:|---:|---:|
| Parent graph, 200,000 records | 0.116 / 0.118 | 0.103 / 0.103 | 1.00x | 225.0 / 225.8 | 75.2 / 75.2 |
| Parent graph, 1,000,000 records | 0.637 / 0.623 | 0.636 / 0.641 | 0.99x | 703.9 / 699.5 | 364.5 / 364.5 |
| PA, mixed pin masks | 1.526 / 1.541 | 0.523 / 0.518 | 1.01x | 351.0 / 339.7 | 50.2 / 50.2 |
| PA, common pin mask | 1.381 / 1.406 | 0.383 / 0.385 | 0.99x | 377.0 / 382.7 | 92.1 / 92.1 |
| PA, intervals only | 1.390 / 1.355 | 0.385 / 0.389 | 0.99x | 279.4 / 272.9 | 16.0 / 16.0 |
| PA, censoring mixture | 1.861 / 1.893 | 0.672 / 0.669 | 1.01x | 407.3 / 445.0 | 64.7 / 64.7 |
| Register scoring, 300 probands | 0.822 / 0.793 | 0.084 / 0.082 | 1.02x | 172.6 / 171.0 | 3.6 / 3.6 |

**Environment.** Free-threaded CPython 3.14.6, NumPy 2.4.6, SciPy 1.18.0,
Numba 0.66.0 (`ltpred314`), one Numba thread, BLAS/OpenMP limits 1, arm64 macOS
with 10 logical CPUs, AC power with Low Power Mode off. The `.venv` that
measured v0.6.1 (§31) no longer imports SciPy on this macOS (the loader rejects
`_spropack`), so absolute times are not comparable with that capsule; both
arms here share one interpreter. Worker-start one-minute load was 2.8–4.1. A
same-day attempt under load 13–26 (a concurrent SMARTpred job) was discarded.
The campaign took 88 s.

[results.json](results.json) holds every repetition and per-worker load;
[provenance.json](provenance.json) the revisions, command and results checksum;
[run.log](run.log) the per-case completion lines.

Reproduce from the v0.7.1 checkout with a new output directory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
~/anaconda3/envs/ltpred314/bin/python benchmarks/bench_time_memory.py \
  --baseline-ref ac78ba36d2069e31c81de89a27c4fd472b579135 \
  --reps 5 --output /tmp/ltpred-v071-time-memory
```
