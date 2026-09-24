# LTpred review fixes — 24 September 2026

The review defects were checked against the live v0.7.2 checkout at
`db2377f0c49b8b6c93ab8ffe27745c5289365991`. Existing uncommitted changes to
`tests/test_estimate.py` and `benchmarks/README.md` were preserved.

## Matched measurements

Table 1 reports local warm microbenchmarks: three runtime calls after warming,
one separate allocation-traced call, and one fresh process per case/source.
AC power and Low Power Mode checks passed. BLAS/OpenMP/Numba were limited to one
thread. Python/NumPy/SciPy/Numba versions, all timings, traced allocations,
process peak RSS and package-source SHA-256 hashes are in the JSON files.
These are workload-specific results on a shared host, not host-isolated speed
rankings or a rerun of the package's full scientific benchmark suite.

**Table 1. Median warm runtime and traced peak allocations, before → after.**

| workload | time (ms) | ratio before/after | allocations (MiB) | maximum absolute output difference |
|---|---:|---:|---:|---:|
| 65,536 node moments | 97.62 → 2.44 | 39.99× | 12.411 → 1.000 | 0 |
| 64 quadrature families, four patterns | 543.14 → 6.73 | 80.66× | 0.978 → 0.437 | 0 |
| 64 quadrature families, unique bounds | 553.52 → 102.85 | 5.38× | 0.978 → 0.444 | 0 |
| 100,000 string pid keys | 46.79 → 10.07 | 4.65× | 1.443 → 1.443 | not recorded; normalization regression-tested |
| 400-person dense simulation (unchanged default) | 4.15 → 4.45 | 0.93× | 3.680 → 3.680 | 0 |

Quadrature comparisons include estimates, posterior variances, refinement
errors and node counts. The scalar moment function remains the numerical oracle;
regressions also check pins, tiny intervals, both tails, distant means and the
non-JIT implementation. Dense simulation keeps identical seeded outputs. The
small dense timing difference is not evidence of a meaningful performance change.

**Table 2. Opt-in Mendelian simulation on the same pedigree generator.**

| people | median time (ms) | traced peak allocations (MiB) | process peak RSS (MiB) |
|---|---:|---:|---:|
| 400 | 1.90 | 0.229 | 101.8 |
| 8,000 | 42.02 | 5.885 | 120.6 |

The 400-person dense comparator is in Table 1; there is no measured 8,000-person
dense comparator in this run. Traced allocations exclude some native work; RSS
includes the interpreter, imports and JIT. Mendelian draws differ from dense
draws at the same seed. Correctness is checked independently by reconstructing
the innovation factor and verifying its covariance against the full relationship
matrix, including shuffled rows, missing parents and inbreeding. Storage is
linear in people plus a bounded ancestor-pair cache; runtime need not be linear
for deep, highly related pedigrees.

`probe.py` is the rerun driver. `sources.tar.gz` contains the exact measured
`before/ltpred` and `after/ltpred` package sources (hash-verified). Extract it to a
temporary directory and pass either parent directory as `--source`; run with
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1`. `--case` selects the workload;
`--values` optionally saves numerical outputs for comparison. Later source edits clarify missing-parent docstrings/comments and use
diagnostic prefixes in score exports to avoid trait-name collisions. The
measured numerical and identity-normalization functions are unchanged.

## Review disposition

- Fixed identity fallback, pandas/text missing markers, duplicate probands,
  empty family lists, missing h² errors and prevalence-unit guidance. Zero is
  still a valid explicitly listed person ID; treating every zero as missing
  would break supported zero-based pedigrees.
- Added score table exports, distinct SE/variance fields on scoring containers,
  public register/CIP discovery and warnings for non-default Gibbs controls on
  deterministic paths. Explicitly passing a control's existing default cannot
  be distinguished from omitting it without changing the signature contract.
  Multi-trait reproducibility is documented as requiring an explicit seed.
- Compiled quadrature moments, deduplicated deterministic bound rows, shortened
  string-ID normalization and removed redundant within-family pid passes in
  single-trait fitters. Added opt-in bounded-memory Mendelian simulation.
- Fixed raw Sphinx roles and indentation on the rendered API page, the fitter
  choice guide, missing public-name entries, parameter descriptions, stale
  references, enrichment-floor formula and prediction example.
- Retained independent Gibbs streams: sharing one estimate would correlate
  Monte Carlo errors across families and change seeded outputs. A smaller first
  Gibbs round likewise needs convergence/parity evidence beyond a single
  low-dimensional timing probe; sampling defaults are unchanged.
- Retained register covariance checks: they validate relationship PSD, repaired
  liability covariance and PA input at different public boundaries. A trusted
  internal path may eventually avoid repeated work; merely removing these
  checks is not justified by the review's timing percentage.
- Unnumbered `s` is part of the documented inherited grammar, not a defect.
  Existing helper names/defaults remain compatible; their roles are explained
  instead of introducing wholesale renames. Fitter parameter uncertainty is
  not given an invented individual-posterior `.var` field.

## Validation

- Full package and research tests: **1,202 passed, 1 skipped**. The final export-name refinement was
  then checked by **25 focused regressions**, including real pandas NA/NaT and
  a trait name ending in `_se`.
- **25 final regressions passed with JIT disabled**; tail/narrow-interval and Mendelian covariance checks are included.
- Ruff, whitespace checks, `scripts/check_evidence.py`, and strict MkDocs build
  passed. Browser inspection confirmed the corrected estimator paragraphs; the
  rendered API contains zero raw Sphinx roles and covers all public exports.
- Built wheel/sdist and checked metadata; installed the wheel away from the
  checkout and exercised public exports, DataFrame export, quadrature and
  Mendelian simulation. No sampler defaults or scientific benchmark claims were
  changed, and this validation preceded the release commit. The release is v0.7.3;
  archived measurements retain the v0.7.2 working-tree stamp used at run time.

The repository's old `.venv` could not load its SciPy integration extension on
this host. Validation used a temporary environment inheriting the working
`ltpred314` environment, with pandas and packaging/documentation dependencies.
