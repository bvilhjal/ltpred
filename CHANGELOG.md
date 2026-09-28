# Changelog

All notable changes to ltpred are recorded here. This project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html); while the major
version is 0 the public API may still change between minor releases.

Entries list user-visible behaviour: API changes, changed numbers, fixed defects
and measured performance. Documentation-only corrections, benchmark
housekeeping and the review findings behind each change are summarised, not
itemised; the full per-release notes up to v0.7.4 are in git history
(`git show dc5336c:CHANGELOG.md`), and review findings are indexed in
[`docs/REVIEWS.md`](docs/REVIEWS.md).

## Unreleased

### Changed

- Small PA batches run on serial kernels. Below 128 families per batch (or
  per observation-mask group), the Pearson–Aitken fold skips the
  parallel-kernel launch, whose fixed cost a small batch never repays; values
  are identical either way. This removes most of the per-chunk overhead of
  the chunked/streaming drivers on pin- and censor-rich bounds, where each
  chunk otherwise launched one parallel kernel per mask group.
- The Gibbs convergence loop computes each structure group's collapse
  invariants (BLUP map, kept-block conditional-regression factors) once
  instead of per round; `gibbs_estimate_batched` accepts a caller-owned cache
  dict for the same purpose. Draws and summaries are bit-identical.
- PA-FGRS mixture folds skip absent relatives, exactly as the no-mixture
  kernel already did: an absent row carries no bound and no mixture pair, so
  folding it was a no-op that injected last-ulp rounding into the posterior
  variance. A family with extra absent relatives now scores bit-identically
  to the same family with those rows deleted; other mixture scores are
  unchanged (variances move at most one ulp).
- Member pid normalization is cached on the instance (invalidated when
  `pid` is reassigned), so repeated validation of the same family objects —
  every bootstrap resample revisit — no longer re-normalizes every pid.
- The register driver reuses the extracted pedigree's member indices instead
  of rebuilding them per proband. Scores are bit-identical.

### Fixed

- `scripts/check_evidence.py` pins `bench_scaling.csv` by its SHA-256 instead
  of resolving a `v0.4.0` tag that was never pushed; the docs CI job had
  failed on `main` since v0.7.4.
- `docs/inference.md` again states the cost difference between the sampling
  and deterministic fitters (lost when the method guide was retired in
  v0.7.4). Speedup figures for v0.7.1 and v0.7.4 that no committed artifact
  backs are restated qualitatively below.
- A register-driver test now shows that `cip_by_stratum` routes each record
  to its own curve; before, only permutation invariance was tested.
- A family's Gibbs score no longer depends on the other families it is
  sampled with. The sampler integrates out each family's own unbounded
  coordinates (Algorithm G, step G2) instead of those unbounded in every
  family of a batch, so the chunked drivers reproduce the array functions bit
  for bit and later convergence rounds cannot change a family's chain.
  **Seeded Gibbs values change** for families whose unobserved relatives
  differ from others in their structure group (the draws target the same
  posterior, but stored seeded results for such cohorts -- regression
  snapshots, recorded draws -- no longer reproduce); cohorts where every
  family has the same observed roles are bit-identical.
- The PA-FGRS mixture path applies the same pin checks as ordinary PA. A pin
  on a coordinate that earlier pins already determine (for example one person
  recorded twice) crashed with an opaque `SystemError`; it now scores as the
  single pin does, and contradictory pins raise `ValueError`. Other mixture
  scores are bit-identical.
- The PA pin-compatibility checks absorb storage rounding at the bounds'
  stored precision. A coherent configuration whose pin and the bound it
  determines round to opposite sides at float32 (the batched APIs'
  memory-saving dtype) was rejected as a contradiction by both PA paths; it
  is now accepted, while pin-versus-pin contradictions keep the tight
  float64 tolerance.
- `fit_heritability` and `fit_variance_components` reject `n_iter - burn_in < 4`
  before sampling; the whole fit used to run first and then fail inside the
  batch-means SE.
- `observed_to_liability_h2`, `liability_to_observed_h2` and
  `liability_r2_from_z` raise on a NaN or infinite `pop_prev`/`prop_cases`
  instead of returning NaN, as `ltpred.thresholds` already did.
- `estimate_liability_quadrature_arrays` accepts a length-one `out` sequence
  (`("genetic",)`), like the other single-column array APIs.

### Added

- `bench_personalised_compare.py` locks the LT-FH++ features the classic
  R-package lock does not cover: age-CIP personalised bounds (pinned cases,
  censored controls) against LTFHPlus Gibbs, and the PA-FGRS censoring
  mixture against LTFGRS `useMixture=TRUE` (RESULTS section 34). Building it
  found an input-contract difference: LTFGRS consumes the censored control's
  passed `upper` and double-corrects an age-specific bound (corr 0.71), so
  the lock writes each package its own encoding. Censored-cohort fixtures
  extend `tests/test_r_lock.py`, so both locks run in CI without R.

### Removed

- `ltpred.gibbs.gibbs_advance_moment` and `_offset_seed`, used only by the
  checkout-only `research/` fitters, moved to `research/advanced_fitting.py`
  (outputs bit-identical). The private compatibility wrapper
  `fit._prepare_group_vc` and an unreachable branch in `fit_pairwise` are
  gone. `bench_calibration.csv` drops its ten all-NaN `*_gibbs` columns.
- The 19 committed benchmark figures (2.2 MB of PNG). Nothing linked them and
  the CSV beside each script is the evidence; scripts still draw the figures
  locally, `benchmarks/*.png` is gitignored, and `run_benchmark.py` accepts
  only CSV artifacts.

### Tests

- `test_review_regressions.py` is split into the module test files; the three
  benchmark test files are one `test_benchmarks.py`; `test_validation.py`
  merged into `test_validation_numbers.py`. Two duplicate tests are removed,
  and tests no longer pass Gibbs controls to the PA default (suite warnings
  12 → 1).

### Documentation

- Condensed this changelog and the benchmark ledger (`RESULTS.md` 1,904 →
  1,462 lines, no result changed); merged the six dated reviews into one
  [review ledger](docs/REVIEWS.md) of open, declined and resolved findings;
  merged `docs/PAPER_PLAN.md` into `docs/ROADMAP.md`, whose priorities now
  include the open review findings; dropped redundant files from the
  benchmark evidence capsules.
- Every core algorithm is stated step by step in its module docstring, with
  the methods report's step names: G (Gibbs), P (Pearson–Aitken with the
  pin reduction), M (censoring mixture), Q (quadrature), K (kinship),
  R (register driver), H (moment fit) and L (pairwise likelihood).
  `docs/algorithm.md` gains a numbered summary of each. Docstring-only
  changes; three docstrings that misstated the code are corrected (the
  batched Gibbs SE denominator, `liability_r2_from_z(subtract_null=False)`,
  and an unconditional claim that chunked Gibbs reproduces the array API).
- A second audit checked every public docstring, user guide, benchmark header
  and the methods report against the code. Parameters, result fields, errors
  and warnings that went undocumented are now stated (`fit_pairwise*`,
  `tetrachoric`, the h² bridges, the thresholds, `estimate_liability`,
  `families_from_columns`); the API pages parse NumPy-style sections; private
  helper names are gone from user-facing text. Corrected claims include the
  Gibbs seeding contract (Numba's thread RNG, not NumPy's global one), the
  chunked-Gibbs equivalence, `Covmat.matrix` in the research examples,
  benchmark arm counts, run commands and output names, and speedup ratios
  that no committed artifact backs. The methods report now states P1's fold
  order, per-family G2 collapse, the G4 far-tail approximation, G6 rounds,
  Algorithm Q's refinement and mode rules, and the case-rate screen's exact
  thresholds (PDF rebuilt).

## 0.7.4 — 2026-09-25

### Performance

All outputs are bit-identical.

- The register driver certifies each proband's kinship once (one `eigvalsh`
  per proband, down from three) and builds dense `A` directly from the parent
  graph's integer indices. The dense-versus-selected kinship rule compares
  requested pairs with `0.03·m²`, so deep pedigrees stop switching to dense
  too early.
- Family-free, non-inbred probands on the kinship route use the scalar ADuLT
  moments; single-family PA calls skip batch routing.
- The PA object path stacks each structure group's members in one pass
  instead of per-member scalar writes.
- `construct_covmat_multi` builds one shared-DNA table instead of re-parsing
  roles per phenotype pair. The fitters share one stacked member-bounds array
  across their guards, and `_probability_limits` is memoised.
- `estimate_liability_gibbs_chunked` builds each chunk's seeds from its
  global offset, restoring the bounded-memory promise at very large `F`.

### Fixed

- `estimate_liabilities` warns that `kinship_cache_size=0` disables
  ancestor-pair reuse (much slower, no real memory saving).
- The v0.4.0 array-API scaling figures are attributed to their tag and thread
  count rather than described as current; `check_evidence.py` pins them and
  integrity-checks every committed `results.json` capsule.

### Documentation

- Onboarding is consolidated into Getting started and the executable
  tutorial; the vignette's tested examples became Numerical checks
  (`examples/validation.py`). The web model reference is shortened, with full
  derivations in the methods PDF.

## 0.7.3 — 2026-09-24

### Changed

- Identified relatives now require a proband `o` row with its `pid`; for
  prediction keep that row with `(-inf, inf)` bounds. Pandas NA/NaT and
  shared missing markers are recognised in family and register inputs;
  unlisted zero parent markers are unknown. Duplicate probands and empty
  family lists are rejected.
- Register scoring and CIP estimators are in the curated public namespace.
  Results gain `to_dict` and optional-pandas `to_frame`. Non-default Gibbs
  controls passed to deterministic engines warn.
- `simulate_register_liabilities(method="mendelian")` draws exact pedigree
  innovations, including inbreeding, without a dense matrix. The `dense`
  default keeps its seeded draws.

### Performance

- Quadrature node moments are compiled; identical deterministic bound rows
  are scored once. String-pid validation is faster.

### Fixed

- `from ltpred import tetrachoric` returns the function in every import order
  (previously the submodule could shadow it). The module remains importable
  by its dotted name.

## 0.7.2 — 2026-09-23

### Changed

- Array and chunked estimators share their validation; the pairwise fitters
  share their cluster sampling-covariance code. Chunked scoring writes
  directly into the output arrays; Gibbs keeps variances only when
  `return_var=True`. Signatures and seeded draws are unchanged.

### Fixed

- Duplicate-person checks normalise `pid`s (`"sib"`, `" sib "` and `b"sib"`
  are one person), and scoring rejects one non-missing `pid` under two roles
  in a family (a duplicated sibling record used to count twice).
- `families_from_columns` rejects a `fam_id` column mixing strings and
  non-strings (NumPy merged `[1, "1"]`).
- Concurrent calls from Python threads no longer abort the process on Numba's
  `workqueue` threading layer (the macOS wheel's only layer).
- The numerical-check decile calibration uses a stable sort, so its figures
  no longer depend on platform.

### Performance

- `estimate_liabilities` validates each CIP curve once per call, not per
  proband; `estimate_liability_gibbs_batches` builds only the current batch's
  seeds.

## 0.7.1 — 2026-09-23

### Added

- Chunked array drivers (`ltpred.chunked`): `estimate_liability_pa_chunked`,
  `estimate_liability_gibbs_chunked` and the `*_batches` generators stream
  homogeneous family batches with a bounded working set. `h2` is required.
- Public simulation generators, so an installed user can simulate every
  supported route: `simulate_pedigree`, `pedigree_birth_times`,
  `simulate_register_liabilities` / `RegisterSimulation`,
  `simulate_followup_records` / `FollowupSimulation` and
  `simulate_under_LTM_multi` / `MultiTraitSimulation`. The benchmarks now call
  these, pinned bit for bit by a test.
- `docs/tutorial.md`, an executable simulated-register analysis checked
  block by block by `tests/test_tutorial.py`.

### Changed

- **Seeded simulations changed values.** `simulate_under_LTM_single` draws
  through the unique Cholesky factor instead of `rng.multivariate_normal`'s
  SVD, whose sign depends on the LAPACK build, so seeds now agree across
  machines. The distribution is unchanged; the families a seed produces are
  not. `h2 = 1` lifts the spectrum by ~1e-12 to keep a unique factor.
- The README quickstart simulates with `use_age=False`, matching the quoted
  figures.

### Performance

Outputs are unchanged. The speedups were measured during development; the
committed v0.7.1 time/memory capsule covers graph construction, PA and
register scoring, not these three changes.

- `kinship_from_pedigree` fills one vectorised row per individual.
- `estimate_liabilities` uses a dense matrix over the extracted pedigree when
  selected pairs are numerous.
- `simulate_under_LTM_single(use_age=True)` computes control thresholds once
  per role; the fitters' guards are vectorised.

### Removed

- `examples/joint_inference.py`, `examples/role_pipeline.py` and
  `examples/ltfh_power_demo.py`; the tutorial and quickstart cover them.

## 0.7.0 — 2026-09-16

### Added

- `fit_pairwise_multi`: opt-in joint liability-scale heritabilities, genetic
  and residual environmental correlations, with optional multivariate
  sibship/couple components, per-trait thresholds, missing phenotypes and
  positive family IPW. PSD fits from aggregated binary-pair counts, with
  conditional family-cluster SEs withheld at covariance boundaries.
- `benchmarks/bench_pairwise_multi.py` and `bench_pairwise_recovery.py`:
  recovery, SE calibration and coverage for the pairwise fitters.
- The R lock gains an LT-FH++ age-of-onset cohort under both case encodings
  (point pin and one-sided interval) against LTFHPlus 2.2.0 and LTFGRS 1.0.1.

### Changed

- **Breaking:** `h2` is required by `estimate_liability`,
  `estimate_liabilities`, `estimate_liability_pa_arrays`,
  `estimate_liability_gibbs_arrays`, `estimate_liability_from_kinship` and
  `estimate_liability_quadrature_arrays` (the `0.5` default was removed).
- Benchmark numbers have one prose home, `benchmarks/RESULTS.md`;
  `check_evidence.py` recomputes every cell of the static paper tables.

### Fixed

- Haseman–Elston and research fits reject covariance models unidentified by
  the jointly observed phenotypes.
- Quadrature no longer refuses families whose posterior mode is already
  stationary (49 → 0 refusals over 1,200 random families), and its refinement
  error reports the change the acceptance criterion tests.
- Benchmark scripts parse arguments (`--help` no longer overwrites a CSV),
  and precision panels no longer crash on zero spread.

### Removed

- The 2026-09-05 efficiency pilot and its 2.3 MB capsule; orphaned paper
  tables and artifacts; `research/pipeline.py`; dead helpers
  (`pearson_aitken._tnorm_mean` / `_tnorm_var`, `fit._prepare_group`).

## 0.6.2 — 2026-09-09

- Documentation only: the clean-source v0.6.1 versus v0.6.0 time/memory rerun
  is recorded and bound by the evidence check.

## 0.6.1 — 2026-09-09

### Performance

- PA groups observation masks by their byte representation and avoids
  identity-permutation copies; parent-graph construction builds ordered
  sibling lists directly. Outputs unchanged.

## 0.6.0 — 2026-09-05

### Added

- `method="quadrature"`: additive nuclear-family liabilities by integrating
  at most two parental factors, with refinement diagnostics
  (`quadrature_error`, `quadrature_nodes`). Excludes C/M, inbred or extended
  pedigrees, multiple traits, the censoring mixture and `h2 = 1`.
- `fit_pairwise`: opt-in A/C/M fits from common-threshold binary family data,
  with population/IPW sampling and independent-family sandwich SEs.

### Changed

- Ordinary PA marginalises absent observations and conditions all pins
  jointly before folding intervals. **This changes some pinned-family PA
  scores**; pin-only inputs and pins followed by one interval are now exact.
  The PA-FGRS mixture keeps its semantics.
- The register driver computes selected relationships from the complete
  ancestry graph with bounded reuse (`kinship_cache_size`).

## 0.5.2 — 2026-09-04

### Added

- Register-driver diagnostics: `PopulationScores.frac_records_with_unresolved_parents`
  (zero resolved parents raise, more than half warn) and, under
  `use="prediction"`, `PopulationScores.proband_state`
  (`disease_free_and_followed`, `prevalent_case`, `exited_before_index`).
  Scores are unchanged.

### Changed

- CI pins BLAS/OpenMP threads to 1 for deterministic R-lock comparisons.

## 0.5.1 — 2026-08-30

- Documentation only: the vignette and methods note follow the public
  register driver; the total-variance check is interpreted correctly.

## 0.5.0 — 2026-08-30

### Added

- A supported `ltpred.pipeline` (`PopulationScores`, `estimate_liabilities`):
  pinned-onset PA scoring from population trio records and empirical CIPs,
  for GWAS or prospective prediction.
- Arbitrary-pedigree scoring accepts shared-environment kernels
  (`c2`/`c_kernel`, `m2`/`m_kernel`), validated and never guessed from `A`.
- The user vignette, pipeline figure and MkDocs site theme.

### Changed

- **Breaking:** the six public threshold/conversion helpers require an
  explicit `pop_prev`.

### Fixed

- `extract_pedigree(max_degree=...)` marks kinship-only ancestors as
  `closure_only`, and the register driver keeps them uninformative by default
  (`condition_closure=True` opts in).
- Prospective register scoring censors each relative at their own age at a
  shared calendar `index_time`.
- `fit_heritability` and `fit_variance_components` reject a `pid` in more
  than one family; an empty family raises instead of returning the prior.
- `tetrachoric_table` rejects fractional and non-finite counts;
  `correct_positive_definite(correction_limit=0)` rejects instead of
  correcting once; `n_fam={"s1": 1}` keeps role `s1`.
- Near-symmetric kernels are canonicalised before inference.
- Documentation: the liability covariance is `h2 A + c2 C + m2 M + e2 I`
  (unit diagonal), not `h2 A`; Falconer's `h2 ≈ 2ρ` uses parent–offspring,
  not sib, pairs.

## 0.4.2 — 2026-08-21

- The PA censoring mixture keeps the scalar-bound diagnostic; benchmark
  provenance is hashed in one JSONL row per run.

## 0.4.1 — 2026-08-20

- Reconciled the benchmark ledger, docs and PDF with the committed CSVs and
  added `scripts/check_evidence.py`. Hardened mixture, missing-family-ID and
  finite-input validation; made the PGS joint score genuinely held out.

## 0.4.0 — 2026-08-20

### Changed

- **Breaking:** `estimate_liability_from_kinship` returns `(est, se, var)` on
  both engines.
- The Gibbs batch-means SE divides by the draws that enter the batches (R
  `batchmeans` convention; 0.07% at the default `n_sim`).
- Declared platforms are Linux and macOS, as tested.

### Added

- `tests/test_r_lock.py`: locked LTFHPlus 2.2.0 / LTFGRS 1.0.1 reference
  scores, checked without R.
- CI: Python 3.10/3.11/3.14t legs, a wheel build-and-import job and GitHub
  Pages deployment.

### Fixed

- Threshold helpers use `Φ⁻¹(1 − p) = −Φ⁻¹(p)` and no longer lose the far
  tail (`p ≤ 1.1e-16` gave `+inf`).
- `families_from_columns` length-checks optional columns and rejects textual
  and NaN `fam_id` sentinels (`""` merged unrelated probands).
- The public PA and array estimators validate bound shapes against the
  covariance, roles and target; coincident infinite bounds are rejected; a
  length-1 `h2` vector is a scalar.

### Removed

- The prose-number guard scripts, provenance capsules and their meta-tests;
  the benchmark suite went from 31 to 26 scripts without losing evidence.

## 0.3.4 — 2026-08-15

- Gibbs collapses coordinates untruncated in every family of a structure
  group (Rao–Blackwellised genetic means; same estimand, fewer coordinates).

## 0.3.3 — 2026-08-15

- `use_mixture=True` no longer raises when one structure group has no
  censored control of its own.
- Added `bench_ltfhplus_compare.py`, the locked comparison with R LTFHPlus
  and LTFGRS.

## 0.3.2 — 2026-08-15

- Strict validation of binary phenotypes and event indicators; rounded onset
  times are encoded as recording-bin bounds. The two variance-component APIs
  share one engine, and bootstrap resampling keeps family weights aligned.

## 0.3.1 — 2026-08-14

### Added

- `sampling="ipw"` with per-family weights on `fit_heritability` and
  `fit_variance_components`; `sampling="population"` runs a per-role binomial
  case-rate screen (z ≥ 6) against gross ascertainment.
- `bench_ascertainment.py`: what moment fitters return on selected samples.
- `simulate_under_LTM_single`: `onset_model` (`threshold_crossing`,
  `stochastic`, `liability_dependent`), `case_encoding` and
  `onset_resolution`.
- Gibbs reports the posterior variance (`LiabilityResult.var`).
- Public type annotations; the technical report; CI jobs for the
  pure-Python fallback, the dependency floor and the examples.

### Changed

- **PA `out="full"` is the same estimand as Gibbs**: a lone case no longer
  returns 0.
- `estimate_liability_from_kinship` accepts the PA-FGRS mixture.

### Fixed

- The moment fitters again reject personalised or onset-pinned bounds.
- Negative or non-integer `burn_in`, non-finite `tol` and non-positive
  `max_rounds` are rejected; PA validates bounds and covariance.
- Duplicate `phen_names`, a supplied role `g`, duplicate roles in `fam_vec`
  and out-of-range prevalences are rejected.

## 0.3.0 — 2026-07-31

### Changed (breaking)

- The supported API became the lean estimation core. Moved to the
  unsupported checkout-only `research/` package: the register pipeline (since
  re-added as `ltpred.pipeline`), `fit_genetic_correlation`,
  `fit_genetic_correlation_decay`, `fit_genetic_factor`, `fit_nurture`, the
  significance tests, MCEM variance components
  (`fit_variance_components_mcem`), and the sex-limited and nurture covariance
  builders.
- Removed: `liability_sensitivity`, `convert_observed_to_liability_scale`
  (use `observed_to_liability_h2`), the gencov/rg liability-scale converters,
  `tnorm_moments`, `tnorm_mixture_conditional`, `truncated_normal_cdf`,
  `extract_pedigrees`, the `construct_covmat` dispatcher, and the public
  `estimate_liability_single/_pa/_multi` (use `estimate_liability`).
- Signatures: fitters take `sampling=`; `fit_variance_components` drops
  `method=`; `liability_r2_from_z` subtracts the null contribution by
  default; `aalen_johansen_cip` drops `n_boot`/`seed`; `convert_age_to_thresh`
  and `convert_liability_to_aoo` drop `dist=`; result classes drop unused
  fields; singular Gibbs covariances are rejected; `tetrachoric_matrix` warns
  on non-PSD output.

### Fixed

- Aalen–Johansen variance uses the finite-risk-set `cmprsk::cuminc`
  recurrence. Moment fitting preserves exact A/C/M kernels.

## 0.2.0 — 2026-07-29

- Added sex-limited covariance (now in `research/`). The multi-trait
  dispatcher rejects nonzero `c2`/`m2` instead of dropping them.

## 0.1.0 — 2026-07-22

First tagged release: a Python implementation of the liability-threshold
genetic-liability estimators, ported from the R package
[LTFHPlus](https://github.com/EmilMiP/LTFHPlus).

- LT-FH, LT-FH++ and ADuLT from shared threshold/covariance machinery; the
  PA-only PA-FGRS censoring mixture.
- Gibbs (Numba JIT, batch-means convergence) and Pearson–Aitken engines;
  object and array APIs.
- Role-based and arbitrary-pedigree covariance, C/M shared environment,
  threshold builders, Kaplan–Meier and Aalen–Johansen CIPs, pedigree
  discovery.
- `fit_heritability`, `fit_variance_components`, `bootstrap_fit`,
  liability-scale transformations and tetrachoric diagnostics.
