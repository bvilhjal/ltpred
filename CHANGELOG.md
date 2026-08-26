# Changelog

All notable changes to ltpred are recorded here. This project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html); while the major
version is 0 the public API may still change between minor releases.

## Unreleased

### Fixed

- **Seeded simulations were not reproducible across machines.**
  `simulate_under_LTM_single` drew through `rng.multivariate_normal`, which
  factors the covariance with an SVD; an SVD has no canonical sign, so the
  factor -- and every family drawn from a given seed -- depended on the LAPACK
  build under NumPy. Draws now come from the (unique) Cholesky factor, with a
  sign-pinned eigendecomposition for the exactly singular `h2 = 1` case. The
  distribution is unchanged, but the *values* a given seed produces are not:
  re-running a seeded script gives different families than before this
  release.
- Every figure quoted in `docs/vignette.md` was wrong for anyone who ran the
  script: they were produced under the old, platform-dependent sampler. All of
  them are regenerated, and `tests/test_vignette_numbers.py` now fails if the
  page and `examples/vignette.py` ever separate again.
- The vignette illustrated Falconer's `h2 ~ 2 rho` with a parent-offspring
  tetrachoric on 800 families at `K = 0.05`, where the whole estimate rests on
  a handful of case-case pairs: it is centred correctly but has an
  across-cohort SD of 0.13 and comes out *negative* about one run in fifteen.
  The page now reports the standard error `tetrachoric` already returned, says
  plainly that the small cohort carries no information, and demonstrates
  convergence on 25,000 families (`2 rho = 0.494 +/- 0.046` against 0.5).
- The vignette's age-censored comparison ran at 800 families, where censoring
  leaves one to five observed proband cases -- so the own-status baseline it
  divides by, and the eff-N ratios built on it, were noise. That block now
  uses 20,000 families (99 observed cases) and prints the case count. Its
  age-term gain is `1.05x`, inside the `1.02-1.05x` of RESULTS section 10
  rather than merely "consistent with" it.
- `examples/vignette.py` hard-coded two of its own results in the prose it
  prints (a correlation and a case-rate ratio), so they contradicted the lines
  computed just above them. Both are interpolated now.
- The vignette's "Did it work?" snippet aligned `mu` with the per-row `status`
  column subset to role `o`, which lines up only if every family has an `o`
  row -- and the same page says role `o` is optional.
- `examples/vignette.py` imports `tetrachoric` from its owning module. The
  package exports a function under the same name as its module, so in a
  process that imported `ltpred.tetrachoric` first, `from ltpred import
  tetrachoric` yields the *module* and calling it raises `TypeError`. The
  underlying collision is **not** fixed here -- both spellings are documented
  (the function in the vignette and README, the module in `docs/api.md`), so
  resolving it is an API decision. `tests/test_public_api.py` pins it as a
  strict `xfail` that will start reporting once it is.
- Vignette and figure stated the family covariance as `Sigma = h2 A`. The
  liability covariance has a **unit diagonal** — `Sigma = h2 A + (1 - h2) I` —
  which is what keeps `Phi^-1(1 - K)` a prevalence threshold. `h2 A` alone is
  the covariance of the additive genetic values, not of the observed
  liabilities. Corrected in `docs/vignette.md`, `docs/assets/pipeline.svg` and
  `README.md`; a covariance is also no longer described as *being* the BLUP
  weights.
- Vignette offered the **sib** tetrachoric as an equal alternative to
  parent-offspring for Falconer's `h2 ~ 2 rho`. Full sibs also share the
  sibship kernel, so `2 rho_sib` estimates `h2 + 2 c2`. Only the
  parent-offspring route is now given.
- Vignette claimed all four published names are observation models "not
  engines"; PA-FGRS names its engine too, as `docs/guide.md` already said.
- Table 1 gave use I steps "0-4" while Table 2, section 5 and the example
  script all give it a step 5. Both uses now read 0-5.
- Table 1 told ADuLT readers to skip step 3, which builds their only
  observation. It now says: all of step 1, and the relatives' rows in step 3.
- Step 1's parent-pointer example ran `kinship_from_pedigree` before
  `extract_pedigree`, reused one set of variables for two different tables and
  discarded `ped`. Reordered so the data flows one way; the default
  `max_degree=2` is used, with depth guidance from RESULTS section 12.
- `use I` now keeps the role-`o` row with `(-inf, inf)` bounds rather than
  dropping it: `pids` is read off that record and silently falls back to
  `fam_id` when it is absent. Corrected in the vignette, `docs/guide.md` and
  `README.md`.
- Kinship-vs-role agreement was called "numerical noise". It is 5.6e-4, PA
  fold-order approximation error; the columns route is the exact one (0.0).
- `res.se` being 0 under PA now carries the no-*sampling*-error caveat the
  docstrings and README already used.
- `\operatorname` in `docs/vignette.md` — GitHub's MathJax rejects that macro
  and rendered the three correlations as an error string on the markdown
  source README links to. Now `\mathrm`. Inline math no longer sits inside
  bold, and section headings no longer contain math (it leaked raw TeX into
  the sidebar and the search index).
- Empty header rows in the vignette's Table 1 and "Where to go next" table;
  every table on the page now has real column labels.
- `report/README.md` wrote `h^2` with `\( \)` delimiters, which GitHub does
  not render.
- `docs/assumptions.md` checklist started at `0.`, which python-markdown
  renumbers to 1-13 on the site while GitHub shows 0-12. The use-selection
  item is now a sentence above the list.
- `docs/cip-estimation.md` defined the CIP as "before a given age" against its
  own formula and the vignette ("at or before").
- mkdocs nav called `inference.md` "Fitting the model" while the page and
  every link say "Inference"; `docs/guide.md` was titled "ltpred user guide"
  while four places called it "Choose a method".
- `docs/REVIEW_2026-08.md` and `docs/RELEASING.md` were published, sitemapped
  and searchable but absent from the nav. Both are now in the nav, and the
  review carries a banner saying it audits v0.3.4 and predates the 0.4.0
  lean-down.
- The three uses are numbered I/II/III everywhere (`docs/guide.md`,
  `README.md`, `PAPER_PLAN.md` previously used 1/2/3 while referring to them
  by roman numeral).
- Pages that already wrote `h²` as a code span no longer also carry `$h^2$`
  math on the same page.
- The covariance correction above is stated for the general case:
  `Sigma = h2 A + c2 C + m2 M + e2 I` with `e2 = 1 - h2 - c2 - m2`. The
  simpler `h2 A + (1 - h2) I` holds only when the shared-environment
  components are zero, and the same page offers `c2`/`m2` two sections later.
- Pedigree-depth guidance said the extended pedigree cost "less than one
  standard error at every prevalence tested". True at K = 0.05 and K = 0.20;
  at K = 0.01 the extended arm is about 2 SE *lower* (0.249 +/- 0.004 against
  0.259 +/- 0.003). Stated as measured.
- The "Did it work?" checks indexed `mu` with the page's per-row `status`
  column; they need the per-proband subset, and the proband count is
  `len(families)`.

- Methods note stress-pedigree PA–Gibbs floor is 0.9991 (CSV minimum 0.99916),
  not 0.9992. `scripts/check_evidence.py` now recomputes the floor from
  `bench_pa_robustness.csv`.
- `fit_heritability` and `fit_variance_components` reject a `pid` that appears
  in more than one family (or twice in one family). Overlapping register
  pedigrees remain valid for `estimate_liability`. Bootstrap copies of the
  same `fam_id` are not treated as overlap. Members without a `pid` cannot be
  checked.
- A family with no members now raises rather than warning and returning the
  prior mean 0 (a silent-looking GWAS phenotype if the warning was ignored).
- `research/README.md` no longer claims its tests run in CI.

### Added

- A user vignette (`docs/vignette.md`, runnable as `examples/vignette.py`)
  on how to run ltpred. It distinguishes three uses — family-history
  risk prediction (optional PGS), GWAS-signal enhancement, and
  aetiology from h²/r_g and/or CIP — then the steps: h²/covariances,
  pedigree, CIP, family history, `estimate_liability`. A pipeline figure
  (`docs/assets/pipeline.svg`) shows how those inputs join and where
  each use can stop. The page is part of the MkDocs site (`/vignette/`);
  KaTeX renders the equations. The docs site uses the SMARTbiomed /
  Aarhus University teal–navy palette from
  [smartbiomed.dk](https://smartbiomed.dk/).

- Vignette: a **"Did it work?"** section with the law-of-total-variance check
  (`Var(mu) + mean posterior var == h2`) and what a failure means; a **use-I
  risk conversion** from the liability scale to `P(case)`, with its
  under-prediction in the top decile stated; a **"How much it buys you"**
  section quoting the effective-N proxy and its collapse in case-enriched
  cohorts; **sizing guidance** (families/s for both engines, when to switch to
  the array path); a **migration table for LTFHPlus / LTFGRS** users; the
  shape of a real input table; ragged-family handling; multi-stratum CIP
  stitching; and the GWAS consequence of an unstratified `K`.
- Vignette Table 3 replaces the run-on paragraph of correlations, and the
  simulated cohort is now defined before its numbers are quoted.
- `examples/vignette.py` prints and asserts the sanity checks, and prints the
  risk-calibration and age-aware-cohort numbers the vignette quotes.
- `CITATION.cff` gained the five method foundations the vignette cites:
  Aitken 1935, Mendell & Elston 1974, Falconer 1965, Henderson 1975 and
  Lee et al. 2011.
- `tests/test_vignette_numbers.py`: runs `examples/vignette.py` and checks
  every figure quoted in `docs/vignette.md` against it, at the page's own
  rounding. CI also runs the example on every matrix leg, macOS included --
  the platform coverage is the point, since a platform-dependent sampler is
  what let the figures drift.
- `ltpred.simulate._stable_factor`, with regression tests pinning the draws to
  the Cholesky factor so a refactor cannot quietly restore the
  platform-dependent path.

### Changed

- `docs/assets/pipeline.svg` redrawn: body type raised from 11-13 to 17-22
  units in a shorter viewBox (smallest label goes from ~8px to ~14px on a
  desktop content column), every foreground/background pair now meets WCAG AA
  (the old `#7a8a99` glosses were 3.55:1 and the teal panel label 4.07:1), a
  `prefers-color-scheme` dark palette replaces the baked-in light slab, the
  Lee dependency finally has an arrowhead, and the figure is wrapped in a link
  so a phone reader can open it full size. Detail that duplicated Table 2 was
  removed rather than shrunk.

- README, RESULTS.md, and the headline tables state that the genotype-GWAS
  NCP ratios are independent-SNP results.
- The methods note, user guide, algorithm page, estimation /
  inference / CIP / data-preparation pages, assumptions checklist,
  README, and vignette now use the same three uses of the
  liability-threshold core: (I) family-history risk prediction,
  (II) GWAS-signal enhancement, (III) architecture / relationships /
  aetiology from h²/r_g and/or CIP.
- The vignette expands CIP, LT-FH, LT-FH++, ADuLT, PA-FGRS, PGS,
  GWAS, PA, $r_g$, IPW and related abbreviations at first use, and
  cites the method papers (Hujoel, Pedersen, Dybdahl Krebs, Lee,
  Aitken, Mendell–Elston, Zhuang).
- README and the vignette each stated the own-status-in-versus-out choice
  three times in adjacent paragraphs; each now states it once.

## 0.4.2 — 2026-08-21

### Fixed

- Preserve the single-trait scalar-bound diagnostic when the PA censoring
  mixture is enabled, and correct the documented PA-versus-LTFHPlus RMSE to
  0.0046. The evidence gate now derives and checks both R-lock RMSE values.

### Changed

- Restore lean prospective benchmark provenance: retained artifacts and
  external inputs are hashed in one JSONL row together with the clean source
  commit, command, runtime stack, thread settings, machine profile and run
  status. Historical artifacts remain explicitly unattributed.

## 0.4.1 — 2026-08-20

### Fixed

- Reconciled the v0.4 benchmark ledger, public documentation and tracked methods
  PDF with the committed CSV artifacts, and added a compact CI/release check for
  the release-defining scaling, IPW and public-R-lock claims.
- Hardened mixture, missing-family-ID and finite-input validation; made the PGS
  joint-score evaluation genuinely held out; and made the stochastic R lock
  reproducible. Each defect now has a focused regression test.

## 0.4.0 — 2026-08-20

### Removed

- Repository lean-down. Removed the comprehensive prose-number guard machinery
  (`scripts/check_results.py`, `scripts/check_docs.py`,
  `scripts/make_results.py`), the benchmark
  provenance capsules (`benchmarks/run_benchmark.py`, `run_manifest.jsonl`,
  `run_logs/`, `run_sources/`, `reference_env.json`), and the meta/hygiene
  tests that pinned them (`tests/test_check_results.py`,
  `tests/test_check_docs.py`, `tests/test_version_metadata.py`,
  `tests/test_run_benchmark.py`), plus `tests/test_pgs_fh_identities.py`,
  which tested no package code. Benchmark CSVs, RESULTS.md, `paper/`, and
  `report/` remain as static, historical documentation.
- Deleted the tracked landing page `index.html`; the mkdocs site remains the
  documentation face.
- `research/` is now dormant and unmaintained: removed from `testpaths` and
  from CI. The code stays in the repository for reference but may drift out
  of sync with the core package.
- Benchmark suite consolidated from 31 to 26 scripts; no evidence was lost —
  the absorbed scripts live on as panels of their merge targets, and their
  committed CSV/PNG artifacts stay in place:
  - `bench_aod_decay_robustness.py` → panel (d) of `bench_aod_decay.py`
    (`--robustness`, on by default).
  - `bench_age_onset.py` → panel (e) of `bench_fh_prediction.py`.
  - `bench_env_components.py` → panel (c) of `bench_shared_env.py`.
  - `bench_genetic_factor.py` → panel (c) of `bench_genetic_correlation.py`.
  - `bench_sex_limitation.py` + `bench_nurture.py` → the two panels of the
    new `bench_covariance_extensions.py` (CSV/PNG names unchanged).
- `bench_calibration.py` panel (a) is now PA-only; the Gibbs-agreement
  evidence lives in `bench_accuracy.py` and `bench_pa_robustness.py`. The
  `*_gibbs` CSV columns are still written but carry NaN in `correct` rows.
- `bench_ascertainment.py` default grid reduced to a moderate run
  (5 reps x 3,000 families x 800 iterations; was 10 x 10,000 x 1500, which
  took hours and was killed by the OS under memory pressure). The historical
  campaign settings remain available via the new `--full` flag.

### Changed

- Declared support matches what CI tests: added the 3.14 trove
  classifier, and replaced `Operating System :: OS Independent` with
  Linux and macOS, which are the platforms actually exercised.
- RESULTS.md: §15 Δcorr CI half-width is 0.00111 (was 0.00110), §20
  quotes the between-arm correlation as 0.8979 ± 0.0065 (0.898 ± 0.007
  double-rounded a 0.0065 SE up), and §17 no longer cites the ROADMAP
  for a ">20x" `h2_se` figure the ROADMAP no longer contains and that
  no artifact can reproduce — §17 is log-only. It now cites the
  `FitResult` caveat and the `>5x` bound locked in
  `test_bootstrap_fit_scalar_and_calibration`.
- README and the methods note now state that the code and
  documentation were written together with AI.
- README and the methods note now state that ltpred is much more
  computationally efficient than LTFHPlus and LTFGRS, quoting the
  current locked same-algorithm fold times (6.79× Gibbs, 1418× PA).
- Rewrote the method and algorithm documentation in Knuth style:
  numbered equations for the estimand and BLUP identity, and
  Algorithms G (Gibbs), P (Pearson–Aitken) and M (mixture) as
  named numbered steps in `docs/algorithm.md` and the methods note.
- **Breaking:** `estimate_liability_from_kinship` now returns
  `(est, se, var)` on both engines — `se` the Monte-Carlo SE (exactly
  zero under PA) and `var` the posterior variance — instead of a
  two-array `(est, uncertainty)` whose second element changed meaning
  with `method`.
- The Gibbs batch-means Monte-Carlo SE now divides by the draws that
  actually enter the batches (the R `batchmeans::bmmat` convention)
  rather than all retained draws; the difference is 0.07% at the
  default `n_sim = 100,000`.
- RESULTS.md corrections from the 2026-08 review
  (`docs/REVIEW_2026-08.md`): the stress-pedigree PA–Gibbs headline is
  0.9981 (worst single seed 0.99813; the 0.9984 figure quoted the first
  seed), the sex-CIP gap closure is 0.05092 (female shift −0.03691),
  the peak-RSS sentence names each method, and the §2 load-average and
  §24 M-component figures match their recorded artifacts.

### Added

- `tests/test_r_lock.py`: locked LTFHPlus 2.2.0 / LTFGRS 1.0.1 reference
  scores on a committed 48-family table, checked in pytest (no R
  needed at test time), so a numerical regression in the port fails the
  suite instead of waiting for an opt-in benchmark rerun.
- `tests/test_pgs_fh_identities.py`: Monte-Carlo verification of the
  PGS × family-history correlation and joint-R² identities quoted in
  `docs/algorithm.md`.
- CI: 3.10/3.11 and a free-threaded 3.14t leg in the test matrix, and a
  build job that builds the sdist/wheel, twine-checks them, installs
  the wheel and smoke-imports it outside the source tree.
- CI: the `docs` job now publishes to GitHub Pages with `mkdocs
  gh-deploy` on pushes to `main`. The strict build already ran on every
  push; only the publish step was missing, so the site had gone stale at
  v0.1.0 while the package reached 0.3.4. Pull requests still get the
  check without touching the live site.
- `docs/REVIEW_2026-08.md`: the 2026-08 independent review.
- Two guards pinning previously unguarded RESULTS.md prose: the §15
  sex-CIP adjusted Δcorr/ΔNCP increments and the §20 between-arm score
  correlation. Both were verified to fail against the pre-correction
  digits.
- Regression tests for every fix above, plus stronger oracles replacing
  four assertions that could not fail: the relatives-only predictor is
  now pinned to the closed form `E[g | l_m > t] = (h2/2) * phi(t)/(1-Phi(t))`
  (exact under PA for a single truncation), the IPW variance-components
  test asserts that uniform weights reproduce population sampling and
  that skewed weights change the answer, the Kaplan-Meier test compares
  against the textbook product-limit form, and the PA R-lock band is
  tightened from `rmse < 1e-3` to `< 1e-4` against a measured 2.53e-5.

### Fixed

- **CI had not run since the previous commit:** the `test` job's
  `runs-on: ${{ matrix.os }}` was absorbed into the end of a comment
  line while the matrix was being extended, so the job had no
  `runs-on` and GitHub rejected the whole workflow file. Every job —
  including the new build job and the R-lock leg added alongside it —
  was silently skipped.
- Threshold helpers no longer lose the far tail. `liability_threshold`,
  `convert_age_to_thresh`, `thresholds_from_cip`, `observed_to_liability_h2`
  and the PA mixture split computed `Phi^-1(1 - p)`; that subtraction
  discards the tail below ~1e-16 and returned `+inf` for `p <= 1.1e-16`,
  so a legal `pop_prev = 0.1, slope = 1.0` gave a case at age 20 the
  degenerate pin `(inf, inf)`. They now use the exact, cancellation-free
  identity `Phi^-1(1 - p) = -Phi^-1(p)`, matching `scipy.stats.norm.isf`
  to zero error down to `p = 1e-300` and leaving the ordinary range
  unchanged.
- `families_from_columns` length-checks the optional `pid`, `K_i`,
  `K_pop` and `aod` columns, which are indexed positionally: an
  over-long column was silently truncated (shifting mixture estimates)
  and a short one raised a bare `IndexError`.
- `families_from_columns` also rejects *textual* missing `fam_id`
  sentinels (`""`, `"NA"`, `"nan"`, `"None"`, `"null"`, `"."`,
  whitespace). These are worse than `NaN`: `NaN != NaN` fragments
  records into singletons, whereas every `""` compares equal and
  **merges** unrelated probands into one family.
- `pa_algorithm` and `pa_estimate_batched` validate the bounds shape
  against the covariance and the `target` index. The previous fix
  reached only the private role-array helpers, so the public entry
  points still dropped bound columns past `d` without complaint.
- `research.covariance_extensions`: a `sex` mapping whose `'g'` and
  `'o'` disagree is rejected rather than accepted — `g` is `o`'s
  genetic component, not a second person.
- `research.advanced_fitting`: `fit_genetic_correlation_decay` reshapes
  its `se["rg"]`/`["re"]`/`["rp"]` to `(P, P)`, matching the parent
  `GenCorrResult`; they were flat `(P*P,)` vectors, so `se["rg"][i, j]`
  raised. `fit_genetic_factor` validates only the off-diagonal of
  `weights`, so the documented `1 / se**2` DWLS recipe — `+inf` on the
  diagonal because the `r_g` diagonal is the constant 1 — is accepted
  and ignored, as `_minres_loadings` already intended.
- `families_from_columns` rejects missing (NaN/None) `fam_id` values
  instead of silently fragmenting records into one-member families, and
  validates that `lower`/`upper` shapes match.
- The array estimators (`estimate_liability_pa_arrays` /
  `estimate_liability_gibbs_arrays`) validate the column count against
  `roles`; extra columns were silently dropped and short inputs failed
  with a raw `IndexError`.
- `validate_bounds` rejects coincident infinite bounds
  (`lower == upper == ±inf`), which previously produced NaN Gibbs draws.
- A length-1 vector `h2` is treated as a scalar single-trait request
  instead of routing to the multi-trait error.
- `simulate_under_LTM_single` warns when `case_encoding="pin"` is
  combined with a non-threshold-crossing onset model (the pin records
  information the generating process does not contain).
- `aalen_johansen_cip` accepts integral float `event_type` arrays,
  matching `kaplan_meier_cip`.
- `extract_pedigree` rejects non-integer `max_degree` instead of
  silently truncating it.
- The CI oldest-deps job runs the kernel test selection instead of the
  full suite on the pure-Python fallback (the full suite took hours
  without the JIT).
- The research pipeline's stratified-CIP test now asserts that
  stratification changes the estimate; it previously could not fail.

## 0.3.4 — 2026-08-15

### Changed

- Gibbs estimation collapses coordinates that are untruncated in every
  family of a structure group (the genetic rows on the public path, and
  every multi-trait genetic coordinate). The sweep runs on the remaining
  truncated liabilities; genetic posterior means are the Gaussian
  conditional means given those draws. Same estimand, fewer coordinates
  per sweep, and a Rao–Blackwell Monte-Carlo SE. `rtmvnorm_gibbs` still
  draws the full chain.
- Reran the 4-thread scaling grid and the LTFHPlus / LTFGRS lock after
  that collapse. Intra-package PA / Gibbs object-path speed-up is now
  384–488×; LTFHPlus Gibbs / ltpred Gibbs is 6.87 ± 0.12×.
- Methods note and docs describe the collapsed sweep
  (`Var(E[g|y])+Var(g|y)`). The leftover Implementation timing
  paragraph now quotes the post-collapse array path
  (2.15–6.53 M fam/s; 13–30× vs the object path), not the pre-collapse
  203–492× / 1.2–3.8 M figures.

## 0.3.3 — 2026-08-15

### Fixed

- Estimation with `use_mixture=True` no longer raises when one role-structure
  group contains no censored control of its own (e.g. an all-case family set
  alongside families that do supply valid `K_i`/`K_pop`): the
  at-least-one-valid-pair gate now runs once per call, not again per group.
  Regression-tested; the public array estimator keeps its own gate (it has no
  global pre-check).
- Corrected the stale C/M-omission bias figures in `docs/algorithm.md` to the
  committed `bench_couple_env.csv` values (+0.047/+0.101/+0.156 and
  −0.005/+0.025/+0.034), and added `check_results` guards for both the
  RESULTS.md omission table and that prose so it cannot drift again.
- Labelled the PA-FGRS largest-shift "±" figures as 95% CI half-widths in
  README, `docs/algorithm.md`, RESULTS.md, `index.html` and the methods report
  (the RESULTS.md convention reserves a bare ± for SEs), updating the five
  anchored guards to the new wording.
- Fixed two `docs/algorithm.md` display details: the Gibbs conditional-SD
  contraction now states the zero-in-slot-`j` embedding of `P[:, j]` it
  presumes, and the PGS–family-history identity is the population `Corr`, not
  `E[Corr]`.
- Documented the kinship route's genetic-row scale explicitly: under
  inbreeding the prepended `g` is the target's genetic liability over its
  full-liability SD (variance `h2·A[t,t]/V[t,t]`), not the role-based
  `Var(g) = h2` scale.

### Added

- Multi-trait Gibbs oracle test: with exactly one bounded coordinate the
  posterior means match the closed-form truncated-normal conditional mean
  across traits, the multi-trait counterpart of the single-trait
  inverse-Mills oracles.
- **`benchmarks/bench_ltfhplus_compare.py`**: locked comparison to public R
  LTFHPlus 2.2.0 (Gibbs) and LTFGRS 1.0.1 (`method="PA"`) on the same
  families and bounds. LTFHPlus has no PA path. Isolated-process peak RSS
  uses a stdlib `wait4` launcher (the ldpred3 inherited-floor pattern).
  Wall-clock is a cohort total, milliseconds per family, and fold times
  versus LTFHPlus and LTFGRS (mean of per-replicate ratios). Opt-in
  (exits 2 if R or LTFHPlus is missing; LTFGRS is the PA arm).

## 0.3.2 — 2026-08-15

### Changed

- Strictly validate binary phenotype and event indicators before constructing
  thresholds, and encode rounded onset times as recording-bin bounds that
  contain the generating liability.
- Share one variance-component fitting engine between the heritability and
  multi-component APIs; validate fitter controls; and keep family weights
  aligned during bootstrap resampling.
- Describe the population/IPW case-rate gate as a marginal inconsistency
  screen, not as certification of joint-pattern positivity or weight validity.
- Validate Pearson--Aitken covariance matrices as positive semidefinite with
  strictly positive marginal variances.
- Generate the report's load-bearing tables from benchmark CSVs, retain source
  snapshots for future dirty benchmark runs, and remove redundant CI work,
  compatibility branches, and orphan benchmark logs.

## 0.3.1 — 2026-08-14

### Added

- **Inverse-probability weighting for selected samples**: `sampling="ipw"` with
  per-family `weights = 1 / P(family sampled)` on `fit_heritability` and
  `fit_variance_components`. The augmentation for a *given* family with *given*
  statuses is already the correct conditional distribution -- selection breaks
  the **mix** of families -- so re-mixing the per-family moment contributions is
  a better-matched remedy than a scale transform such as Lee et al., which
  operates on a different (observed-scale) estimand and cannot help here: at the
  same K, true h2 = 0.5 and h2 = 0 both produce 1.000, and no multiplicative
  factor is invertible across that.
  Benchmarked: a 50/50 case/control cohort goes from h2 = 1.000 (pinned) to
  0.495 against a truth of 0.5, and 20%-enriched from 1.000 to 0.473.
  Two limits, both reported rather than assumed. **Positivity** -- designs that
  sample no families from some complete status-pattern stratum (ascertainment
  through an affected proband) have inclusion probability zero there. The
  benchmark labels these designs undefined before calling the weighted fitter,
  because no finite valid weights exist.
  **Efficiency** -- weights reach 19x at K = 0.05 with a 50/50 cohort, so the
  benchmark reports the inflated across-replicate SD next to the bias.
- The case-rate check is weight-aware, using Kish's effective sample size
  `(sum w)^2 / sum w^2`. It can falsify weights whose role-wise marginals do not
  reproduce the asserted prevalence; passing does not establish joint-pattern
  positivity or weight correctness.
- `bench_ascertainment.py` arm G measures the IPW remedy through the real gate
  (the historical arms run under an explicit `unguarded()` context, since their
  purpose is to record what the number would have been without the check).
- **`sampling="population"` is now screened for gross marginal inconsistency.**
  It used to check a string, so an ascertained cohort fitted straight
  through to a fixed point at the clamp. The supplied thresholds already assert
  a prevalence, and under population sampling each role's case count is
  `Binomial(n_families, K)`, so `fit_heritability`,
  `fit_variance_components` and the `research/` moment fitters now run a
  per-role binomial check and raise on a gross mismatch. A mismatch is equally
  consistent with a mis-specified prevalence, which is a real and fixable
  cause, so the message reports the observed and asserted rates rather than
  assuming ascertainment.
  Calibrated from both sides: every ascertainment scheme in the new benchmark
  raises at z = +67 to +436 (N = 10,000), and a 12-cohort specificity sweep
  raised nothing.
  The bar (z >= 6) is set by `bootstrap_fit`, whose resamples centre on the
  cohort's rate rather than on K; a z >= 4 bar fired on a legitimate cohort.
  It catches catastrophic designs and does **not** certify population sampling:
  detectable enrichment is `1 + 6*sqrt((1-K)/(K*n))`, i.e. ~1.67x at N = 1,500
  and ~1.26x at N = 10,000 when K = 0.05.
- **`benchmarks/bench_ascertainment.py`** (RESULTS.md section 29): what the
  moment fitters return on selected samples, which no benchmark previously
  measured. Six arms over five ascertainment schemes plus a phenotype-
  independent negative control. Headline: at true h2 = 0 every
  phenotype-selected scheme returns h2 = 1.0 while the negative control returns
  0.025; the distortion is bias, not noise (constant +0.500 across
  N = 2,500-40,000) and not non-convergence (same estimate from h2_init 0.05
  and 0.95); and a realised case share only 1.27x the assumed prevalence
  already inflates h2 by +0.23.

- **`simulate_under_LTM_single(..., onset_model="liability_dependent")`**
  with `onset_rho` in `[0, 1]` (default 0.6): among lifetime cases,
  onset is coupled to liability by a Gaussian copula. `rho = 0`
  recovers `"stochastic"`; `rho = 1` recovers `"threshold_crossing"`
  among lifetime cases. This is the onset-timing assumption-stress
  used by `bench_pafgrs_mixture`.
- Technical report `report/ltpred_methods.pdf` (LaTeX source)
  `report/ltpred_methods.tex`): estimand, theory, implementation
  (grouping, streaming batch-means, object vs array path, `var` vs
  `se`), and the load-bearing simulation tables — PA–Gibbs agreement
  and speed, classic-LT-FH NCP, the LT-FH++/ADuLT increment, cohort
  confounding, calibration under a wrong `h²`, PGS complementarity,
  the PA-FGRS mixture, and ascertainment/IPW. Included in the source
  repository as versioned documentation.
- **Gibbs now reports the posterior variance**, so both engines fill
  `LiabilityResult.var` and the two are directly comparable. The batched kernel
  streams a sum of squares alongside the existing mean and batch-mean
  summaries, so the cost is one multiply-add per retained draw and the memory
  stays `O(ncols)` per family. `estimate_liability_gibbs_arrays(...,
  return_var=True)` exposes it on the array path. `var` (the spread of the
  posterior — it does not shrink with more draws) and `se` (the estimator's own
  error — it does) are now documented as the distinct quantities they are.
  This also gives PA's sequential-moment `var` an independent reference: a new
  test holds PA within 5% of the sampler on a seven-observation fold, the
  multi-truncation regime where `var` is actually used and where it was
  previously checked only for the single truncation PA is exact for.
- **Type annotations on the public API**, backing the `py.typed` marker the
  package already shipped. Every exported function now carries parameter and
  return types. `__init__.py` gained an `if TYPE_CHECKING` re-export block:
  the lazy PEP 562 `__getattr__` is a wildcard to a type checker, so without it
  *every* `from ltpred import X` resolved to `Any` and the annotations were
  invisible on the documented import path. `tests/test_public_api.py` pins the
  two export lists together.
- `simulate_under_LTM_single(onset_resolution=...)` makes the recording grid for
  a case's age of onset explicit (default `1.0`, whole years, as before;
  `None` records the exact simulated onset).
- CI jobs for the **pure-Python fallback** (installed without the `fast` extra —
  the path a plain `pip install ltpred` gets, previously never exercised because
  both matrix legs installed Numba), the **declared dependency floor**
  (`numpy==1.20.*`, `scipy==1.6.*`), and **`examples/*.py`**, which were run by
  neither the tests nor CI. The test matrix adds Python 3.13 (already claimed in
  the classifiers) and a macOS leg.

### Changed

- `bench_age_onset` now scores interval and pinned case encodings on the
  same families, so the pin-equals-liability increment can be separated
  from the onset lower bound. `bench_pafgrs_mixture` adds a
  liability-dependent onset arm. `bench_cip_estimation`,
  `bench_tetrachoric`, and `bench_liability_scale` write CSVs and were
  rerun under the current SE / `subtract_null` conventions. The methods
  note, `docs/algorithm.md`, `docs/assumptions.md`, and the README
  encoding guidance now follow those runs: pin only under
  threshold-crossing onset; most of the onset increment survives an
  interval encoding.
- **`simulate_under_LTM_single` applies the onset recording grid to
  `case_encoding="interval"` as well as `"pin"`.** Both encodings derive the
  case bound from the *recorded* onset age, so they now describe the same
  observation process; previously `pin` rounded to whole years and `interval`
  did not. The one-year default is unchanged for `pin`, so existing pinned
  artifacts are unaffected; interval-case simulations shift slightly. The
  docstring no longer claims a pin recovers the true liability outright — at the
  default grid it lands within ~0.02 of it (~2% of a liability SD), which is a
  floor under any oracle comparison built on the simulator; `onset_resolution=None`
  recovers it exactly.
- `kinship_from_pedigree` sorts the pedigree topologically with Kahn's
  algorithm instead of repeated scans. Any valid topological order yields the
  same `A` (pinned by a new record-order-invariance test), but the scan needed
  one pass per generation and degraded to `O(n^2)` when records were listed
  youngest-first — measured 1.25× total runtime on that ordering, the rest
  being the inherently `O(n^2)` dense-`A` fill.
- A family with **no members at all** now warns instead of silently returning
  the prior mean 0, which is indistinguishable in the output from a real
  estimate and in practice means a join dropped the member rows.

- **Pearson–Aitken `out="full"` is now the same estimand as Gibbs:**
  \(\mathbb{E}[l_o \mid\) own interval and relatives\(]\). The sweep applies the
  target's own bound after folding the other members (unbounded `g` is a
  no-op). A lone case therefore no longer returns PA `full = 0`. Omit `o` or
  unbind it for a relatives-only predictor. Object, array, and kinship PA
  paths all go through this update.
- **`simulate_under_LTM_single(use_age=True)` uses generation-consistent
  ages and `onset <= current age` as the observed-case rule.** A young
  high-liability person is a censored control, not a case pinned at a future
  onset. New arguments: `onset_model` (`"threshold_crossing"` default, or
  `"stochastic"` — lifetime status at \(T\), onset drawn from the CIP
  independently of \(l\)) and `case_encoding` (`"pin"` / `"interval"` /
  `"lifetime"`). Stochastic + lifetime/interval is the mode that does not
  pin cases at their true liability.
- **`estimate_liability_from_kinship` accepts the PA-FGRS mixture**
  (`use_mixture=True` with `K_i`/`K_pop`). Gibbs still rejects it.
- `probit_liability_r2` accepts allele frequency in `[0, 1]`, not only
  minor-allele frequency in `[0, 0.5]`.
- Single-trait object estimators are adapters over the array kernels
  (canonical sorted-role covariance, one PD-nudge/warning path).
- `docs/inference.md` matches the restored common-threshold guard on
  `fit_heritability` / `fit_variance_components` (personalised/onset-pinned
  bounds are rejected, not accepted).

### Fixed

- **IPW recovery figures** quoted in the methods note, `fit_heritability`
  docstring, `docs/inference.md` and `docs/ROADMAP.md`. The committed
  `bench_ascertainment.csv` `ipw` arm gives 0.495 (50/50) and 0.473
  (20% enriched). The previously quoted 0.456 is the `random_50`
  control arm, not the case/control cell; 0.481 is not in the CSV.
  `benchmarks/RESULTS.md` §29 already had the artifact values.
- §13 confounding table last-digit offsets against
  `bench_confounding.csv` (e.g. 16.461 ± 0.397 → 16.460 ± 0.399).
- **The moment fitters again reject personalised/onset-pinned LT-FH++ bounds.**
  `fit_heritability` and `fit_variance_components` pool cross-products under a
  single-threshold-per-trait assumption; on personalised (age-/CIP-specific) or
  onset-pinned bounds — even perfectly coherent ones — that moment design is
  not identified. Exploratory failures motivated the contract, but are not a
  current quantitative benchmark for personalised fitting. A guard rejecting
  those inputs had been removed and the docstrings rewritten to
  claim such bounds "are accepted"; the guard is restored (the docstrings now
  state the requirement honestly) and extended to the `research/` pooled-moment
  fitters `fit_variance_components_mcem` and `fit_genetic_correlation`. The guard
  tolerates the float32/float64 representations of one common threshold and lets
  NaN/reversed-bound errors surface first. `estimate_liability` is unaffected —
  it conditions on `h2` rather than fitting it, and personalised bounds are its
  intended input.
- `research.advanced_fitting.fit_nurture` no longer rejects strong negative-
  contrast models (`nurture < -0.5`, where parent-offspring covariance is
  negative) that `construct_covmat_nurture` emits as valid PSD models and the
  closed form inverts exactly; only the genuinely degenerate `nurture = -0.5`
  corner (`po = 0` / `ss = 0`) and covariances no standardised model can
  reproduce are refused.
- Negative `burn_in` is rejected everywhere it reaches a sampler:
  `rtmvnorm_gibbs`, every Gibbs estimate path (validated once in
  `_estimate_group`, the shared choke point), `fit_heritability`,
  `fit_variance_components`, and the research fitters. The sweep kernels loop
  `range(-burn_in, n_sim)`, so a negative value silently undercounted draws;
  boolean and non-integer burn-ins are rejected rather than truncated. The
  estimator choke point also validates `tol` (finite and > 0 — a NaN tolerance
  never converged yet suppressed the unconverged warning) and `max_rounds`
  (positive integer — a non-positive count returned all-zero estimates).
- `pa_algorithm` and `pa_estimate_batched` now validate their truncation
  bounds (no NaN, no reversed intervals — a reversed pair narrower than 1e-3
  was silently "sorted" by the narrow-interval quadrature) and their
  covariance (square, all-finite, symmetric within a scale-relative
  tolerance). This is deliberately lighter than the Gibbs gate: no strict
  positive-definite check, since the production paths route through
  `correct_positive_definite` upstream.
- Multi-trait estimation rejects duplicate `phen_names`; result dicts are
  keyed by (output, phenotype) name, so duplicates silently collapsed two
  traits' columns onto one key.
- A member or column with role `"g"` is rejected on the object and array
  estimator paths. The genetic-liability coordinate is added by the estimator,
  so a supplied `"g"` silently conditioned it on that row's bounds.
- `construct_covmat_single` / `construct_covmat_multi` reject duplicate roles
  in `fam_vec` (`["m", "m"]` built a singular matrix with two
  perfectly-correlated "mothers") and `n_fam` counts above 1 for singleton
  roles such as `m` or `mgm` (previously dropped silently; numbered roles like
  `s1`, `s2` are the way to request multiples).
- The simple threshold helpers (`liability_threshold`,
  `convert_age_to_cir`, `convert_age_to_thresh`, `convert_liability_to_aoo`,
  `prevalence_thresholds`, `age_thresholds`, `pa_thresholds`) now require
  prevalences in the open interval (0, 1): 0 or 1 produced infinite
  thresholds, and out-of-range values silent NaN ones.
- Docstring corrections: `estimate_liability` quotes the current benchmark
  figure (PA 203–492× faster at four threads on the no-mixture grid, was the
  superseded 315–510×); `ltpred.pedigree` states the proband's own mate is
  graph distance 2 (only a relative's mate is 3); `get_relatedness` documents
  the inherited LTFHPlus convention that two same-side half-sibs are related
  0.5·h² *to each other* (an implied shared second parent), and
  `construct_covmat_from_kinship` scopes its entry-for-entry agreement claim
  to pedigrees consistent with that convention.

- The research fitters `fit_variance_components_mcem`,
  `fit_genetic_correlation` and `fit_genetic_correlation_decay` add the core's
  `sampling=` contract gate: pass `sampling="population"` to affirm
  independent, non-overlapping, unascertained population-sampled families;
  omission warns, and any other value raises.
- `research.advanced_fitting._decay_cov` drops its dead `rp` parameter (the
  phenotypic correlation never entered the per-family covariance).

## 0.3.0 — 2026-07-31

### Changed (breaking)

Leanness refactor: the supported public API is now the lean estimation core.
The advanced fitting and covariance machinery moved to an **unsupported
`research/` package** at the repository root — importable as
`research.<module>` from a checkout, not installed with the distribution and
not covered by the compatibility policy. A capability (re)joins `ltpred`
proper only when wired into the core estimation path and benchmarked.

**Moved to `research/`** (same names unless noted):

- `ltpred.pipeline` (`PopulationScores`, `estimate_liabilities`) →
  `research/pipeline.py`.
- `fit_genetic_correlation`, `fit_genetic_correlation_decay`,
  `fit_genetic_factor`, `fit_nurture`, `test_variance_component`,
  `test_genetic_correlation` and their result types →
  `research/advanced_fitting.py`. The MCEM variance-component route is now
  `fit_variance_components_mcem` there (was
  `fit_variance_components(..., method="mcem")`, aliases `"ml"`/`"reml"`);
  its `MCEMVarCompResult` keeps the `loglik`/`aic` fields.
- `construct_covmat_sex_limited` and `construct_covmat_nurture` →
  `research/covariance_extensions.py`.

**Removed outright** (no replacement unless noted):

- `liability_sensitivity` / `SensitivityResult` — re-estimate over an `h2`
  grid and correlate the scores directly.
- `convert_observed_to_liability_scale` — an exact duplicate of
  `liability_scale.observed_to_liability_h2`; use that.
- `observed_to_liability_gencov`, `liability_to_observed_gencov` and
  `observed_to_liability_rg` from `ltpred.liability_scale`.
- `tnorm_moments` and `tnorm_mixture_conditional` from
  `ltpred.pearson_aitken` (internal helpers, no longer exported).
- `truncated_normal_cdf` from `ltpred.thresholds`; `convert_cir_to_age` is
  now private (`_convert_cir_to_age`).
- `extract_pedigrees` — a trivial generator; iterate `extract_pedigree`.
- The `construct_covmat` dispatcher — call `construct_covmat_single` /
  `construct_covmat_multi` directly.
- `estimate_liability_single` / `estimate_liability_pa` /
  `estimate_liability_multi` are now private (`_estimate_liability_*`); use
  `estimate_liability` or the array APIs.

**Signature changes:**

- `fit_heritability` and `fit_variance_components` add a transitional
  `sampling=` contract. Pass `sampling="population"` to affirm independent,
  non-overlapping population-sampled families; omission warns, and no
  ascertained-sample mode is implemented.
- `fit_variance_components` no longer takes `method=` — the Haseman–Elston
  moment regression is the only core route (MCEM lives in `research/`).
- `liability_r2_from_z` now subtracts the unit expected null contribution,
  using `(z² - 1) / N` by default. Individual estimates may therefore be
  negative; aggregate before interpretation. Pass `subtract_null=False` only
  to reproduce the former raw `z² / N` second moment.
- `VarCompResult` drops the `loglik`/`aic` fields.
- `Covmat` drops the write-only `h2` dataclass field.
- `LiabilityResult` drops the `column` attribute.
- `estimate_liability`'s `out=` accepts only the exact strings `"genetic"` /
  `"full"` (singly or as a tuple).
- `aalen_johansen_cip` drops `n_boot`/`seed` — the closed-form Aalen (1978)
  variance is the only standard error.
- `gibbs_params` / `rtmvnorm_gibbs` now reject singular as well as indefinite
  covariance matrices, including when precomputed Gibbs parameters are passed.
- `tetrachoric_matrix` warns when independently fitted pairwise correlations
  do not form a positive-semidefinite matrix; pass `check_psd=False` only after
  making an explicit downstream handling choice.
- `convert_age_to_thresh` and `convert_liability_to_aoo` drop `dist=` (and
  the normal-branch min/max parameters) — they always use the logistic
  mapping.

### Fixed

- Aalen-Johansen pointwise uncertainty now uses the finite-risk-set,
  grouped-tie `cmprsk::cuminc` recurrence rather than a large-risk-set
  approximation. Event-coded zero follow-up and censoring as `cause=0` are
  rejected; `n_entered` counts records with positive follow-up.
- Moment fitting preserves the exact theoretical `A`, `C`, and `M` kernels
  instead of nudging singular-but-valid component matrices. Public update
  controls are validated so the assembled covariance retains a numerical
  positive residual floor.
- The genetic-nurture covariance derivation now includes the changed proband
  genetic row, and the PGS/family-history correlation identity states
  conditional independence as an assumption rather than inferring it from
  non-overlapping cohorts.
- Benchmark reports now distinguish historical outputs from current-tree
  validation, scope uncertainty claims to the evidence retained, and provide
  a provenance runner with hashed source state, console logs, and canonical
  artifacts.

### Added

- The genetic-nurture additions below landed in `research/`
  (`construct_covmat_nurture` in `research/covariance_extensions.py`,
  `fit_nurture` in `research/advanced_fitting.py`), not in the core package.
  `construct_covmat_nurture` separates a proband's **direct** genetic effect
  from parental **indirect** (genetic-nurture) effects, building the covariance
  from the path model rather than from kinship. Parent-offspring covariance
  becomes `h2/2 + n*h2` and sib-sib `h2/2 + 2*n*h2 + 2*n^2*h2` -- inflated by
  different amounts, which is what identifies `n` -- while `Cov(A_o, l_parent)`
  stays at `h2/2`, `Cov(A_o, l_o)` becomes `h2*(1+n)`, and
  `Cov(A_o, l_sibling)` becomes `h2/2 + n*h2`. That selective, directional
  pattern is what a single symmetric kinship-scaled matrix cannot express.
  `nurture = 0` reproduces `construct_covmat_single` exactly. Nuclear roles
  only; the closed form was verified against a 4,000,000-family Monte-Carlo
  simulation of the path model. Note that on sibling covariance alone nurture is
  indistinguishable from a sibship `C`; parent-offspring covariance is what
  separates them.
- `fit_nurture` makes the indirect coefficient a **fitted** quantity rather than
  a supplied one. The parent-offspring and sib-sib moments are two equations in
  two unknowns, so the ratio isolates the indirect path and the estimates are
  closed-form and exact: `1 + 2n = cov_sib / cov_parent_offspring`,
  `h2 = 2*cov_parent_offspring^2 / cov_sib`. It also reports what a
  nurture-blind additive model would claim from each relative type alone, whose
  disagreement is the diagnostic for an indirect path and is zero exactly when
  `n` is zero. A moment estimator: no standard errors, and inputs must be
  liability-scale (use `ltpred.tetrachoric` on binary data).

## 0.2.0 — 2026-07-29

### Added

- `construct_covmat_sex_limited` puts sex into the covariance rather than only
  the thresholds: sex-specific heritabilities and a cross-sex genetic
  correlation, `Cov(g_i, g_j) = 2*phi_ij * sqrt(h2_i * h2_j) *
  rg_cross^[sex_i != sex_j]`. Equal heritabilities with `rg_cross = 1`
  reproduce `construct_covmat_single` exactly, and the result is positive
  semi-definite for any `|rg_cross| <= 1`. This changes the relative weights
  carried by same- and opposite-sex relatives. Personalised thresholds already
  affect calibration and can also change ordering through the truncated means;
  the covariance model adds sex-specific genetic weighting. The parameters are
  inputs, not fitted;
  see the sex-limitation section of `docs/algorithm.md` for the identification
  requirements.

### Fixed

- The high-level multi-trait dispatcher now rejects nonzero `c2`/`m2` instead
  of silently dropping them; defining their cross-trait covariance remains
  future work.
- The covariance fitters again accept valid person-specific one-sided,
  two-sided, and pinned rectangles. Geometry-only filtering no longer blocks
  those observations or the `mcem`/`ml`/`reml` likelihood routes; callers remain
  responsible for supplying a coherent fitting observation model.
- The MkDocs site has a real root page, the sdist includes the benchmark helper
  imported by its shipped tests, and documentation fence validation now matches
  delimiter type and opening length.

## 0.1.0 — 2026-07-22

First tagged release. A from-scratch Python implementation of the
liability-threshold family of genetic-liability estimators, ported from the R
package [LTFHPlus](https://github.com/EmilMiP/LTFHPlus).

### Models and inference

- **LT-FH** (family history), **LT-FH++** (age-, birth-year- and sex-dependent
  prevalence) and **ADuLT** (the personalised construction without family
  history), from the same threshold/covariance machinery.
- Two inference engines: a Numba-JIT truncated-multivariate-normal **Gibbs**
  sampler with batch-means convergence, and a deterministic **Pearson–Aitken**
  sequential-moment approximation (the single-trait default). Across matched
  no-mixture bounds their posterior-mean estimates correlated ≥ 0.997, with PA
  315–510× faster in the controlled threaded benchmark.
- The PA-only **PA-FGRS** age-censored-control mixture.
- Single- and multi-trait `estimate_liability`, plus an array API
  (`estimate_liability_pa_arrays` / `_gibbs_arrays`) that skips `Family` objects
  for biobank-scale runs, with an optional float32 bounds path.

### Covariance, thresholds and data preparation

- Role-based covariance over the LTFHPlus relative grammar, and arbitrary
  pedigrees via `kinship_from_pedigree` / `estimate_liability_from_kinship`.
- Sibship (`C`) and couple (`M`) shared-environment components, accepted as
  `c2`/`m2` by the single-trait role/object and array estimators. The tagged
  high-level multi-trait dispatcher exposed these arguments but did not apply
  them; this is fixed under Unreleased above.
- Threshold builders (`prevalence_thresholds`, `age_thresholds`, `pa_thresholds`,
  `thresholds_from_cip`), CIP estimation from follow-up records
  (Kaplan–Meier and Aalen–Johansen, `ltpred.cip`), pedigree discovery from trio
  records (`ltpred.pedigree`), and an end-to-end register pipeline
  (`ltpred.pipeline`).

### Model fitting

- `fit_heritability`, `fit_variance_components` (A + C + M, by multiple
  Haseman–Elston or a Monte-Carlo EM-style likelihood fit),
  `fit_genetic_correlation`, `fit_genetic_correlation_decay` (onset-age-structured
  `r_g`), and `fit_genetic_factor` (a Genomic-SEM-style common-factor model).
- `bootstrap_fit` for family-cluster intervals, `liability_sensitivity` for `h²`
  sensitivity, and parametric-bootstrap significance tests.
- Liability-scale transformations (`ltpred.liability_scale`) and
  tetrachoric-correlation diagnostics (`ltpred.tetrachoric`).

### Notes for users

**The tagged 0.1.0 fitters enforce a common case/control threshold per trait.**
`fit_heritability`, `fit_variance_components`, and
`fit_genetic_correlation` reject person-specific, two-sided, or pinned bounds;
the guard also precedes the variance-component likelihood aliases. Version
0.2.0 briefly relaxed that restriction; version 0.3.1 restored it for the
pooled-moment fitters as an explicit estimating-contract limitation.

For the Haseman–Elston and genetic-correlation fits, reported `se` values are
**within-dataset Monte-Carlo diagnostics**, not sampling standard errors. The
MCEM variance-component fit instead reports an approximate OPG/BHHH
observed-information SE, still subject to Monte-Carlo, finite-iteration, and
model-correctness assumptions. Use `bootstrap_fit` when its independent-cluster
assumptions hold. The PA-FGRS censoring mixture and the inference machinery
(bootstrap intervals, MCEM SEs, parametric-bootstrap tests) have not been
calibration-benchmarked; see `docs/assumptions.md` and `benchmarks/RESULTS.md`
for what is and is not validated. The `fit_heritability` benchmark grid covers
prevalence 0.1 with parents + 2 sibs at 3,000 families; rarer disease or sparser
families are outside it and show noticeably more spread.

`h2` must lie in `(0, 1]`: a zero heritability makes the genetic liability
identically zero, leaving a covariance no positive-definite correction can
repair. This is now rejected at the API boundary rather than failing opaquely
further down.
