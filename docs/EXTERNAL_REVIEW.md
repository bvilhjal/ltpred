# ltpred — independent review and future developments

Date: 2026-07-19
Reviewer: independent review (requested by the project owner), of the working
tree at commit `673bb99`. Method: full read of README, ROADMAP, algorithm.md and
the PA/Gibbs cores first-hand; the whole test suite run locally; a complete
static review of all 12 package modules and 11 test files; a complete read of
RESULTS.md plus all 16 benchmark scripts with every quoted number mechanically
re-derived from the stored CSVs. Two internal defects were independently
confirmed by direct inspection before inclusion.

---

## 1. Verdict

**ltpred is unusually careful scientific software and, unlike most research
ports, it is honest about what it is.** The Pearson–Aitken and Gibbs cores are
correct, tail-stable, deterministic under threading, and well tested (324/324
tests pass in 107 s). The documentation (README, algorithm.md, ROADMAP) is
accurate, quantitatively verifiable — every RESULTS.md number re-derives from
the committed CSVs — and candid about approximation error vs Monte-Carlo error,
identifiability limits, and unvalidated components. The defects found are real
but concentrated at the edges (one latent cross-trait bug, one extreme-tail
initialization bug, some missing input guards). The evidence gaps are equally
specific: the shipped PA-FGRS censoring mixture has no generative validation,
headline grids are partly single-seed, and the inference machinery (bootstrap
tests, intervals, MCEM SEs) has no calibration benchmarks. This is a project
one hardening pass away from being citable as *the* reference Python
implementation of LT-FH++.

## 2. Scope and quality of the theory documentation

`docs/algorithm.md` is the strongest document in the repo. It gives the
liability-threshold model, the family covariance, both inference engines with
their exactness boundaries (PA exact for one truncation; sequential-selection
approximation beyond), the BLUP/selection-index framing (thresholded nonlinear
generalization of breeding-value prediction), the Sham/So lineage, the
identifiability arithmetic for environmental components (rank of the pair
design ≤ number of distinct relationship contrasts), and why dominance and
vertical "maternal environment" are correctly *not* offered. The
threshold-personalization section is careful to state that age/sex/cohort enter
only through the CIP, never the covariance, and the pin-vs-interval case
encoding distinction (LT-FH++ pinned vs PA-FGRS lifetime interval) is
documented precisely — a distinction the literature regularly confuses.

Minor documentation issues:

- `benchmarks/README.md` says the benchmarks cover "LT-FH, LT-FH++, ADuLT and
  PA-FGRS inputs" — overstated: no benchmark passes `use_mixture=True`.
  RESULTS.md and ROADMAP correctly disclaim it; this one sentence should match.
- ROADMAP quotes "23×" for the `h2_se` gap (single bootstrap dataset) while the
  benchmark shows a 21–28× across-cohort range — different estimands, easily
  conflated.
- The role grammar inherits LTFHPlus's convention that two same-side half-sibs
  are related 0.5·h² *to each other* (covariance.py:173,185) — an implied shared
  second parent. Defensible, but undocumented in the Python docstring and
  untested.

## 3. Implementation review

Verified first-hand or by the full static review (file:line in the working
notes):

- **PA engine (correct, excellent numerics).** The rank-1 update matches the
  selection formula exactly; fold order is last-to-first and canonicalized to
  sorted roles so results don't depend on input order; pinned points are exact
  Schur-complement conditioning. Truncated-normal moments are computed with
  three-regime stability (log-domain `expm1`, Gauss-Legendre for narrow
  intervals and far tails to |z|>16, asymptotic `log Phi` below −37). The
  mixture splits at the lifetime threshold, with the right limits.
- **Gibbs engine (correct).** Precision-matrix conditionals (one O(d³)
  inverse), survival-scale interpolation for tail-safe truncated sampling,
  batch-means convergence checked on *every* requested estimate, per-family
  seed blocks that make results thread-scheduling-independent (tested at 1 vs N
  threads). The `gibbs_advance` fit path pre-draws uniforms outside `prange` —
  the right fix for Numba's thread-local RNG.
- **Covariance (faithful port + correct generalization).** The role grammar
  matches LTFHPlus's R source branch-for-branch (including fixing an R regex
  bug); `kinship_from_pedigree` is the textbook tabular method with inbreeding;
  `construct_covmat_from_kinship` reproduces the role grammar exactly where
  they overlap.
- **Fitters (sound).** HE regression correctly pooled; the genetic-correlation
  fit keeps `G` and `E` PSD and coherent (`rp = G + E` exactly); the factor
  MINRES is off-diagonal with the correct analytic gradient; `bootstrap_fit`
  resamples families; the MCEM M-step is the correct Gaussian likelihood of the
  imputed liabilities.

### Defects (prioritized; all edge-located)

1. **`covariance.py:360` — cross-trait bug with `add_ind=False`** (confirmed by
   direct inspection): the same-person cross-trait covariance is keyed on
   `a == 0` as a proxy for the `g` role, but with `add_ind=False` row 0 is a
   *relative*, so that relative's cross-trait covariance is set to the genetic
   covariance instead of `full_corrmat[p1,p2]`. Silent wrong model; untested
   path. Fix: key on the role label, add a test.
2. **`fit.py:134-139` — pinned-coordinate initialization saturates.** For a pin
   with |z| ≳ 8.2, `ndtr` rounds to 0/1, `norm_ppf` → ±inf, and the fallback
   pins the coordinate at 0.0 *permanently* (fixed coordinates are never
   resampled), conditioning the whole family on a wrong value. The estimate
   path's `_std_tnorm_quantile`-based init handles this correctly; reuse it.
3. **No PSD/symmetry guard on the public Gibbs sampler** (`gibbs_params`,
   `rtmvnorm_gibbs`): a non-PD but invertible covmat silently yields NaN
   conditionals. `correct_positive_definite` exists but is never invoked there.
4. **Missing input guards:** `fit_heritability` accepts `burn_in ≥ n_iter`
   (→ opaque `ZeroDivisionError`) and invalid `h2_init`; public `batch_means`
   has no n≥4 guard; the single-trait Gibbs path silently truncates vector
   (multi-trait) member bounds to the first column where the PA path raises;
   `validate_mixture_inputs` doesn't reject K's on pinned rows, which can
   invert a mixture interval → NaN.
5. **Inconsistent PD-correction reporting:** only the multi-trait estimator
   warns when `correct_positive_definite` fires; the single-trait, PA, array
   and kinship paths discard the count silently.

### Testing

Strong where it matters (algebraic identities, tail regimes, thread
independence, role/kinship equivalence, recovery bands). Gaps: the
`add_ind=False` multi-trait path, half-sib-pair relatedness, PA-vs-Gibbs
agreement under *pinned* LT-FH++ bounds (PA's weakest regime), the mixture
weight against a paper reference number, and the two real bugs above.

## 4. Benchmark and evidence review

**Every headline number in RESULTS.md re-derives exactly from the committed
CSVs** (checked mechanically): PA–Gibbs corr ≥ 0.997 (≥ 0.998 on stress
pedigrees), 315–510× speed ratio, the 1.47 ± 0.04× replicated LT-FH NCP ratio,
the integrated panel (ADuLT 1.049 vs LT-FH++ 1.194, paired increment
+0.1454 ± 0.0154, calibration slope 0.995), the sex-CIP gap removal
(0.05091 ± 0.00096), and the fitter bias/SD tables. Estimands are honest: the
ADuLT arm reuses identical proband bounds (a clean matched contrast), the
NCP-ratio vs squared-correlation-proxy distinction is enforced, and the
covariate adjustment residualizes both genotypes and phenotype (FWL) — the
right way. Weaknesses, all disclosed somewhere in the repo but collected here:

1. **The PA-FGRS censoring mixture has zero generative validation** — the
   single unit guard is one seed with loose tolerances, and its own comment
   records calibration slope ≈ 0.84 for the interval-case encoding. This is the
   repo's largest evidence gap: the mixture is the piece real registry users
   (censored controls) will actually switch on.
2. **Headline grids are single-seed or thin:** the 27-cell accuracy grid and
   calibration grids are single-seed; the GWAS headlines are 3-replicate (SEs
   properly derived but tail-fragile, CI half-width ≈ 4.3× the SE).
3. **The 315–510× speed claim** is measured against a cheaper-than-default
   Gibbs (tol=0.03, n_sim=25k vs default tol=0.01, n_sim=100k) — directionally
   conservative for the claim, but not the default configuration users get.
4. **No calibration evidence for the inference machinery:** parametric-
   bootstrap component tests, `bootstrap_fit` intervals, MCEM OPG SEs, and the
   factor-model `srmr` have no Type-I/coverage benchmarks.
5. **No model-misspecification robustness:** every generative model is exactly
   the Gaussian LTM the estimators assume — no non-Gaussian liability, G×E, or
   generative assortative mating stress.
6. Stored `bench_shared_env.csv` predates the MCSE instrumentation (disclosed).

## 5. Future developments — suggested, prioritized

**Tier 1 — correctness and trust (days, do first).**
1. Fix the five defects in §3 (each is small and well-specified; add the
   regression tests alongside).
2. Write `bench_pafgrs_mixture.py`: a generative validation of the censoring
   mixture — simulate age-censored controls and lifetime cases under a known
   CIP, run PA with `use_mixture=True`, and measure calibration slope, bias,
   and correlation vs the mixture-free and Gibbs references, including the
   interval-case encoding's known ≈0.84 slope as a documented contrast. This
   converts the repo's largest honesty caveat into a result.
3. Fix the `benchmarks/README.md` PA-FGRS sentence to match reality.

**Tier 2 — evidence hardening (days).**
4. Calibration benchmarks for the inference machinery: Type-I of
   `test_variance_component`/`test_genetic_correlation` under the null,
   interval coverage of `bootstrap_fit`, and stability of the MCEM OPG SEs —
   the question a skeptical reviewer asks first ("do your p-values calibrate?").
5. Multi-seed the flagship grids (accuracy, calibration) and raise the GWAS
   headlines to ≥10 replicates, or label the CIs with the t-multiplier.
6. A model-misspecification benchmark (non-Gaussian liability tails, a
   generative assortative-mating arm) — the analogue of PLDSC's
   `misspecified_background.py`, and the only way to learn where the LTM
   assumption actually bends.

**Tier 3 — capability (the roadmap's own items, re-ordered by value).**
7. **Wire fitted `C`/`M` back into `estimate_liability`** (algorithm.md's
   "sharper genetic estimate" promise): the fitter already estimates the
   components; feeding them into the covariance used by the estimator is
   mostly plumbing plus one benchmark showing calibration improves on
   environmentally clustered families. Highest user-facing value per effort.
8. **PyPI release** — the CI/docs are in place; a release makes the package
   citable as infrastructure (the CITATION.cff is already curated).
9. **Multi-trait PA** (roadmap #12): a genuine research question (sequential
   moment approximation across trait blocks); gate it with a PA-vs-Gibbs
   accuracy benchmark before shipping.
10. Scale: chunked/streaming driver (#9) before sparse single-pedigree (#8);
    the end-to-end real-LD HAPNEST GWAS example (#10) is the right final
    demonstration.

## Appendix — verification performed

- Read first-hand: README, ROADMAP, algorithm.md, `pearson_aitken.py`,
  `gibbs.py` (core), `covariance.py` (bug confirmation at line 360).
- Ran the full test suite in the project conda env (`ltpred`, Python 3.13):
  **324 passed** (107 s).
- Delegated static review: all 12 package modules + 11 test files (code
  correctness, API guards, testing gaps); RESULTS.md + all 16 benchmark
  scripts + stored CSVs (every quoted figure re-derived, none mismatched).
- Not done: running the benchmark scripts end-to-end (hours), the HAPNEST
  real-LD path (opt-in, multi-GB), or comparison against the original R
  LTFHPlus outputs numerically (the covariance grammar was diffed against the
  R source textually instead).
