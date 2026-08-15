# ltpred — independent review and future developments

> **Archived review snapshot.** This document records the v0.3.0 tree described
> below; its file locations, test counts, provenance status, and open findings
> are historical. Use the current guide, roadmap, changelog, and benchmark report
> for present behavior.

Date: 2026-08-07
Reviewer: independent review (requested by the project owner), of the working
tree at commit `0e4b59b` (v0.3.0). This review supersedes the 2026-07-19 review
of commit `673bb99` (retained in git history). Method: full test suite run
locally (576 passed, 1 conditional skip, 179 s, Python 3.14.6); `ruff check`
clean; the docs fence/anchor checker run (fence check passed; the
`mkdocs build --strict` half could not run locally — watchdog has no cp314
wheel — and is covered by CI on Python 3.12); a complete static review of all
16 package modules, the `research/` package, all test files, and all user/theory
docs; every headline number in RESULTS.md mechanically re-derived from the
committed CSVs and local run logs; every defect reported below confirmed by
direct first-hand inspection of the cited lines before inclusion.

---

## 1. Verdict

**ltpred 0.3.0 remains unusually careful scientific software, and the 0.3.0
leanness refactor addressed essentially every actionable finding of the
previous review.** All five previously confirmed defects are fixed, with
regression tests that explain the old bug in comments: the `add_ind=False`
cross-trait bug, the pinned-initialization saturation, the missing PSD/symmetry
guard on the public Gibbs paths, the missing input guards, and the silent
PD-correction paths. The previous review's entire Tier-1/Tier-2 programme has
been executed: the PA-FGRS mixture now has generative validation, a
model-misspecification benchmark exists, the inference machinery has a
calibration benchmark, and the headline grids are multi-seed. The test suite
has grown from 324 to 577 tests.

The new findings are concentrated in two places. First, a residue of
**unguarded invalid inputs that silently produce wrong answers** — negative
`burn_in`, unvalidated PA bounds, duplicate `phen_names` — in a package whose
stated policy is to raise instead; each is small and well-specified. Second,
and now the repository's largest honesty gap: **RESULTS.md was only partially
refreshed after the 2026-07-31/08-01 benchmark re-campaign, and roughly a third
of its quantitative tables do not match the stored artifacts** — including one
reversed qualitative direction (the register-pipeline leakage contrast) and one
sign flip (the genetic-correlation null bias). The load-bearing claims
(PA–Gibbs agreement, GWAS power ratio, the integrated LT-FH++ panel, the
confounding grid, the timing re-baseline) all re-derive exactly; the skew is in
the fitter/pedigree/pipeline tables. The project remains one hardening pass —
this time mostly a documentation-artifact reconciliation — away from being the
reference Python implementation of LT-FH++.

## 2. Theory documentation

`docs/algorithm.md` remains the strongest document in the repo. Every
derivation that could be checked was re-verified and is correct: the
liability-threshold setup and `A`-scaled covariance; the kinship tabular method
with the inbreeding standardization `h2 A_tt/(1+h2(A_tt−1))`; the PA rank-1
updates; the censoring-mixture weights with the lifetime-threshold split; the
BLUP conditioning identity; the nurture path-model algebra and its closed-form
inversion; the sex-limitation PSD proof; the PGS–FH correlation and joint-R²
formulas; the Haseman–Elston updates; the Greenwood and Aalen/`cmprsk`
variance recurrences (the worked Aalen–Johansen example in cip-estimation.md
recomputes exactly, all six rows); the factor-model degrees-of-freedom formula.
Theory-vs-code consistency post-refactor is essentially perfect — every
0.3.0 demotion and signature change is reflected in the docs. The errors found
are bibliographic, not mathematical:

1. **Wrong Sham book title** — `algorithm.md:404-405` and the reference list
   cite "Sham, *Statistical Methods in Genetic Epidemiology*, 1998 (Oxford)".
   Sham's 1998 book is ***Statistics in Human Genetics*** (Edward Arnold).
   *Statistical Methods in Genetic Epidemiology* is Duncan C. Thomas's 2004
   Oxford book.
2. **Merged/misdated threshold-model citation** — `algorithm.md:1001`:
   "go back to Wright, Dempster & Lerner (1950)" implies a joint 1950 paper
   that does not exist. Wright's threshold-model paper is 1934 (*Genetics*);
   Dempster & Lerner 1950 is correctly listed separately. Wright has no
   reference-list entry.
3. **Dangling citation** — `algorithm.md:73`: "The LT-FGRS pipeline (Pedersen
   et al. 2026)" has no reference-list entry (the list has Pedersen et al.
   2025, *Front Genet*, for graph extraction; an LTFGRS package by E. M.
   Pedersen exists, but the year is unresolved).
4. **Formula scope imprecision** — `algorithm.md:806`: "The full pedigree
   covariance is `G ⊗ A + E ⊗ I`" is exact only for the observed
   full-liability block; the appended target genetic coordinate has
   `Var(g^p) = h2_p` and same-person `Cov(g^p, g^q) = G[p,q]`
   (`covariance.py:456-462`). The shorthand should be scoped.

Externally verified and correct: PA-FGRS = Dybdahl Krebs et al. 2024 *AJHG*;
the 2026 *AJHG* PA-FGRS+PGS paper; Zhuang et al. 2022 *Bioinformatics*
btac459; Aitken 1935 (and the explicit contrast with the GLS paper); Hujoel
2020/2022; Pedersen 2022/2023; So–Kwan–Cherny–Sham 2011; Hazel 1943;
Henderson 1975; Aalen 1978. Could not verify verbatim (stated honestly, not
counted as errors): the LT-FH++ Aalen–Johansen quotation, the PA-FGRS
supplement equation numbering S3–S5, and the Dybdahl Krebs 2026 quotation.

**Documentation gaps, ordered by importance:**

1. **The PA-FGRS mixture's onset-timing approximation is never stated as an
   assumption.** The weight `(K_pop − K_i)/K_pop` implicitly assumes onset
   timing among future cases follows the population CIP curve independent of
   liability (higher-liability future cases in fact onset earlier), and
   inherits non-informative censoring given stratum. `algorithm.md:659-683`
   presents the formula without flagging this; `assumptions.md`'s censoring
   discussion covers CIP *estimation* but is never tied back to the mixture.
2. **The pinned-case encoding's modeling weight is under-discussed.** Pinning
   makes a case's liability a deterministic function of onset age (zero
   conditional variance) and hard-codes the CIP curve as the liability–onset
   map — error-free onset dates, homogeneous severity. The benchmarks show the
   case encoding dominates calibration; no theory doc says what pinning
   *assumes* or when an interval case is safer.
3. **Sequential-PA order dependence is not documented.** The approximation
   error depends on fold order; the implementation canonicalizes to sorted
   role order for determinism (`estimate.py:505-507`) — a reproducibility
   choice, not an accuracy-motivated one. `estimation.md`'s advice to
   cross-check Gibbs on unusual pedigrees partially covers this, but the
   order-dependence itself is never stated.
4. Minor: the extreme-prevalence numerical regime (far-tail quadrature,
   log-space CDF, `min_cip` clipping) is invisible in the theory docs —
   relevant for rare traits with thresholds beyond ~4 SD.

## 3. Implementation review

### 3.1 Previous review's findings — all closed

Verified first-hand against the current tree:

1. ~~`covariance.py` cross-trait `add_ind=False` bug~~ — **fixed**
   (`covariance.py:460-462`, now keyed on the role label with a comment naming
   the old bug; regression tests `test_covariance.py:254-284` pin both
   directions).
2. ~~`fit.py` pinned-initialization saturation~~ — **fixed** (`fit.py:155,330`
   initialize via the sampler's `_init_chain`, routing through the tail-safe
   `_std_tnorm_quantile`; regression test `test_fit.py:391-412` holds a pin
   at z=9).
3. ~~No PSD/symmetry guard on public Gibbs paths~~ — **fixed**
   (`gibbs.py:168-193` `_validate_covmat`: symmetry and strict PD via
   scale-relative `eigvalsh` tolerances; called from `gibbs_params` and from
   `rtmvnorm_gibbs` **including the precomputed-params bypass**,
   `gibbs.py:538-539`; tests `test_gibbs.py:314-348`).
4. ~~Missing input guards~~ — **fixed** (`fit_heritability` rejects
   `burn_in ≥ n_iter` and `h2_init ∉ [0,1]`, `fit.py:194-197`; `batch_means`
   guards `n ≥ 4`, `estimate.py:176-179`; `validate_mixture_inputs` rejects K
   on pinned rows, `_validation.py:95-99`; the single-trait Gibbs path raises
   on multi-trait bounds with a helpful pointer, `estimate.py:416-420`).
5. ~~Inconsistent PD-correction reporting~~ — **fixed**
   (`_warn_if_corrected`, `estimate.py:335-346`, fires on all six estimator
   paths; tests `test_estimate.py:548-573`).
6. ~~`benchmarks/README.md` PA-FGRS overstatement~~ — **fixed** (the mixture
   benchmark `bench_pafgrs_mixture.py` now exists and the README text is
   accurate).

Carried forward (still open):

- **The same-side half-sib convention** (two `mhs`/`phs` related 0.5·h² *to
  each other* — an implied shared second parent, inherited from LTFHPlus)
  remains **undocumented and untested** (`covariance.py:170-171,182-183`). Its
  stakes have risen: `construct_covmat_from_kinship`'s docstring
  (`covariance.py:550-552`) claims "for a standard pedigree the two agree
  entry for entry", which is **false** for a pedigree with two same-side
  half-sibs whose other parents differ (grammar: 0.25; pedigree with distinct
  mothers: 0.0 — the grammar cannot express that pedigree). Two documented
  paths silently diverge on a common pedigree shape. Fix: document the
  convention, scope the claim, add the test.
- ROADMAP's "23×" `h2_se`-gap figure vs the benchmark's across-cohort range —
  still quoted as a single-dataset number (ROADMAP.md) while RESULTS.md §5's
  table itself no longer matches its CSV (see §5 below).

### 3.2 New defects (confirmed first-hand; all edge-located, silent-wrong-input class)

Ordered by severity. None affects a supported call path with valid inputs.

1. **Negative `burn_in` silently corrupts every Gibbs path.**
   `gibbs.py:235` (`_gibbs_sweep`) and `gibbs.py:285`
   (`_gibbs_estimate_batched`) loop `for k in range(-burn_in, n_sim)`; with
   `burn_in = −m` the loop starts at `k = m > 0`, so `rtmvnorm_gibbs` returns
   its first m rows as unwritten `np.empty` garbage, and the batched kernel
   accumulates only `n_sim − m` draws while `_estimate_group`
   (`estimate.py:264-266`) credits `n_sim` — biased means and wrong batch-means
   SEs. The fitters slice `trace[int(burn_in):]` (`fit.py:226,451`), so a
   negative `burn_in` silently selects a trailing window as "post-burn-in".
   `rtmvnorm_gibbs` merely `int()`-coerces `burn_in` (`gibbs.py:586-587`); no
   non-negativity check exists anywhere. Every other numeric input on these
   APIs is meticulously validated; this one produces silent wrong answers.
2. **The public PA entry points skip bound (and covariance) validation.**
   `pa_algorithm` (`pearson_aitken.py:387-401`) and `pa_estimate_batched`
   (`:404-432`) never call `validate_bounds`, unlike `rtmvnorm_gibbs`
   (`gibbs.py:547`) and the Family-layer PA paths. NaN bounds propagate
   silently; reversed bounds wider than 1e-3 yield NaN via
   `log(-expm1(positive))` (`:201`), while a reversal *narrower* than 1e-3 is
   silently "sorted" by the narrow-interval quadrature (`:187-188`). A non-PD
   covmat drives a fold diagonal ≤ 0, which raises under the pure-Python
   fallback but returns NaN under Numba — a backend inconsistency. This is
   precisely the failure class 0.3.0 fixed on the Gibbs side.
3. **Duplicate `phen_names` silently collapse multi-trait output.**
   `_estimate_liability_multi` builds `col_names` straight from `phen_names`
   (`estimate.py:565-566`) and allocates the result dicts keyed by name
   (`:570-571`); the fill loop (`:603-605`) writes both phenotypes into the
   same key, last write wins — a valid-looking `LiabilityResult` with one
   phenotype silently lost. A one-line uniqueness guard is missing.
4. **A member with role `"g"` silently conditions the genetic coordinate.**
   Nothing rejects `"g"` in user-supplied roles (`family.py` has no role
   validation; `_check_unique_roles` only catches duplicates;
   `_expand_family` strips and re-adds `"g"`/`"o"`, `covariance.py:252-256`).
   The role lookups (`estimate.py:204-210,625-629`) then map that member's
   bounds onto the genetic liability — a silent model change on pathological
   input.
5. **Covariance-constructor robustness gaps** (public API, below the estimator
   front doors, which do reject duplicates): duplicate singleton roles
   (`fam_vec=["m","m"]`, `covariance.py:251-256`) build a singular matrix with
   two perfectly-correlated "mothers"; an `n_fam` count > 1 for a singleton
   role is silently dropped (`:245-249`).
6. **No prevalence guards in the simple threshold helpers.**
   `liability_threshold(0)` = +inf, `(1)` = −inf; a K=0 "case" pin at
   `(inf, inf)` passes `validate_bounds` and flows into the sampler; the
   logistic helpers accept `pop_prev` outside (0, 1) → silent NaN thresholds.
   `thresholds_from_cip` is properly guarded; these are not.
7. **Mixture kernel edge `lower == split`** (`pearson_aitken.py:291-293`): a
   row whose lower bound exactly equals the lifetime threshold collapses the
   genuine-control component to a point and forces the future-case component
   to `(0, 0)` — wrong moments. Unreachable from `pa_thresholds`; pathological
   direct calls only.
8. **`research/` package:** the research fitters
   (`fit_variance_components_mcem`, `fit_genetic_correlation`,
   `fit_genetic_correlation_decay`) lack the `sampling="population"` contract
   gate the core added in 0.3.0 despite embedding the identical
   population-sampling assumption. `_decay_negq_grad`
   (`research/advanced_fitting.py:559-561`) returns a **zero gradient** in the
   non-PSD penalty region — a stationary point for L-BFGS-B, so an M-step
   landing there terminates instead of walking back to feasibility (latent:
   M-steps start from the previous PSD-projected iterate). Dead `rp`
   parameter in `_decay_cov` (`:395`).
9. **Doc/comment nits in code:** `estimate.py:772` still claims PA "ran
   315–510x faster" — the superseded 10-thread figure, contradicting
   RESULTS.md ("quote it with its thread count or not at all") — and this
   docstring feeds the published API reference via mkdocstrings.
   `pedigree.py:22` says mates are "distance 3 via the child"; the proband's
   own mate is graph distance 2 (the code and tests agree on 2).

The verified-clean list is long and worth stating: the PA rank-1 fold algebra,
the exact Schur-complement pinned-point limit, fold-order canonicalization,
the three-regime tail-moment machinery, the mixture weight formula and its
0/0 guard, the precision-matrix Gibbs conditionals, survival-scale quantile
interpolation with the far-tail Rayleigh handoff, streaming batch-means
pooling across rounds, per-family seed blocks (thread-scheduling
independence), the chunked-uniform advance path, the kinship tabular method
(verified against a brute-force recursive implementation on 40 random inbred
pedigrees — exact match), the env-component PSD bank (300 random role-set
sweep — all PSD), the tetrachoric MLE, and the Aalen (1978) variance
recurrence (diffed statement-by-statement against `cmprsk`'s Fortran,
including the tie factors and the exhausted-risk-set boundary).

## 4. Computational efficiency

**Architecture (verified, sound).** The core is Numba-JIT'd with `prange`
over families grouped by canonical structure; one O(d³) precision
factorization per structure group is reused across all families and all
Gibbs rounds (`estimate.py:238`). PA uses active-block rank-1 updates
(⅓–½ the naive work). Streaming batch-means keep the SE state at O(ncols)
per family. Uniforms are pre-generated in 8 MiB chunks outside `prange`,
buying exact scheduler independence at small RNG overhead. The array APIs
skip Python objects entirely and keep float32 bounds (halving the big
`(F, d)` inputs) while accumulators stay float64. The 0.3.0 timing re-baseline
is honest: 203–492× (PA object path vs Gibbs, no-mixture, **4 threads**, load
~7–11 recorded in the run manifest), with the earlier 10-thread 315–510×
figure explicitly retired as non-reproducing; the array API sustains
1.18–3.79M already-aligned families/s hot, 20.6–52.3× over the object path.
(All re-derived from `bench_scaling.csv`; see §5.)

**Optimization opportunities, ranked by leverage:**

1. **Halve the PA moment evaluations** (small, free). `_pa_family_nomix`
   (`pearson_aitken.py:349-351`) evaluates `_std_tnorm_moments(a, b)` twice
   per fold with identical arguments (via `_tnorm_mean` and `_tnorm_var`);
   `_tnorm_mixture` (`:289-296`) makes 4 kernel calls where 2 suffice. One
   call scaled into both outputs roughly halves the moment-evaluation cost
   that dominates the PA sweep at small/moderate d.
2. **Chunked/streaming driver** (roadmap #9). The real biobank enabler —
   memory, not speed: today a run holds all aligned families in RAM.
3. **Unit-stride Gibbs conditionals.** The inner loop reads `P[:, j]`
   column-wise against C-contiguous storage (`gibbs.py:137-138,221`); storing
   `P` transposed makes the hot loop unit-stride. Irrelevant at `d ≲ 10`;
   matters for multi-trait/pedigree `d`.
4. **O(n) topological order in `kinship_from_pedigree`.** The current
   repeated-scan sort is O(n²) worst case (`covariance.py:513-523`); Kahn's
   algorithm is O(n). Fine at extracted-pedigree scale; avoid at population
   scale.
5. Document, don't fix: the fitters' per-iteration re-factorization of `sigma`
   (`fit.py:216-217,438`) is inherent to the fixed-point design and is the
   dominant non-sampling cost for many-group fits; the object PA path runs one
   full sweep per requested output column (`estimate.py:529-535`), unavoidable
   given per-target reordering; `bootstrap_fit` is deliberately serial.

## 5. Benchmark and evidence review

Every headline claim in RESULTS.md was mechanically re-derived from the
committed CSVs (19 files) and the local `run_logs/`; the four most substantive
discrepancies were additionally confirmed first-hand for this review.

**The load-bearing claims reproduce exactly:**

- PA–Gibbs grid agreement 0.9977–0.9999 (27 cells, 5 seeds) and ≥0.9984 on
  stress pedigrees; the 1.47 ± 0.04× causal-SNP NCP ratio (both engines);
  the §15 integrated panel **bit-for-bit from the 70 committed replicate
  rows** (ADuLT 1.0490 ± 0.0043, LT-FH++ 1.1944 ± 0.0063, paired increment
  +0.14540 ± 0.01543, calibration slope 0.9952 ± 0.0153, sex-CIP gap
  0.05092 ± 0.00096); the confounding grid (cohort-blind λ_GC up to
  16.460 ± 0.399, cohort-aware 0.960–1.007); the scaling tables (Gibbs
  295–305 f/s, PA 71k–78k f/s, array 1.18–3.79M f/s; object-path ratio
  202.9–491.6× at 4 threads, median of 5 timing reps, load provenance in the
  manifest). The two local 10-thread re-runs (102–242×) corroborate the
  retraction of the old 315–510× figure.
- Old evidence gaps genuinely closed: the **mixture benchmark** (§16) runs
  mixture arms under two observation models — near-exact under its native
  stochastic-onset model (slopes 0.997/0.988), slope improvement under
  threshold-crossing (1.199→1.174 MID), correlation differences ≤ 0.0005 —
  and the **misspecification** (§18) and **inference-calibration** (§17:
  Type-I 0/25, coverage 24/25, SE/SD ≈ 0.93) benchmarks exist. Replicate
  counts improved broadly (accuracy/calibration 5 seeds, GWAS 3, integrated
  panel 10 + 5).

**But roughly a third of the tables are stale relative to the stored
artifacts.** RESULTS.md states it was "assembled from focused runs through
2026-07-30" (`RESULTS.md:12`); every stored CSV hash-matches a run in the
2026-07-31/08-01 campaign manifest. The tables were transcribed from an
earlier campaign and only partially refreshed. Confirmed mismatches
(computed-from-artifact vs quoted):

| § | Item | Stored artifact | RESULTS.md |
|---|---|---|---|
| 3 | age-onset rows | 0.4275/0.4297/1.0102 | 0.4328/0.4349/1.0097 |
| 5 | h²=0.6 fitter bias | **−0.0230** | −0.004 |
| 7 | r_g=0 null bias / SD (25 reps) | **−0.0061 / 0.0798** | +0.0068 / 0.0688 |
| 7 | N-sweep SDs (1k/2k/4k/8k) | 0.148/0.108/0.078/0.042 | 0.124/0.099/0.064/0.037 |
| 8 | shared-env table | wholesale different | quotes older run; its "predates MCSE instrumentation" caveat is no longer true — the CSV now carries MCSE columns |
| 9 | couple-env M at m²=0.3 | 0.2856 | 0.301 |
| 10 | ascertainment eff-N SE | 2.212 ± **0.015** | 2.212 ± 0.127 (8× off) |
| 12 | extended-pedigree K=0.01 slope | 0.9892 ± 0.0266 | 0.954 ± 0.036 |
| 14 | densely-affected corr | 0.4893/0.4902 | 0.4657/0.4665 (p95 ordering also flipped) |
| 20 | pedigree payoff | **0.5318 vs 0.4521 (+17.6%)** | 0.618 vs 0.559 (+10.6%) |
| 21 | register pipeline: leaking relatives' post-index events | **0.1689 → 0.1481 (a decrease)** | 0.090 → 0.132 ("+0.04") — **direction reversed** |
| 25 | decay data-requirement row | n_fam=1000: 0.578/0.063 | "n~1200": 0.618/0.064 |
| 26 | sex-limitation replicate count | **reps=3** (every CSV row) | "5 replicates" |
| 27 | nurture CIs | ±0.0008/±0.0010/±0.0025 | ±0.0006/±0.0007/±0.0017 |

In every case examined, the qualitative conclusion survives (the §7 null
still shows no spurious r_g on average; §12's "~3 SE below 1" still holds at
z = 3.0) — **except §21, where the stored logs show the leakage contrast in
the opposite direction from the published text** (both stored runs agree:
honest 0.1689, +relatives' events 0.1481, +own outcome 0.3985). Individual
numbers should not be cited from RESULTS.md without re-deriving them.

**Traceability gap:** sections 16–24 (mixture, inference calibration,
misspecification, CIP, pedigree, pipeline, tetrachoric, liability scale,
env wiring) come from stdout-only scripts whose output survives only in
`benchmarks/run_logs/` — which is **git-ignored**, as is
`run_manifest.jsonl`, the only machine-readable provenance record. The
committed repo therefore offers CSVs with no provenance and nine sections
resting on uncommitted evidence.

Remaining estimand caveats (all disclosed somewhere, collected here): the
mixture benchmark retains no paired uncertainty, so "no correlation cost" is
not established (means of 5 reps only); `bench_pa_robustness` remains
single-seed (labeled); three-replicate SEs are coarse (labeled);
`test_genetic_correlation`'s parametric-bootstrap test has **no** null
calibration anywhere (only `test_variance_component("C")` does, §17).

## 6. Testing

576 passed, 1 conditional skip, 179 s. Strengths remain where they matter:
algebraic identities, tail regimes, thread independence, role/kinship
equivalence, regression tests that document the bug they pin. Gaps found this
pass:

- The streaming batch-summaries are checked against the analytic posterior
  for one family, one round; the **multi-round pooled reconstruction** in
  `_estimate_group` is never compared to offline `batch_means` on concatenated
  draws — the natural algebraic test, currently absent.
- No tests for the §3.2 defects (negative `burn_in`, PA bound rejection,
  duplicate `phen_names`, `"g"` as a member role, duplicate singleton roles).
- No test with a family **lacking an `o` member** on any engine; multi-trait
  `out=("full",)` never exercised; the `gibbs_out` offset arithmetic for
  `full` untested.
- Same-side half-sib pair relatedness (0.5·h²) untested; grammar↔kinship
  equivalence tested only on a half-sib-free pedigree.
- CIP: no hand-computed oracle combining left truncation + competing risks +
  ties in one dataset (each covered separately); Greenwood/Aalen SEs have no
  simulation calibration in the unit tests.
- `research/`: only the decay M-step's analytic gradient is
  finite-difference-pinned (`research/tests/test_decay.py` — excellent); the
  MCEM M-step and MINRES gradients have no FD test. The parametric-bootstrap
  tests are tested for power, not size. The GHK likelihood is never checked
  against a brute-force rectangle probability. The pipeline censoring test
  re-implements the censoring rule inline — a regression pin, not an
  independent oracle.
- `examples/*.py` are verified accurate today but are executed by neither
  tests nor CI.

## 7. Future developments — suggested, prioritized

**Tier 1 — correctness and trust (days, do first).**

1. **Add the missing input guards with regression tests** (§3.2, items 1–6):
   non-negative `burn_in` on all Gibbs and fit paths; `validate_bounds` on the
   public PA entry points; `phen_names` uniqueness; reject `"g"`/`"o"` in
   user member roles; reject or honor duplicate singleton roles and
   `n_fam` singleton counts; prevalence ∈ (0, 1) in the simple helpers. Each
   is a few lines plus a test, in the project's existing validation idiom.
2. **Reconcile RESULTS.md with the stored artifacts** — re-derive every table
   from the committed CSVs (or re-run the stale sections), fixing above all
   the §21 reversed leakage direction, the §7 null sign, the §20 payoff, the
   §26 replicate count, and the §8 stale MCSE caveat. **Commit
   `run_manifest.jsonl` and the referenced run logs** (or CSV-ify the
   stdout-only benchmark scripts) so every section has committed, checkable
   evidence. This restores the repo's own standard — "every quoted number
   re-derives" — which held at the last review and no longer does.
3. **Fix the four bibliographic errors** (§2), the stale `estimate.py:772`
   speed figure, the `pedigree.py:22` mate-distance nit, and document the
   half-sib 0.5·h² convention with the kinship-divergence caveat
   (`covariance.py:550-552` claim scoped).

**Tier 2 — evidence hardening (days to weeks).**

4. State the mixture's onset-timing assumption and the pinned-encoding
   assumptions in `algorithm.md`/`assumptions.md`; retain paired uncertainty
   in `bench_pafgrs_mixture.py` so the "no correlation cost" claim is
   established rather than asserted.
5. Close the §6 test gaps — starting with the multi-round streaming
   batch-means identity, the `out=("full",)` and no-`o`-member paths, FD pins
   for the MCEM and MINRES gradients, and a null-calibration artifact for
   `test_genetic_correlation`.
6. Add an examples-execution smoke test (both `examples/*.py` run in CI) to
   keep the verified-accurate examples that way.

**Tier 3 — capability (re-ordered by value per effort).**

7. **Graduate `research/covariance_extensions` and `research/pipeline` into
   the core.** Both are validated, benchmarked, and essentially done;
   graduation means wiring a user-supplied covariance/pipeline into the
   supported estimation path plus docs. `fit_nurture` follows. (The previous
   review's "wire fitted C/M into `estimate_liability`" is **done** — the
   0.3.0 `c2`/`m2` arguments and the §24 env-wiring benchmark, calibration
   slope 0.927 → 0.986.) Graduation order for the rest, by readiness:
   `pipeline` > `fit_nurture` > `fit_genetic_correlation`/`fit_genetic_factor`
   > the parametric-bootstrap tests > MCEM > the decay fitter (correctly
   parked deepest: data-hungry, ridge-dominated at moderate n, its
   `shared_env` remediation claim unbenchmarked).
8. **Efficiency:** the PA moment-evaluation halving (§4.1) — days, free —
   then the chunked/streaming driver (roadmap #9), which is the biobank-scale
   enabler.
9. **PyPI release** (roadmap #11): only the one-time trusted-publisher
   registration remains; CI, RELEASING.md, and CITATION.cff are verified
   consistent and current.
10. **Multi-trait PA** (#12), gated on a PA-vs-Gibbs accuracy benchmark
    before shipping; **sparse single-pedigree** (#8); the **HAPNEST real-LD
    end-to-end demo** (#10) as the final demonstration.

## Appendix — verification performed

- Read first-hand: README, ROADMAP, algorithm.md (full), estimation/inference/
  assumptions/cip-estimation docs; all 16 `ltpred/` modules in full via
  delegated static review with every reported defect re-confirmed by direct
  inspection of the cited lines; `research/` package and its tests.
- Ran: the full test suite in the project conda env (`ltpred314`,
  Python 3.14.6): **576 passed, 1 skipped (conditional), 179.25 s**;
  `ruff check .` — clean; `scripts/check_docs.py` — fence check passed,
  mkdocs half not runnable locally (no watchdog wheel for cp314; the docs
  job covers `mkdocs build --strict` on Python 3.12 in CI).
- Re-derived mechanically: every RESULTS.md headline against the 19 committed
  CSVs (csv+numpy; no pandas in the env) and the git-ignored `run_logs/`;
  the four most substantive mismatches (§7 null sign, §20 payoff, §21 leakage
  direction, §26 replicate count) independently recomputed first-hand.
  Verified `run_manifest.jsonl`/`run_logs/` git-ignore status and the
  RESULTS.md snapshot date against artifact timestamps.
- Verified algebraically by hand (delegated, spot-checked): PA rank-1 updates,
  mixture weights, HE normalizations, MCEM M-step gradient, OPG score, GHK
  recursion, decay block gradients, MINRES gradient, nurture path model,
  sex-limitation PSD argument, Aalen variance recurrence vs `cmprsk` Fortran.
- Not done: running the benchmark scripts end-to-end (hours), the HAPNEST
  real-LD path (opt-in, multi-GB), numerical comparison against the original
  R LTFHPlus outputs, and the mkdocs strict build locally (see above).
